import os
import time
import pickle
import re
# import random
from glob import glob

import requests
import shutil
import pandas as pd
from sqlalchemy import create_engine
# from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.common.exceptions import NoSuchElementException, TimeoutException
# from selenium.webdriver.common.proxy import Proxy, ProxyType
from proxy_ferret import ferret
from scraper.scr import Scraper
from multiprocessing import Process
# from multiprocessing import Manager


HEADLESS = False
num_workers = 1
proxer = ferret.Ferret()
# proxer.check_proxies(wait_limit=len(proxer.new_proxies))
# check all the things twice
proxer.check_proxies(wait_limit=len(proxer.new_proxies),
                     check_url="http://auto.ru")
proxer.check_proxies(wait_limit=len(proxer.new_proxies),
                     check_url="http://auto.ru")

while len(proxer.new_proxies) < num_workers:
    proxer.get_new_proxies()
    proxer.check_proxies(wait_limit=len(proxer.new_proxies),
                         check_url="http://auto.ru")
    proxer.check_proxies(wait_limit=len(proxer.new_proxies),
                         check_url="http://auto.ru")

pcl_name = "auto-ru/auto_ru_ads.pcl"
all_ads = pickle.load(open(pcl_name, "rb"))
# manager_dict = Manager().dict()
done_links = pickle.load(open("auto-ru/auto_ru_ads.pcl", "rb"))
# manager_dict.update(done_links)
with open("auto-ru/auto_ru_regions") as f:
    regions = f.readlines()
regions = [l.strip() for l in regions]
regions.append("")


class AutoRuAdScraper:
    def __init__(self, external_ads_dict=None, engine="pickle", page=1,
                 worker_id=None, new_only=True, scrape_regionwise=False):
        """
        engines: (pandas, sqlite, pickle)
        """
        self.proxy = None
        self.page = page
        self.region = ""
        self.region_counter = -1
        self.captcha_id = 0
        self.scrape_regionwise = scrape_regionwise
        if new_only:
            self.new_only = "new"
        else:
            self.new_only = "all"
        # "voronezhskaya_oblast/"
        self.home_link = f"https://auto.ru/{self.region}cars/{self.new_only}/"\
            f"?sort=fresh_relevance_1-desc&output_type=list&page={self.page}"
        self.parsed_number = 0
        self.captcha = False
        if type(worker_id) == int:
            worker_id = str(worker_id)
        else:
            worker_id = ""
        self.folder = "auto-ru/"
        self.common_pcl = f"{self.folder}auto_ru_ads.pcl"
        if worker_id:
            worker_name = f"_{worker_id}"
        else:
            worker_name = ""
        self.pcl_name = f"{self.folder}auto_ru_ads{worker_name}.pcl"
        self.pandas_name = f"{self.folder}auto_ru_pandas.csv"
        self.scraper = None
        self.done_proxies = set()
        self.engine = engine
        self.done_links = set()
        self.all_ads = dict()
        if self.engine == "pandas":
            self.done_links = set(pd.read_csv(self.pandas_name)["keys"].values)
        elif self.engine == "sqlite":
            self.sql_engine = create_engine('sqlite:///auto-ru/auto_ru.db')
            self.done_links = pd.read_sql("auto-ru", self.sql_engine)["keys"]
        elif external_ads_dict:
            self.all_ads = external_ads_dict
        else:
            if os.path.exists(self.pcl_name):
                self.all_ads = pickle.load(open(self.pcl_name, "rb"))
                self.done_links = pickle.load(open(self.common_pcl, "rb"))
                self.done_links = set(self.done_links.keys())
            else:
                self.all_ads = dict()

    def init_scraper(self, proxy=None):
        """
        relaunch selenium until it works
        """
        content = None
        i = 0
        while not content:
            if self.scraper:
                self.scraper.browser.close()
            self.scraper = None
            if proxy or self.proxy:
                proxy = self.get_proxy()
            content = self.try_scraper(proxy=proxy)
            i += 1

    def detect_captcha(self):
        try:
            captcha_header = self.scraper.browser.find_element_by_xpath("//h1")
            if captcha_header.text == "ой…":
                self.captcha = True
                return True
        except NoSuchElementException:
            return False

    def try_scraper(self, proxy=None):
        self.scraper = Scraper(headless=HEADLESS, proxy=proxy)
        self.scraper.browser.set_page_load_timeout(60)
        self.scraper.browser.get(self.home_link)
        time.sleep(10)
        # accept terms
        try:
            confirm_button = self.scraper.browser.find_element_by_xpath(
                "//div[@id='confirm-button']")
            confirm_button.click()
            time.sleep(60 * 5)
        except Exception as ex:
            print("no confirm button")
        # close pop up
        try:
            promo_close_button = self.scraper.browser.find_element_by_xpath(
                "//div[@class='ModalDialogCloser_color-white "
                "ModalDialogCloser_size-s "
                "ModalDialogCloser PromoPopupHistory__closer']")
            promo_close_button.click()
            time.sleep(2)
        except Exception as ex:
            print(ex)
            pass
        try:
            region = self.scraper.browser.find_element_by_xpath(
                "//div[@class='GeoSelect__title']")
            region.click()
            time.sleep(1)
        except Exception as ex:
            print("No region", ex)
            return
        try:
            my_region = self.scraper.browser.find_element_by_xpath(
                "//button[@class='Button Button_color_blue Button_size_m "
                "Button_type_button Button_width_default "
                "GeoSelectPopupRegion']")
            my_region.click()
            time.sleep(1)
        except NoSuchElementException:
            pass
        try:
            save_button = self.scraper.browser.find_element_by_xpath(
                "//button[@class='Button Button_color_whiteHoverBlue "
                "Button_place_bottom "
                "Button_size_xl Button_type_button Button_width_full']")
            save_button.click()
            time.sleep(1)
        except NoSuchElementException:
            pass
        try:
            content = self.scraper.browser.find_element_by_xpath(
                "//div[@class='content']")
        except Exception as ex:
            print("content", ex)
            content = None
        if not content:
            captcha = self.detect_captcha()
            if captcha:
                print("captcha")
                time.sleep(5 * 60)
                self.recognize_captcha()
        return content

    def recognize_captcha(self):
        images = self.scraper.browser.find_element_by_xpath(
            "//img[@class='image form__captcha']")
        image_source = images.get_attribute("src")
        response = requests.get(image_source, stream=True)

        image_file = f"captcha_{self.page}{self.captcha_id}.jpg"
        self.captcha_id += 1
        if response.status_code == 200:
            with open(image_file, 'wb') as f:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, f)

    def get_seller_info(self, content):
        """
        get seller info from a pop-up
        """
        try:
            phone = content.find_element_by_xpath(
                "//div[@class='CardPhone-module__phone "
                "CardOwner-module__phone "
                "CardPhone-module__preview CardPhone-module__minimized']")
        except NoSuchElementException:
            try:
                phone = content.find_element_by_xpath(
                    "//div[@class='CardPhone-module__phone "
                    "CardOwner-module__phone "
                    "CardPhone-module__preview']")
            except NoSuchElementException:
                phone = content.find_element_by_xpath(
                    "//div[@class='CardPhone-module__phone "
                    "CardPhone-module__preview']")
        phone.click()
        time.sleep(2)
        address = None
        region_name = None
        metro_stations = None
        seller_name = content.find_element_by_xpath(
            "//div[@class='SellerPhonePopup-module__name']").text
        phone_number = content.find_element_by_xpath(
            "//div[@class='SellerPhonePopup-module__phoneNumber']").text
        phone_schedule = content.find_element_by_xpath(
            "//div[@class='SellerPhonePopup-module__phoneSchedule']").text
        try:
            # seller_address = content.find_element_by_xpath(
            #     "//span[@class='MetroListPlace MetroListPlace_showAddress "
            #     "SellerPhonePopup-module__place']")
            seller_address = content.find_element_by_xpath(
                "//span[@class='MetroListPlace MetroListPlace_ellipsis "
                "SellerPhonePopup-module__place']")
        except NoSuchElementException:
            seller_address = None
        metro_stations = []
        if seller_address:
            try:
                metro_stations = seller_address.find_elements_by_xpath(
                    "//span[@class='MetroList__station']")
                metro_stations = [m.text for m in metro_stations]
                metro_stations = [m for m in metro_stations if m]
            except NoSuchElementException:
                pass
            region_name = seller_address.find_elements_by_xpath(
                "//span[@class='MetroListPlace__regionName']")
            region_name += seller_address.find_elements_by_xpath(
                "//span[@class='MetroListPlace__regionName "
                "MetroListPlace_space']")
            region_name = [r.text for r in region_name]
            region_name = [r for r in region_name if r]
            if region_name:
                region_name = region_name[0]
            else:
                region_name = ""

            address = seller_address.find_elements_by_xpath(
                "//span[@class='MetroListPlace__address']")
            address += seller_address.find_elements_by_xpath(
                "//span[@class='MetroListPlace__address "
                "MetroListPlace_space']")
            address = [a.text for a in address]
            address = [a for a in address if a]
        if address:
            address = address[0]
        else:
            address = ""
        output = dict()
        output["address"] = address
        output["region"] = region_name
        output["metro"] = metro_stations
        output["phone_schedule"] = phone_schedule
        output["phone_number"] = phone_number
        output["seller_name"] = seller_name
        return output

    def parse_card_info(self, content):
        try:
            card_info = content.find_element_by_xpath(
                "//div[@class='CardInfoGrouped-module__CardInfoGrouped']")
            card_info_text = card_info.text.split("\n")
            # Remove 'Характеристики'
            card_info_text = card_info_text[1:]
            keys = card_info_text[0:len(card_info_text):2]
            values = card_info_text[1:len(card_info_text):2]
        except NoSuchElementException:
            try:
                card_info = content.find_element_by_xpath(
                    "//div[@class='CardInfo-module__CardInfo']")
                keys = card_info.find_elements_by_xpath(
                    "//div[@class='CardInfo-module__CardInfo__cell']")
                keys = [k.text for k in keys]
                values = card_info.find_elements_by_xpath(
                    "//div[@class='CardInfo-module__CardInfo__cell "
                    "CardInfo-module__CardInfo__cell_right']")
                values = [v.text for v in values]
            except NoSuchElementException:
                card_info = content.find_element_by_xpath(
                    "//div[@class='CardInfoGrouped']")
                keys = card_info.find_elements_by_xpath(
                    ".//div[@class='CardInfoGrouped__cellTitle']")
                keys = [k.text for k in keys]
                values = card_info.find_elements_by_xpath(
                    ".//div[@class='CardInfoGrouped__cellValue']")
                values = [v.text for v in values]
        keys = [k.lower() for k in keys]
        values = [v.lower() for v in values]
        for key_i, key in enumerate(keys):
            if key == "двигатель" and "/" in values[key_i]:
                engine_characteristics = values[key_i].split("/")
                for e_i, e in enumerate(engine_characteristics):
                    if "л.с." in e:
                        keys.append("мощность")
                    elif " л" in e:
                        keys.append("литраж")
                    else:
                        keys.append("тип топлива")
                    values.append(e.strip())
        card_info_text = dict(zip(keys, values))
        return keys, values, card_info_text

    def parse_comment(self, content):
        try:
            comment = content.find_element_by_xpath(
                "//div[@class='CardDescription-module__CardDescription__"
                "textInner_cut']").text
        except NoSuchElementException:
            try:
                comment = content.find_element_by_xpath(
                    "//div[@class='CardDescription-module__"
                    "CardDescription__text']")
                comment = comment.text
            except NoSuchElementException:
                comment = ""
        return comment

    def parse_complectation(self, content):
        complectation_dict = dict()
        try:
            try:
                # used cars
                complectation = content.find_element_by_xpath(
                    "//div[@class='CardComplectation-module__"
                    "CardComplectation "
                    "PageCard-module__Card__contentIsland']")
            except NoSuchElementException:
                complectation = content.find_element_by_xpath(
                    "//div[@class='CardOfferHeader-module__"
                    "Card__twoColumnsRight']")
            complectation_items = complectation.find_elements_by_xpath(
                "//div[@class='Treeview__container Treeview__container_"
                "collapsed']")
            for c_i, c in enumerate(complectation_items):
                key = c.text
                key = key.split("\n")[0]
                c.click()
                values = c.text
                complectation_dict[key] = values.split("\n")[2:]
                time.sleep(1)
        except NoSuchElementException:
            pass
        return complectation_dict

    def get_new_add_link(self, add):
        add.location_once_scrolled_into_view
        time.sleep(0.5)
        try:
            add.click()
        except Exception as ex:
            print("click add exception", ex)
            time.sleep(5)
            add.click()
        time.sleep(0.2)
        add.location_once_scrolled_into_view
        time.sleep(0.2)
        add_link_cont = add.find_elements_by_xpath(
            "//a[@class='Link']")
        add_link_cont = [a.get_attribute("href") for a in add_link_cont if
                         "cars/new" in a.get_attribute("href")][-1]
        time.sleep(0.1)
        return add_link_cont

    def parse_new_car(self, content):
        try:
            more_button = content.find_element_by_xpath(
                "//button[@class='Button Button_color_white Button_size_xl "
                "Button_type_button Button_width_full']")

            time.sleep(0.1)
            more_button.location_once_scrolled_into_view

            try:
                ads_num = more_button.text.replace("Показать ещё 10 из ", "")
                ads_num = ads_num.replace(" предложений", "")
                ads_num = ads_num.replace(" предложения", "")
                try:
                    ads_num = int(ads_num)
                except Exception as ex:
                    print(ex)
                    ads_num = 10
                ads_num -= 10
                clicks_num = ads_num // 10 + 1
            except Exception as ex:
                print(ex, "ads_num")
                clicks_num = 1
            # temporary
            clicks_num = min(1, clicks_num)
            for i in range(clicks_num):
                more_button.click()
                more_button.location_once_scrolled_into_view
                time.sleep(1)
        except NoSuchElementException:
            pass
        new_car_adds = content.find_elements_by_xpath(
            "//div[@class='CardGroupListingItem-module__container "
            "CardGroupListingItem-module__expanded']")
        new_car_adds += content.find_elements_by_xpath(
            "//div[@class='CardGroupListingItem-module__container']")
        new_add_links = []
        for add in new_car_adds:
            try:
                add_link = self.get_new_add_link(add)
                new_add_links.append(add_link)
            except Exception as ex:
                print(ex)
        new_add_links = self.filter_links(new_add_links, href_already=True)

        new_add_links = new_add_links[:5]

        for n_l in new_add_links:
            self.parse_auto_ru_add(n_l)

    def parse_price(self, content):
        try:
            # new cars
            price_block = content.find_element_by_xpath(
                "//div[@class='Price-module__caption "
                "CardSidebarActions__price-caption']")
        except NoSuchElementException:
            try:
                price_block = content.find_element_by_xpath(
                    "//div[@class='Price-module__container "
                    "Price-module__container_interactive']")
            except NoSuchElementException:
                # used cars
                price_block = content.find_element_by_xpath(
                    "//div[@class='CardHead-module__price']")
        price_block_text = price_block.text.split("\n")
        for p_i, p in enumerate(price_block_text):
            price_block_text[p_i] = p.replace("Цена без скидок", "").strip()
        return price_block_text

    def parse_discount(self, content):
        try:
            discounts = content.find_elements_by_xpath(
                "//div[@class='DiscountList__item']")
            discounts = [d.text.split("\n") for d in discounts]
            max_discount = content.find_element_by_xpath(
                "//div[@class='DiscountList__item DiscountList__itemTotal']")
            max_discount = max_discount.text.split("\n")[1]
        except NoSuchElementException:
            discounts = []
            max_discount = ""
        return discounts, max_discount

    def parse_specification(self, content):
        specification_variants = [
            "//a[@class='Link SpoilerLink CardCatalogLink-module__"
            "CardCatalogLink']",

            "//a[@class='Link SpoilerLink CardCatalogLink-module__"
            "CardCatalogLink SpoilerLink_type_default']",

            "//a[@class='Link SpoilerLink CardCatalogLink-module__"
            "CardCatalogLink SpoilerLink_type_default']",

            "//a[@class='Link SpoilerLink CardCatalogLink "
            "SpoilerLink_type_default']"
        ]
        specification_link = None
        for s in specification_variants:
            try:
                specification_link = content.find_element_by_xpath(
                    "//a[@class='Link SpoilerLink CardCatalogLink-module__"
                    "CardCatalogLink']")
                break
            except NoSuchElementException:
                pass
        if specification_link:
            specification_link = specification_link.get_attribute("href")
            specification_link = re.sub("\?sale_id=.*", "", specification_link)
        return specification_link

    def parse_auto_ru_add(self, l, parsed_number=0, save_every_n=25):
        """
        parse an add
        """
        try:
            self.scraper.browser.get(l)
        except TimeoutException:
            time.sleep(10)
        time.sleep(5)
        try:
            # If the car is sold, return None
            self.scraper.browser.find_element_by_xpath(
                "//div[@class='CardSold']")
            self.all_ads[l] = dict()
            return None
        except NoSuchElementException:
            pass
        try:
            content = self.scraper.browser.find_element_by_xpath(
                "//div[@class='LayoutSidebar__content']")
        except NoSuchElementException as ex:
            print(ex)
            time.sleep(60 * 2)
            self.captcha = True
            return None
        try:
            title = content.find_element_by_xpath(
                "//div[@class='CardHead-module__title']").text
        # the car is new; a different interface is used
        except NoSuchElementException:
            # new cars
            self.parse_new_car(content)
        try:
            year = content.find_element_by_xpath(
                "//div[@class='CardHead-module__info-item']").text
        # the car is new; a different interface is used
        except NoSuchElementException:
            # new cars
            year = None
        price_block_text = self.parse_price(content)
        keys, values, card_info_text = self.parse_card_info(content)

        specification_link = self.parse_specification(content)
        discounts, max_discount = self.parse_discount(content)
        complectation_dict = self.parse_complectation(content)
        comment = self.parse_comment(content)
        time.sleep(0.2)
        try:
            seller_dict = self.get_seller_info(content)
        except Exception as ex:
            print("seller_dict", ex)
            seller_dict = dict()
        ad_dict = dict()
        ad_dict.update(seller_dict)
        ad_dict["ad_link"] = l
        ad_dict["year"] = year
        ad_dict["comment"] = comment
        ad_dict["options"] = complectation_dict
        ad_dict["max_discount"] = max_discount
        ad_dict["discounts"] = discounts
        ad_dict["car_model_link"] = specification_link
        ad_dict["characteristics"] = card_info_text
        ad_dict["title"] = title
        ad_dict["price"] = price_block_text[0]
        ad_dict["price_without_discounts"] = ""
        if len(price_block_text) > 1:
            if len(price_block_text) == 3:
                ad_dict["max_discount"] = price_block_text[1]
                ad_dict["price_without_discounts"] = price_block_text[2]
            else:
                ad_dict["price_without_discounts"] = price_block_text[1]
        self.all_ads[l] = ad_dict
        print(self.all_ads[l])
        # if parsed_number % save_every_n == 0 and self.engine == "pickle":
        #     pickle.dump(self.all_ads, open(self.pcl_name, "wb"))
        time.sleep(3)

    def filter_links(self, links, href_already=False):
        if not href_already:
            links = [l.get_attribute("href") for l in links]
        links = [l for l in links if l not in self.all_ads]
        return links

    def get_new_links(self):
        links = self.scraper.browser.find_elements_by_xpath(
            "//a[@class='Link ListingItemTitle-module__link']")
        links = self.filter_links(links)
        # links = [l for l in links if l not in manager_dict]
        return links

    def get_proxy(self):
        new_proxies = proxer.new_proxies.difference(self.done_proxies)
        while not new_proxies:
            proxer.get_new_proxies()
            proxer.check_proxies(
                wait_limit=len(proxer.new_proxies),
                check_url="http://auto.ru")
            new_proxies = proxer.new_proxies.difference(self.done_proxies)
        new_proxy = list(new_proxies)[0]
        self.done_proxies.add(new_proxy)
        return new_proxy

    def save_links(self):
        if self.engine in ("pandas", "sqlite"):
            # lazy conversion of a dict to a pandas.DataFrame
            keys = self.all_ads.keys()
            ads = list(self.all_ads.values())
            df = pd.DataFrame(ads)
            df["keys"] = keys
            self.all_ads = dict()
            if self.engine == "pandas":
                with open(self.pandas_name, 'a') as f:
                    df.to_csv(f, header=False, index=False)
            else:
                print("saving")
                df.to_sql(
                    "auto-ru", con=self.sql_engine, index=False,
                    if_exists='append')
                self.done_links = pd.read_sql(
                    "auto-ru", self.sql_engine)["keys"]
        else:
            # manager_dict.update(self.all_ads)
            pickle.dump(self.all_ads, open(self.pcl_name, "wb"))

    def loop_parsing(self, proxy=None):
        self.init_scraper(proxy=proxy)
        while True:
            links = self.get_new_links()
            print("Links #", len(links))
            for l_i, l in enumerate(links):
                # Check one more time,
                # because the link may have been parsed in the loop
                if self.captcha:
                    break
                if l in self.all_ads or l in self.done_links:
                    continue
                try:
                    self.parse_auto_ru_add(l, self.parsed_number)
                except Exception as ex:
                    print(l, ex)
                    continue
                self.parsed_number += 1
                if l_i % 25 == 0 and l_i > 0:
                    self.save_links()
            if self.all_ads:
                self.save_links()
            self.save_links()
            if self.captcha:
                self.init_scraper()
                self.captcha = False
            self.page += 1
            if self.scrape_regionwise:
                if self.page != 0 and self.page % 10 == 0:
                    self.page = 0
                    self.region_counter += 1
                    # not to overflow the regions length
                    self.region_counter = self.region_counter % len(regions)
                    self.region = regions[self.region_counter]
            try:
                self.scraper.browser.get(
                    f"https://auto.ru/{self.region}cars/{self.new_only}/"
                    f"?sort=fresh_relevance_1-desc&page={self.page}")
            except TimeoutException:
                time.sleep(60)
                try:
                    self.scraper.browser.get(
                        f"https://auto.ru/{self.region}cars/{self.new_only}/"
                        f"?sort=fresh_relevance_1-desc&page={self.page}")
                except TimeoutException:
                    self.init_scraper()
            time.sleep(10)

    @staticmethod
    def ads_to_csv(new_only=True):
        ads = pickle.load(open("auto-ru/auto_ru_ads.pcl", "rb"))

        specifications = pickle.load(open("auto-ru/cars.pcl", "rb"))

        specifications = [{k.lower(): v.lower() for k, v in c.items()
                           if type(v) == str}
                          for c in specifications]
        model_spec = {s['ссылка на модель']: s_i
                      for s_i, s in enumerate(specifications)}
        spec_spec = {s['ссылка на спецификацию']: s_i
                     for s_i, s in enumerate(specifications)}
        if new_only:
            ads = {k: v for k, v in ads.items()
                   if "ad_link" in v and "new" in v["ad_link"]}
        ads_list = []
        for a_i, a in enumerate(ads):
            print("\t", a_i, len(ads), end="\r")
            a = ads[a]
            new_dict = dict()
            if "characteristics" in a:
                a.update(a["characteristics"])
            if "metro" in a:
                if a["metro"]:
                    new_dict["метро"] = a["metro"][0]
            if "options" in a:
                for o in a["options"]:
                    o = a["options"][o]
                    new_dict.update(dict(zip(o, ["наличие"] * len(o))))
            for k, v in a.items():
                if k in ["discounts", "двигатель"]:
                    continue
                if type(v) == str:
                    v = v.replace('Смотреть статистику цен', "")
                    v = v.replace('₽', "")
                    v = re.sub("(\d+) (\d+)", r"\1\2", v)
                    v = re.sub("(\d+) (\d+)", r"\1\2", v)
                    v = re.sub("(\d+) (\d+)", r"\1\2", v)
                    v = re.sub("л\.с\.| л$| км$|^от |", "", v)
                    v = v.strip()
                    new_dict[k] = v
            new_dict = {k.lower(): v for k, v in new_dict.items()}
            if a["car_model_link"] in spec_spec or\
                    a["car_model_link"] in model_spec:
                if a["car_model_link"] in spec_spec:
                    spec_id = spec_spec[a["car_model_link"]]
                else:
                    spec_id = model_spec[a["car_model_link"]]
                spec = specifications[spec_id]
                new_dict.update(spec)
            ads_list.append(new_dict)
        df = pd.DataFrame(ads_list)
        df.to_csv("auto-ru/auto_ru_ads.csv")
        # to overcome the Excel limitation of having >65k URLs on a single
        # spreadsheet
        writer = pd.ExcelWriter("auto-ru/auto_ru_ads.xlsx",
                                engine='xlsxwriter',
                                options={'strings_to_urls': False})
        df.to_excel(writer)

    @staticmethod
    def dump_to_main():
        files = glob("auto-ru/*auto_ru_ads_*.pcl")
        main_dict = pickle.load(open("auto-ru/auto_ru_ads.pcl", "rb"))
        for f in files:
            f_dict = pickle.load(open(f, "rb"))
            main_dict.update(f_dict)
        pickle.dump(main_dict, open("auto-ru/auto_ru_ads.pcl", "wb"))


class SeleniumOrchestra:
    # def worker():
    #     while True:
    #         item = q.get()
    #         if item is None:
    #             break
    #         do_work(item)
    #         q.task_done()

    # q = queue.Queue()
    # threads = []
    # for i in range(num_worker_threads):
    #     t = threading.Thread(target=worker)
    #     t.start()
    #     threads.append(t)

    # for item in source():
    #     q.put(item)

    # # block until all tasks are done
    # q.join()

    # # stop workers
    # for i in range(num_worker_threads):
    #     q.put(None)
    # for t in threads:
    #     t.join()
    def __init__(self):
        self.proxer = ferret.Ferret()
        self.proxer.check_proxies(wait_limit=len(self.proxer.new_proxies))
        while len(self.proxer.new_proxies) < 3:
            self.proxer.get_new_proxies()
        scrapers = []
        self.all_ads = pickle.load(open(self.pcl_name, "rb"))
        for i in range(4):
            if i > 0:
                proxy = None
            else:
                proxy = sorted(self.proxer.new_proxies)[i - 1]
            ad_scraper = AutoRuAdScraper(
                external_ads_dict=self.all_ads, proxy=proxy)
            scrapers.append(ad_scraper)
        # ad_scraper.loop_parsing()

    def worker(self, proxy):
        pass


if __name__ == "__main__":
    # ad_scraper = AutoRuAdScraper(proxy=True)
    # ad_scraper = AutoRuAdScraper(engine="sqlite", page=1)
    # ad_scraper.loop_parsing()
    workers = num_workers
    processes = []
    for i in range(workers):
        page = i * 10
        ad_scraper = AutoRuAdScraper(
            engine="pickle", page=page, worker_id=i + 10)

        if i == 0:
            proxy = None
        else:
            proxy = sorted(proxer.new_proxies)[i - 1]
        processes.append(
            Process(target=ad_scraper.loop_parsing, kwargs={
                "proxy": proxy})
        )
    for p in processes:
        p.start()
        # p.join()
