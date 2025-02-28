import pickle
import re
import time
import random

from proxy_ferret import ferret
# from scraper.scr import Scraper
from auto_ru_utils import scraper, get_page, get_specifications


first_load = False
parse_with_bs4 = False

folder = 'auto-ru'
car_links_file = f"{folder}/car_links.pcl"
done_links_file = f"{folder}/done_links.pcl"
cars_dict_file = f"{folder}/cars.pcl"
links = [
    "https://auto.ru/htmlsitemap/mark_model_tech_{}.html".format(i)
    for i in range(1, 7)]

# scraper = Scraper(headless=False)
# proxer = ferret.Ferret()

if first_load:
    car_links = dict()
    for l_i, l in enumerate(links):
        result = scraper.get_link_tags_bs4(l)
        time.sleep(1)
        result = [r for r in result if r[1]]
        result = [r for r in result if "catalog" in r[1]]
        car_links.update(dict(result))
    pickle.dump(car_links, open(car_links_file, 'wb'))


car_links = pickle.load(open(car_links_file, 'rb'))

done_links = set()
done_links = pickle.load(open(done_links_file, "rb"))

cars = pickle.load(open(cars_dict_file, "rb"))

model_links = [c['Ссылка на модель'] for c in cars]
model_links += [c['Ссылка на спецификацию'] for c in cars]

# for key in car_links:
#     years = re.findall("\d{4}", key)
#     if not years:
#         car_links[key] = None
#         continue
#     if int(years[0]) < 2010:
#         car_links[key] = None
parsed_ads = pickle.load(open("auto-ru/auto_ru_ads.pcl", "rb"))

parsed_ads = {v["title"]: v['car_model_link'] for v in parsed_ads.values()
              if "ad_link" in v and "new" in v["ad_link"]}

car_links_tmp = car_links.copy()
car_links = parsed_ads
car_links.update(car_links_tmp)
del car_links_tmp
car_links = {k: v for k, v in car_links.items() if v and v not in
             done_links}
# car_links = {k: v for k, v in car_links.items() if v not in model_links}
# for k, v in car_links.items():
#     v = v.split("__")[0]
#     if any(v in l for l in model_links) or any(v in l for l in done_links):
#         car_links[k] = None
car_links = {k: v for k, v in car_links.items() if v}

print(len(car_links))
for c_i, c in enumerate(car_links):
    # c = random.choice(list(car_links.keys()))
    link = car_links[c]
    car_description = c
    if link in ['http://auto.ru/catalog/cars/',
                "http://auto.ru/htmlsitemap/mark_model_catalog.html"]:
        continue
    # break
    # car = scraper.get_link_tags_bs4(l)
    soup = get_page(link)

    specifications = get_specifications(soup)
    specifications = [s for s in specifications if
                      s["href"] not in done_links]  # "http://auto.ru" +
    if not specifications:
        done_links.add(link)
    for s in specifications:
        # Remove "mark":"
        mark = re.findall('"mark":".*?"', s["data-bem"])[0][8:-1]
        # Remove "model":"
        model = re.findall('"model":".*?"', s["data-bem"])[0][9:-1]

        # specification link
        s_link = s["href"]  # "http://auto.ru" +
        if s_link in done_links:
            print("Already parsed")
            print(s_link)
            print(link)
            continue
        soup = get_page(s_link)
        version = soup.findAll("a", {"href": "/catalog/cars/all/"})
        if version:
            version = version[-1].text
        else:
            version = None
        body = soup.findAll("a", {"href": "/catalog/cars/////"})
        if body:
            body = body[-1].text
        else:
            body = None
        try:
            content = soup.find("div", {"class": "catalog__content"})
            characteristics_names = content.findAll(["dt"])
        except AttributeError:
            # 30 minutes
            print("\nCAPTCHA\n")
            time.sleep(60 * 30)
            continue
        characteristics = content.findAll(["dd"])
        characteristics = [c.text for c in characteristics]
        characteristics_names = [c.text for c in characteristics_names]
        s_dict = {"Описание": car_description, "Марка": mark, "Модель": model,
                  "Ссылка на модель": link, "Ссылка на спецификацию": s_link,
                  "Версия": version,
                  "Кузов": body}
        s_dict.update(zip(characteristics_names, characteristics))

        # print(s_dict)
        soup = get_page(s_link.replace("specifications", "equipment"))

        content = soup.find("div", {"class": "catalog__content"})
        if not content:
            continue
        config_name = content.find(
            "div", {"class": "catalog__package-name"}).text
        config_price = content.find(
            "div", {"class": "catalog__package-price"})
        if "У данной комплектации нет ни одной опции" not in content.text:
            if config_price:
                config_price = config_price.text
            else:
                config_price = None
            config_features = content.findAll(
                "div", {"class": "catalog__package-group clearfix"})
            features = dict()
            for feature in config_features:
                feature_name = feature.find("h3").text
                techs = [l.text for l in feature.findAll("li")]
                techs = [re.sub("Также опцию .*", "", t) for t in techs]
                features[feature_name] = techs
            config_options = content.findAll(
                "div", {"class": "package-option i-bem"})
            options = dict()
            for option in config_options:
                option_name = option.find(
                    "div", {"class": "package-option__name"}).text
                option_summary = option.find(
                    "div", {"class": "package-option__summary"}).text
                option_values = option_summary.split(",")
                option_price = option.text.replace(option_name, "").\
                    replace(option_summary, "").replace(" – ", "")
                for o in option_values:
                    option_price = option_price.replace(o.strip(), "")
                options[option_name] = {"Цена": option_price,
                                        "Опции": option_summary}
            if config_name and config_price:
                s_dict["Конфигурация"] = {
                    "Название": config_name, "Цена": config_price}
                if features:
                    s_dict["Конфигурация"].update(features)
                if options:
                    s_dict["Конфигурация"]["Опции"] = options
        print(s_dict)
        cars.append(s_dict)
        done_links.add(s_link)
        if len(cars) % 10 == 0 or c_i == len(car_links) - 1:
            pickle.dump(
                cars, open("cars_{}.pcl".format(str(len(cars) / 50)[0]), "wb"))
            pickle.dump(
                cars, open(cars_dict_file, "wb"))
            pickle.dump(
                done_links,
                open("done_links_{}.pcl".format(str(len(cars) / 50)[0]), "wb"))
            pickle.dump(
                done_links,
                open(done_links_file, "wb"))
    done_links.add(link)
