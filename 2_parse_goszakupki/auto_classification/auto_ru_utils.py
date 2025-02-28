import time
import requests
from bs4 import BeautifulSoup
from scraper.scr import Scraper


def get_page(link, parse_with_bs4=False):
    print(" " * 100, link, end="\r")
    print("\t", link, end="\r")
    if parse_with_bs4:
        r = requests.get(link, timeout=10)
        r.encoding = "utf8"
        data = r.text
    else:
        try:
            scraper.browser.get(link)
        except Exception as ex:
            print(ex)
            scraper.browser.get(link)
        data = scraper.browser.page_source
        if "Нам очень жаль" in data:
            time.sleep(60)
        while "Нам очень жаль" in data:
            scraper.browser.get(link)
            data = scraper.browser.page_source
            # 2 hours
            time.sleep(60 * 10 * 6 * 2)
    time.sleep(2)
    soup = BeautifulSoup(data, "lxml")
    return soup


def get_specifications(soup):
    specifications = soup.findAll("a")
    specifications = [s for s in specifications if "href" in s.attrs]
    specifications = [s for s in specifications if
                      "specifications" in s['href']]
    specifications = [s for s in specifications
                      if "/login/" not in s["href"]]
    specifications = [s for s in specifications
                      if s.text != "Характеристики"]
    # specifications = specifications[:1]
    return specifications


scraper = Scraper(headless=True)
