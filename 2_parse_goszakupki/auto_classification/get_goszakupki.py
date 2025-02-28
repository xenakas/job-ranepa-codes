import re
import os
import pickle
import pandas as pd
from collections import Counter
from sql.db_interface import SQL
from sql_config import sql_config
from utils_d.web_utils import link2text
from characteristics_extractor import CharactericsExtractor
from utils import (parse_pandas_table, parse_goszakupki_text,
                   filter_pandas_tables)

# "http://zakupki.gov.ru/44fz/filestore/public/1.0/download/rgk2/"\
#     "file.html?uid=3A6A585B74640088E053AC11071A275E"
# "http://zakupki.gov.ru/44fz/filestore/public/1.0/download/rgk2/"\
#     "file.html?uid=398A9786C61B01FAE053AC11071A193B"
"""
29.10.1     Двигатели внутреннего сгорания для автотранспортных средств
29.10.2     Автомобили легковые
    29.10.21    Средства транспортные с двигателем с искровым зажиганием,
                с рабочим объемом цилиндров не более 1500 см3, новые
    29.10.22    Средства транспортные с двигателем с искровым зажиганием,
                с рабочим объемом цилиндров более 1500 см3, новые
    29.10.23    Средства транспортные с поршневым двигателем внутреннего
                сгорания с воспламенением от сжатия (дизелем или полудизелем),
                новые
    29.10.24    Средства автотранспортные для перевозки людей прочие
29.10.3     Средства автотранспортные для перевозки 10 или более человек
29.10.4     Средства автотранспортные грузовые
29.10.5     Средства автотранспортные специального назначения
29.10.9     Услуги по производству автотранспортных средств отдельные,
            выполняемые субподрядчиком
"""

"""
In [529]: for p_i, p in enumerate(pandas_tables):
     ...:     for t_i, t in enumerate(p):
     ...:         print(p_i, t_i)
     ...:         print(parse_pandas_table(t))
"""


def parse_goszakupki_key(key, urls, total_parsed_index=0,
                         save_pandas_only=False):
    all_dicts = []
    if key in urls_parsed_dict:
        return all_dicts
    if save_pandas_only:
        if key in pandas_tables:
            return all_dicts
    if total_parsed_index % 11 == 0:
        pickle.dump(urls_parsed_dict, open("urls_parsed_dict.pcl", 'wb'))
        pickle.dump(all_texts, open("all_texts.pcl", 'wb'))
        pickle.dump(pandas_tables, open("pandas_tables.pcl", 'wb'))
    urls = [u for u in urls if "download" in u]
    # urls = ["http://zakupki.gov.ru/44fz/filestore/public/1.0/download/rgk2/"
    #         "file.html?uid=3A6A585B74640088E053AC11071A275E"]
    texts = []
    folder = key.replace(":", "-")
    if not os.path.exists("all_files"):
        os.mkdir("all_files")
    folder = "all_files/" + folder
    for u in urls:
        texts += link2text(u, folder_name=folder, to_delete=False)
    texts = [t for t in texts if t]
    url_texts = [t[0] for t in texts]
    all_texts[key] = url_texts
    for text in texts:
        tables = []
        if type(text) == list and len(text) == 2:
            tables = text[1]
            text = text[0]
        tables = [t.df for t in tables]
        tables = [t for t in tables if t.size > 0]
        if tables:
            if key not in pandas_tables:
                pandas_tables[key] = tables
            else:
                pandas_tables[key] += tables
        # Do not parse anything
        if save_pandas_only:
            print(tables)
            continue

        parsed = parse_goszakupki_text(text)
        parsed_dict = dict()
        for p in parsed:
            if p:
                parsed_dict.update(p)
        table_dicts = []
        # Pandas DataFrame to dict
        tables = filter_pandas_tables(tables)
        table_dicts += tables
        if table_dicts:
            parsed += table_dicts
        if parsed:
            all_dicts.append(parsed)
    print("Parsing link #", total_parsed_index - 1)
    # if not save_pandas_only:
    print(all_dicts)
    urls_parsed_dict[key] = all_dicts
    return all_dicts


def get_goszakupki_auto_urls(sql, urls_dict, offset, offset_step):
    sql.cur.execute(
        """SELECT
        *
        from "public"."tab_doc_main"
        WHERE doc_xml_content::TEXT LIKE '%29.10.2%'
        ORDER BY doc_id_compound
        LIMIT {}
        OFFSET {}
        """.format(offset_step, offset)
    )
    texts = sql.cur.fetchall()

    # bad_words = ["ремонт", " шин", "автошин", "бензин", "топлива", "гсм",
    #              "дорог", "горюче-смазочн", "благоустр", "дорожек",
    #              "запасных частей", "топливо", "дорожного", "строительств",
    #              "моторного масла", "реконструкция", "мойке", "содержание",
    #              "жидкост", "пневмобал", "по перевозке", "смазочн",
    #              "тосол", "стеклоомы", "для автомоб", "топливн", "амортиз",
    #              "компресс", "нефт", "по установк", "выполнение работ",
    #              "запчаст", "устройств", "доставк", "отсыпк", "услуг",
    #              "перевозк"]

    for t_i, t in enumerate(texts):
        compound_id = t[3]  # doc_id_compound
        t = t[-1]
        info = re.findall("<purchaseObjectInfo>.*</purchaseObjectInfo>", t)
        code = re.findall("<code>.*</code>", t)
        url = re.findall("<url>.*</url>", t)
        url = [re.sub("</*url>", "", u) for u in url]
        urls_dict[compound_id] = url
        code = [c for c in code if "." in c]
        if info:
            info = re.sub("</*purchaseObjectInfo>", "", info[0])
        if code:
            code = re.sub("</*code>", "", code[-1])
        if info:
            # info_l = info.lower()
            # if not any(w in info_l for w in bad_words):
            #     print(info, code, t_i)
            if code.startswith("29.1"):
                print(info, code, t_i)
    return urls_dict


def test_parse_pandas_table():
    pandas_tables = pickle.load(open("pandas_tables_22oct.pcl", 'rb'))
    pandas_tables = [list(v) for k, v in pandas_tables.items()]
    pandas_tables = [t for p in pandas_tables for t in p]
    good_tables = [
        0, 1, 11, 12, 19, 20, 21, 22, 23, 24, 38, 40, 44, 55, 59, 64, 65, 69,
        70, 73, 74, 90, 92, 95, 96,
    ]
    error_tables = []
    for p in good_tables:
        p = pandas_tables[p]
        t = parse_pandas_table(p)
        if not t:
            error_tables.append(p)
    return error_tables


def filter_tables(tables):
    tables = [t for t in tables if extractor.filter_dict(t)]
    for t_i, t in enumerate(tables):
        for va in t.values():
            if type(va) != str:
                tables[t_i] = None
    tables = [t for t in tables if t]
    tables = [
        t for t in tables if len([v for v in t.values()
                                  if re.findall("\d+", v)]) >= 1]
    tables = [t for t in tables if not any("инн" in k for k in t.keys())]
    for t_i, t in enumerate(tables):
        new_dict = dict()
        for key, v in t.items():
            new_key = re.sub(", .*", "", key)
            v = v.replace("мм", "").strip()
            new_dict[new_key] = v
        tables[t_i] = new_dict
    counter = Counter()
    for t in tables:
        for key in t.keys():
            counter[str(key).lower()] += 1
    counter = [c for c in counter if counter[c] > 10]
    for t_i, t in enumerate(tables):
        to_delete = []
        for key in t.keys():
            if str(key).lower() not in counter:
                to_delete.append(key)
        for key in to_delete:
            del tables[t_i][key]
    tables = [t for t in tables if t]
    tables = [t for t in tables if len(t.keys()) > 2]
    for t_i, t in enumerate(tables):
        new_t = dict()
        for k, v in t.items():
            v = re.sub("(\d+) (\d+)", r"\1\2", v)
            v = re.sub("(\d+) (\d+)", r"\1\2", v)
            v = re.sub("(\d+),(\d+)", r"\1\2", v)
            new_t[k.lower()] = v.lower()
        tables[t_i] = new_t
    tables = [
        t for t in tables if len([v for v in t.values()
                                  if re.findall("\d+", v)]) >= 1]
    return tables


def convert_to_csv(filename, dict_filename, out_filename):
    pandas_tables = pickle.load(open(filename, 'rb'))
    pandas_tables = [list(v) for k, v in pandas_tables.items()]
    pandas_tables = [t for p in pandas_tables for t in p]
    pandas_tables = [parse_pandas_table(t) for t in pandas_tables]
    pandas_tables = [p for p in pandas_tables if p and any(bool(t) for t in p)]
    pandas_tables = [t for p in pandas_tables for t in p]
    pandas_tables = filter_tables(pandas_tables)
    # all_texts = pickle.load(open("all_texts.pcl", 'rb'))
    # all_texts = [list(v) for k, v in all_texts.items()]
    # all_texts = [t for p in all_texts for t in p]
    # all_texts = [t for t in all_texts if len(t) > 100]
    # parsed = []
    # for a_i, a in enumerate(all_texts):
    #     print(a_i, end="\r")
    #     parsed.append(parse_goszakupki_text(a))
    pandas_tables_df = pd.DataFrame(pandas_tables)
    urls_parsed_dict = pickle.load(open(dict_filename, 'rb'))
    urls_parsed = [list(v) for k, v in urls_parsed_dict.items()]
    urls_parsed = [t for p in urls_parsed for t in p]
    urls_parsed = [t for p in urls_parsed for t in p]
    urls_parsed = filter_tables(urls_parsed)
    urls_parsed = [{re.sub(", .*", "", k): v for k, v in t.items()}
                   for t in urls_parsed]
    urls_parsed = [{k: v for k, v in t.items()
                    if k in pandas_tables_df.columns}
                   for t in urls_parsed]
    urls_parsed = [t for t in urls_parsed if t]

    urls_parsed += pandas_tables
    urls_parsed_df = pd.DataFrame(urls_parsed)
    urls_parsed_df.to_csv(out_filename)


codes = ["29.10.24.000", "29.10.22.000", "29.10.42.121",
         "29.10.21.000"]
first_time_download = False
save_pandas_only = False
extractor = CharactericsExtractor()
sql = SQL(**sql_config)

if first_time_download:
    urls_dict = dict()
    get_goszakupki_auto_urls(sql, urls_dict)

urls_dict = pickle.load(open("goszakupki_urls.pcl", "rb"))

# urls_parsed = pickle.load(open("urls_parsed.pcl", 'rb'))
# urls_parsed_dict = dict()
# all_texts = dict()

pandas_tables = dict()
urls_parsed_dict = dict()
all_texts = dict()
if os.path.exists("pandas_tables.pcl"):
    pandas_tables = pickle.load(open("pandas_tables.pcl", 'rb'))
if os.path.exists("urls_parsed_dict.pcl"):
    urls_parsed_dict = pickle.load(open("urls_parsed_dict.pcl", 'rb'))
if os.path.exists("all_texts.pcl"):
    all_texts = pickle.load(open("all_texts.pcl", 'rb'))
print("Total links parsed", len(urls_parsed_dict))
n_items = 0
table = '"public"."tab_doc_main"'
# sql.cur.execute("SELECT COUNT(*) FROM {}".format(table))
# n_texts = sql.cur.fetchone()[0]
offset_step = 100
offset = 0
total_parsed_index = 0
if __name__ == "__main__":
    while True:
        new_urls = get_goszakupki_auto_urls(
            sql, urls_dict, offset, offset_step)
        urls_dict.update(new_urls)
        pickle.dump(urls_dict, open("goszakupki_urls.pcl", "wb"))
        offset += offset_step
        for key, urls in new_urls.items():
            parse_goszakupki_key(key, urls, total_parsed_index,
                                 save_pandas_only)
            total_parsed_index += 1
# for u_i, u in enumerate(urls_parsed):
#     united_dict = dict()
#     for u_list in u:
#         for d in u_list:
#             united_dict.update(d)
#     urls_parsed[u_i] = d
# urls_parsed = [u for u in urls_parsed if u]
# for u_i, u in enumerate(urls_parsed):
#     for a_i, a in enumerate(u):
#         for p_i, p in enumerate(p):
#             p = [l for l in p if l]
#             urls_parsed[u_i][a_i][p_i] = p
#         a = [l for l in a if l]
#         urls_parsed[u_i][a_i] = a
#     u = [l for l in u if l]
#     urls_parsed[u_i] = u

# pandas_tables = pickle.load(open("pandas_tables.pcl", 'rb'))
# pandas_tables = [list(v) for k, v in pandas_tables.items()]
# pandas_tables = [t for p in pandas_tables for t in p]
# parsed = [parse_pandas_table(t) for t in pandas_tables]
# for tables_i, tables in enumerate(parsed):
#     for table_i, table in enumerate(tables):
#         result = [k for k in table.keys() if
#                   any(str(k).lower() == o for o in extractor.columns) or
#                   any(o == str(k).lower() for o in extractor.columns)]
#         if len(result) >= 2:
#             print(tables_i, table_i)
#             print(result)
#             print(table)
