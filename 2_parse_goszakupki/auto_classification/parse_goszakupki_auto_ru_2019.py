import pickle
import json
import re
import os
import traceback
import time

import socket

from typing import *
from collections import Counter
# import multiprocessing as mp
from multiprocessing import Process

import numpy as np
import pandas as pd
import requests
import boto3

import random

from ranepa_s3_wrapper.wrapper import MinioS3
from sql.db_interface import SQL
from sql_config import sql_config
from ranepa_string_utils.string_utils import text_pipeline
from utils_d.web_utils import process_file  # , getting_tables
from yargy_cars.yargy_extractor import YargyRulesBuilder
from utils import filter_pandas_tables, init_sql, numerics, ner_dict
from utils import clean_yargy_str
from s3_config_ranepa5 import CONFIG
# from utils import parse_pandas_table
# from utils import parse_goszakupki_text


def init_s3():
    s3_config = json.load(open("s3_config.json"))
    s3_passwd = json.load(open("s3_passwd.json"))
    s3_config = dict(s3_config, **s3_passwd)
    # s3_config["bucket"] = "test"
    # log.info("s3 bucket: %s" % s3_config["bucket"])
    # log.info("create instance session s3")

    s3_session = boto3.session.Session()
    # log.info("create instance client s3")
    s3_client = s3_session.client(
        service_name="s3",
        aws_access_key_id=s3_config["access_key"],
        aws_secret_access_key=s3_config["secret_key"],
        endpoint_url="".join(
            ["http://", s3_config["host"], ":", s3_config["port"]]))

    # log.info("create instance resource s3")
    s3_resource = s3_session.resource(
        service_name="s3",
        aws_access_key_id=s3_config["access_key"],
        aws_secret_access_key=s3_config["secret_key"],
        endpoint_url="".join(
            ["http://", s3_config["host"], ":", s3_config["port"]]))

    # log.info("check bucket: %s" % s3_config["bucket"])
    s3_bucket = None
    if s3_resource.Bucket(s3_config["bucket"]) in\
            s3_resource.buckets.all():
        # log.info("bucket exist")
        s3_bucket = s3_resource.Bucket(s3_config["bucket"])
    else:
        # log.warning("bucket not exist")
        # log.info("exit")
        exit()
    return s3_client, s3_bucket, s3_resource


def export():
    sql_write.cur.execute(
        """select
        *
        from auto_ru;
        """)
    rows = sql_write.cur.fetchall()
    rows = [r[2] for r in rows]
    df = pd.DataFrame(rows)
    merge = dict()
    for c_i, c in enumerate(df.columns):
        # товара характеристик ['товара характеристика',
        # 'товара характеристикам', 'товара характеристики']
        c_spl = c.split()
        start = [c2 for c2_i, c2 in enumerate(df.columns)
                 if c2.startswith(c) and c2_i != c_i and
                 len(c_spl) == len(c2.split()) and len(c) > 3]
        # 'передней подвеске', 'передней подвески'
        start += [c2 for c2_i, c2 in enumerate(df.columns)
                  if c_i != c2_i and
                  len(c2.split()) >= 2 and len(c_spl) >= 2 and
                  c2.split()[0][:3] == c.split()[0][:3] and
                  c2.split()[1][:3] == c_spl[1][:3]
                  ]
        if len(c_spl) == 1:
            start += [
                c2 for c2_i, c2 in enumerate(df.columns)
                if c2_i != c_i and len(c) > 4 and
                len(c2.split()) == 1 and c[:-3] == c2[:-3]]
        start = set(start)
        start = {s for s in start if s not in merge}
        merge[c] = start
        if start:
            print(c, start)
    for k, v in merge.items():
        for c in v:
            # df[k] = df[k] + df[c]
            df[k] = df[k].combine_first(df[c])
    # flattened columns
    to_delete = [l for k, v in merge.items() for l in v]
    to_delete += [c for c in df.columns if len(c.split()) >= 2 and
                  c.split()[1].startswith(c.split()[0][:3])]
    # not latin short; exclude ABS ESP
    to_delete += [c for c in df.columns if len(c) < 5 and ord(c[0]) > 1000]
    df.drop(to_delete, axis=1, inplace=True)
    # df.to_csv("goszakupki_auto_2019.csv", index=False)
    df.to_excel("goszakupki_2019_2.xlsx", index=False)


def get_all_data_from_xml(sql_con, purchase_id):
    sql_con.cur.execute(
        f"""
        SELECT
        "doc_xml_content"
        FROM "public"."tab_doc_main"
        WHERE
        ((xpath('//*[name()="purchaseNumber"]/text()',
        "doc_xml_content"))[1]::TEXT) = '{purchase_id}'
        AND
        "doc_type" = 'n1_x003a_fcsNotificationEF';
        """)
    text = sql_con.cur.fetchone()
    if text:
        text = text[0]
        characteristics = re.findall(r"\. [А-Я].*?[:;] .*?\.", text)
        if characteristics:
            characteristics = [re.split("[:;]", c) for c in characteristics]
            characteristics = [c for c in characteristics if len(c) == 2]
            characteristics = dict(characteristics)
            clean_chars = dict()
            for k, v in characteristics.items():
                k = clean_yargy_str(k)
                v = clean_yargy_str(v, k=k)
                if v and k:
                    clean_chars[k] = v
            clean_chars = json.dumps(clean_chars)
            return clean_chars


def enrich_with_xml(sql_con, purchase_id):
    data = dict()
    for field, value_name in (("maxPrice", "price"),
                              ("quantity", "quantity"),
                              ("date", "date"),
                              ("factAddress", "factAddress"),
                              ("orgPostAddress", "orgPostAddress"),
                              ("postAddress", "postAddress"),
                              # ("deliveryPlace", "deliveryPlace"),
                              ("regNum", "regNum")):
        sql_con.cur.execute(
            f"""
            SELECT
            xpath('//*[name()="{field}"]/text()', "doc_xml_content"::xml)
            FROM "public"."tab_doc_main"
            WHERE
            ((xpath('//*[name()="purchaseNumber"]/text()',
            "doc_xml_content"))[1]::TEXT) = '{purchase_id}'
            AND
            "doc_type" = 'n1_x003a_fcsNotificationEF';
            """)
        value = sql_con.cur.fetchone()
        if value:
            value = value[0]
            value = value.replace("{", "").replace("}", "")
            if "Address" not in value_name:
                value = value.split(",")
            if value:
                try:
                    value = float(value[0])
                except ValueError:
                    pass
            else:
                price = None
        data[value_name] = value
    data = json.dumps(data)
    return data


def query_all_files(sql):
    sql.cur.execute(
        """select
        id_sha256,
        metadata->>'file_href_s3_bucket',
        metadata->>'file_href_s3_key',
        purchase_number
        from proc.tab_files;
        """
    )

    files = sql.cur.fetchall()
    files = [f for f in files if f[1]]
    files = [f for f in files if not f[1].endswith(".content")]
    return files


def get_s3_ranepa5_files():
    df = pd.read_csv("SELECT_uuid__purchase_number__s3_link__n.csv")

    df["s3_link"] = df["s3_link"].str.replace("s3://liori5/", "")

    df["bucket"] = df["s3_link"].str.split("/").str[0]
    df["key"] = df["s3_link"].str.split("/").str[1:].str.join("/")
    df["filename"] = df["s3_link"].str.split("/").str[-1]
    rows = [r[1] for r in df.iterrows()]
    return rows


yargy_cars = YargyRulesBuilder()
yargy_cars.build_rules()

specification_values = set()
for s in yargy_cars.specifications:
    specification_values.update(s.keys())
specification_values = {s for s in specification_values if s}

USE_YARGY = False
PARSE_TABLES = False

WORKERS = 2

LOCAL = False

ALL_VERSIONS = ["1a", "1b", "1c",
                '1e', '1e_no_yargy',
                '1f', '1f_no_yargy',  # the last version
                "1d", "1d_no_yargy",  # the main version
                ]
BASE_VERSION = "_1f"
VERSION = "'1f_no_yargy'"

specifications = pickle.load(open("auto-ru/cars.pcl", "rb"))

specifications = [{k.lower(): v.lower() for k, v in c.items()
                   if type(v) == str}
                  for c in specifications]
allowed_keys = set(ner_dict.values())
# allowed_keys.update(specification_values)
# reverse_specifications = {s["Описание"].lower(): s_i
#                           for s_i, s in enumerate(specifications)}

sql, sql_write, sql_config_write = init_sql()

files = query_all_files(sql)

# sums = []
# for file_i, file in enumerate(files):
#     print("\t", file_i, len(files), end="\r")
#     purchase_id = file[-1]

random.shuffle(files)
s3_client, s3_bucket, s3_resource = init_s3()
if socket.gethostname() == "iori2":
    CONFIG["host"] = "10.8.0.2"
    CONFIG["port"] = "9000"
instance_s3_r5 = MinioS3(CONFIG)

numeric_keys = ("количество", "цена", "цене")
bad_keys = ("товар товар", "коробки")
sql_write.cur.execute(
    """select
    id_sha256
    from goszakupki_2;
    """)  # from auto_ru
done_files = sql_write.cur.fetchall()
texts = [d[-1] for d in done_files]
done_files = [d[0] for d in done_files]
done_files = set(done_files)
print("done", len(done_files))

files = [f for f in files if
         f[0] + VERSION not in done_files and
         f[0] + BASE_VERSION not in done_files]
format_files = []

formats = ("docx", "doc", "rtf")
for form in ("docx", "doc", "rtf"):
    form_files = [f for f in files if f[2].endswith(form)]
    format_files += form_files
# format_files += [f for f in files if not f[2].endswith(formats)]
files = format_files
files += get_s3_ranepa5_files()

# if socket.gethostname() == "iori2":
#     files = files[: int(len(files) / 2)]
# # elif socket.gethostname() == "iori4":
# #     files = files[int(len(files) / 3): 2 * int(len(files) / 3)]
# else:
#     files = files[int(len(files) / 2):]


def parse_yargy(text: str, to_json=True):
    text = " ".join(text.replace("|", " ").split())
    output = yargy_cars.extract(text)
    if not output:
        return None
    # print(output)
    # enrich extracted dict with some information for the current car model
    output = yargy_cars.clean_yargy(output)
    if to_json:
        # json doesnt accept sets
        output = {k: list(v) if type(v) == set else v
                  for k, v in output.items()}
        output = json.dumps(output)
    return output


def get_deeppavlov_ner(text: str, port="5031", url="http://10.8.0.6"):
    output = dict()
    response = requests.post(f"{url}:{port}/",
                             json={"text": text}).json()
    if not response or "NER" not in response:
        return None
    keys = []
    values = []
    for r_i, r in enumerate(response["NER"]):
        if r == "O":
            continue
        if r.startswith("B-"):
            keys.append(response["text"][r_i])
            values.append(r)
        elif r.startswith("I-"):
            if values:
                if values[-1].endswith(r.replace("I-", "")):
                    keys[-1] += " " + response["text"][r_i]
            else:
                keys.append(response["text"][r_i])
                values.append(r)
    if keys:
        output = dict(zip(keys, values))
    output = json.dumps(output)
    return output


def parse_texts(f, sql_con=sql_write, sql_main=sql):

    if type(f) == pd.core.series.Series:
        sha_id, filename = get_s3_ranepa5_file(f)
    else:
        sha_id, filename = get_s3_file(f)

    processed = process_file(filename, None, get_tables=PARSE_TABLES,
                             to_delete=False,
                             folder_name="./s3/")
    if not processed:
        return None
    if PARSE_TABLES:
        text, camelot_output = processed
    else:
        text = processed
        camelot_output = []
    try:
        deepavlov_ner = get_deeppavlov_ner(text)
    except:
        time.sleep(10 * 60)
        deepavlov_ner = get_deeppavlov_ner(text)

    try:
        deepavlov_5032 = get_deeppavlov_ner(text, port=5032)
    except:
        time.sleep(10 * 60)
        deepavlov_5032 = get_deeppavlov_ner(text, port=5032)
    # print(text[:80])
    text = re.sub(r"(\d+) млн\.*", r"\1 000000", text)
    text = re.sub(r"(\d+) тыс\.*", r"\1 000", text)

    texts = text.split("\n")
    texts = [t for t in texts if t]
    yargy_output = dict()
    if USE_YARGY:
        yargy_found = [parse_yargy(t, to_json=False) for t in texts]
        for y_i in range(1, len(yargy_found) - 1):
            y = yargy_found[y_i]
            # leave text after and before yargy matches
            if y:
                if not yargy_found[y_i - 1]:
                    yargy_found[y_i - 1] = True
                if not yargy_found[y_i + 1]:
                    yargy_found[y_i + 1] = True
        texts = [t for t_i, t in enumerate(texts) if yargy_found[t_i]]

        yargy_found = [y for y in yargy_found if type(y) == dict]
        for y in yargy_found:
            yargy_output.update(y)
    print(yargy_output)
    if deepavlov_ner:
        print(json.loads(deepavlov_ner))
    if deepavlov_5032:
        print(json.loads(deepavlov_5032))
    yargy_output = json.dumps(yargy_output)
    if camelot_output:
        try:
            if PARSE_TABLES:
                camelot_output = filter_pandas_tables(camelot_output)
            else:
                camelot_output = dict()    
        except Exception as ex:
            print(ex)
            traceback.print_exc()
            camelot_output = dict()
        print("camelot filtered", camelot_output)

        united_camelot = dict()
        for c in camelot_output:
            united_camelot.update(c)
        camelot_output = united_camelot
    else:
        camelot_output = json.dumps(dict())
    output = {
        "id_sha256": sha_id + VERSION,
        "json_ner": deepavlov_ner,
        "json_ner_al": deepavlov_5032
    }
    if yargy_output:
        output["characteristics_yargy"] = yargy_output
    if camelot_output:
        output["characteristics_pandas"] = camelot_output
    return output


def main_yargy(files=files, sql_main=sql, sql_write=sql_write):
    # sql_write.cur.execute(
    #     """select
    #     id_sha256
    #     from goszakupki_2;
    #     """)
    # yargy_done = sql_write.cur.fetchall()
    # yargy_done = [d[0] for d in yargy_done]
    # yargy_done = done_files.difference(yargy_done)
    if not sql_write:
        sql_write = SQL(**sql_config_write)
    if not sql_main:
        sql_main = SQL(**sql_config)
    for f_i, f in enumerate(files):
        sha_id = f[0]
        purchase_id = f[3]
        if sha_id + VERSION in done_files:
            continue
        print(f_i, len(files))
        # try:
        output = parse_texts(f, sql_con=sql_main)
        if output:
            xml_output = enrich_with_xml(
                sql_con=sql_main, purchase_id=purchase_id)
            # xml_yargy = get_all_data_from_xml(sql_main, purchase_id)
            # output["characteristics_xml_yargy"] = xml_yargy
            output["characteristics_xml"] = xml_output
            try:
                sql_write.create_entry(
                    "goszakupki_2", check_existence_bool=False, **output)
                sql_write.create_entry(
                    "goszakupki_2b",
                    check_existence_bool=False, **output)
            except Exception as ex:
                output = None
                traceback.print_exc()
                print(ex)
        if not output:
            output = {
                "id_sha256": sha_id + VERSION,
            }
            sql_write.create_entry(
                "goszakupki_2",
                check_existence_bool=False, **output)
        # except Exception as ex:
        #     traceback.print_exc()
        #     print("EXCEPTION !!!! main function", ex)


def get_s3_file(f):
    sha_id = f[0]
    bucket = f[1]
    key = f[2]
    filename = "s3/" + key.split("/")[-1]
    key = key.replace("s3://", "")
    # file = s3_client.get_object(Bucket=bucket, Key=key)
    s3_resource.Bucket(bucket).download_file(key, filename)
    # text, tables = process_file(filename, None, get_tables=True,
    #                             to_delete=True,
    #                             folder_name="./s3/")
    return sha_id, filename


def get_s3_ranepa5_file(row):
    filename = row["filename"]
    filebytes = instance_s3_r5.get_key(row["bucket"], row["key"], True)
    with open(f"s3_r5/{filename}", "wb") as file:
        file.write(filebytes)
    filename = f"s3_r5/{filename}"
    sha_id = filename.split(".")[0] + "_r5"
    return sha_id, filename


def remove_rare_cols(df: pd.DataFrame):
    to_drop = ["числовые характеристики", "опции",
               "количествовые характеристики", "ед изм", "задний", "передний",
               "комплектация", "размер", "опция"]
    for col_i, col in enumerate(df.columns):
        if col in ner_dict.values():
            continue
        print("\t", col_i, len(df.columns), end="\r")
        # remove mostly empty columns
        if df[~df[col].isna()][col].shape[0] < 100:
            to_drop.append(col)
    df = df[[c for c in df.columns if c not in to_drop]]
    return df


def convert_add_info(additional_info):
    spec_counts = Counter()
    for a in additional_info:
        for d in a[2:]:
            if d:
                for k in d.keys():
                    spec_counts[k.lower()] += 1
    for a_i, a in enumerate(additional_info):
        specs = dict()
        for spec in a[2: 4]:
            if spec:
                specs.update(spec)
        reverse_specs = dict()

        # не менее 4320 мм': 'B-DLINA' -> 'B-DLINA': не менее 4320 мм'
        # select the most popular value for each key
        for k, v in specs.items():
            if "OPTION" in v:
                if "B-OPTION" in reverse_specs:
                    reverse_specs["B-OPTION"] += 1
                else:
                    reverse_specs["B-OPTION"] = 1
                continue
            if v in reverse_specs:
                if spec_counts[reverse_specs[v]] < spec_counts[k]:
                    reverse_specs[v] = k
            else:
                reverse_specs[v] = k
        if "B-OPTION" in reverse_specs:
            reverse_specs["B-OPTION"] = str(reverse_specs["B-OPTION"])
        specs = reverse_specs

        # add I-Items to B-items (e.g. append I-PRIVOD to B-PRIVOD)
        for k, v in specs.items():
            spec_type = re.sub("^.*-", "", k)
            if k.startswith("I-") and f"B-{spec_type}" in specs.keys():
                specs[f"B-{spec_type}"] += f" {v}"
                specs[k] = None
        # remove 'B-' and 'I-'
        specs = {re.sub("^.*-", "", k): v for k, v in specs.items() if v}
        for k, v in specs.items():
            # look for numbers
            value_int = re.findall(r"\d+", v)
            if k != "SHINY" and value_int:
                # to include '0.45' and '0,49'
                # to exclude 'Независимая .,'
                specs[k] = re.findall(r"[0-9.,]+", v)[0]
        specs = {ner_dict[k]: v for k, v in specs.items()}
        additional_info[a_i][2] = specs
    # sha_id : specs
    additional_info = {a[1]: a[2] for a in additional_info}
    return additional_info


def process_done_files(done_files, additional_info=None):
    outputs = []

    cols_to_remove = ["Address", "Num"]
    for d in done_files:
        output = dict()
        sha_id = d[1]
        output["sha_id"] = sha_id
        for car_dict in d[2:]:
            if car_dict:
                # json artifact
                if type(car_dict) == list:
                    car_dict = car_dict[0]
                if "factAddress" in car_dict and car_dict["factAddress"]:
                    car_dict["org_address"] = car_dict[
                        "factAddress"].split(",")[2].strip()
                car_dict = {k: v for k, v in car_dict.items()
                            if not any(c in k for c in cols_to_remove)}
                output.update(car_dict)
        if additional_info and sha_id in additional_info:
            output.update(additional_info[sha_id])
        outputs.append(output)
    return outputs


def clean_outputs(outputs: List[dict]):
    outputs = [{k.lower(): v for k, v in o.items() if k and v}
               for o in outputs]

    all_keys = {key for o in outputs for key in o.keys()}

    normed_keys = {k: " ".join(text_pipeline(k, lemmatize=False))
                    for k in all_keys}
    normed_keys = {k: clean_yargy_str(v) for k, v in normed_keys.items()}
    normed_keys["sha_id"] = "sha_id"
    outputs = [{normed_keys[k]: v for k, v in o.items() if k and v}
               for o in outputs]
    outputs = [{k: v for k, v in o.items() if type(v) != bool and len(k) > 2}
               for o in outputs]
    outputs = [{k: v for k, v in o.items() if k and v and len(o) > 2}
               for o in outputs]
    bad_output_keys = ("размер обеспечения заявки", "ндс")
    for o_i, o in enumerate(outputs):
        print("\t", o_i, len(outputs), end="\r")
        new_output = dict()
        for k, v in o.items():
            if type(v) == list and v:
                v = v[0]
            if k == "sha_id":
                new_output[k] = v
                continue
            if type(v) == str and k != "sha_id":
                if any(v == w for w in
                       ("наличие", "да", "+", "наличии", "есть",
                        "обязательно", "в наличии")):
                    # v = True
                    continue
                v = clean_yargy_str(v, k)
            if k in bad_output_keys:
                continue
            if v and len(k) > 2:
                new_output[k] = v
        old_keys = set(new_output.keys())
        new_output = yargy_cars.get_car_specs_from_catal(new_output)
        new_keys = set(new_output.keys())
        to_remove = []
        if len(new_keys) != len(old_keys):
            for k in new_keys.difference(old_keys):
                # clean yandex catalogue keys
                new_k = clean_yargy_str(k)
                new_output[new_k] = new_output.pop(k)
        outputs[o_i] = new_output
    outputs = [o for o in outputs if len(o) > 2]
    # outputs = [o for o in outputs if "price" in o]
    return outputs


def done_files2excel():
    sql_write.cur.execute(
        f"""select
        *
        from goszakupki_2
        """)
    done_files = sql_write.cur.fetchall()
    done_files = [d for d in done_files if any(d[2:])]
    done_files = process_done_files(done_files)
    all_done_files = dict()
    for d in done_files:
        if len(d) > 1:
            sha_id = d["sha_id"].split("_")[0]
            if "опции" in d:
                d["количество опций"] = len(d["опции"])
                d["опции"] = None
            if "price" in d and type(d["price"]) not in (int, float):
                d["price"] = None
            if "quantity" in d and type(d["quantity"]) not in (int, float):
                d["quantity"] = None
            d = {k: v for k, v in d.items() if v and k not in
                 ("sha_id", "числовые характеристики")}
            if not d:
                continue
            if sha_id in all_done_files:
                all_done_files[sha_id].update(d)
            if sha_id in all_done_files and "price" in d and d["price"] and\
                    all_done_files[sha_id]["price"] and d["price"] !=\
                    all_done_files[sha_id]["price"]:
                break
            else:
                all_done_files[sha_id] = d
    sql_write.cur.execute(
        f"""select
        *
        from goszakupki_2b
        """)
    # WHERE id_sha256 ILIKE '%_{version}';
    additional_info = sql_write.cur.fetchall()
    additional_info = [a for a in additional_info if any(a for a in a[2:])]
    # tuples are immutable
    additional_info = [list(a) for a in additional_info]

    additional_info = convert_add_info(additional_info)
    for k, v in additional_info.items():
        sha_id = k.split("_")[0]
        if sha_id in all_done_files:
            all_done_files[sha_id].update(v)
        else:
            all_done_files[sha_id] = v

    for key, value in all_done_files.items():
        all_done_files[key]["sha_id"] = key
    outputs = list(all_done_files.values())
    # for o in outputs:
    #     if "date" in o and type(o["date"] == list):
    #         # dicts are mutable
    #         o["date"] = o["date"][0]
    all_files = query_all_files(sql)
    all_files = {f[0]: f[1:] for f in files}

    outputs = clean_outputs(outputs)

    outputs = [{k: v for k, v in o.items() if k and v} for o in outputs]

    for o in outputs:
        if any("ё" in k for k in o.keys()):
            for key in o.keys():
                if "ё" in key:
                    new_key = key.replace("ё", "е")
                    if new_key not in o:
                        o[new_key] = o.pop(key)

    outputs = [remove_extra_cols(o) for o in outputs]
    outputs = [o for o in outputs if o]
    outputs = [o for o in outputs if "price" in o]
    outputs = [o for o in outputs if not any(type(v) == list
                                             for v in o.values())]

    small_df = outputs_to_df(outputs)
    small_df["purchase_number"] = small_df["sha_id"].apply(
        lambda x: all_files[x][-1] if x in all_files else None)
    small_df["filename"] = small_df["sha_id"].apply(
        lambda x: all_files[x][-2].split("/")[-1] if x in all_files else None)
    small_df.to_excel("small_goszakupki.xlsx")
    small_df.describe()
    return small_df


def remove_extra_cols(o):
    to_remove = set()
    to_change = dict()
    more_cols = {"шина", "расположение двигателя", "расположение цилиндров",
                 "org address", "страна марки", "класс автомобиля",
                 "тип передней подвески", "тип задней подвески",
                 "тип двигателя", "версия", "название рейтинга", "тип наддува",
                 "экологический класс", "система питания двигателя",
                 "диаметр цилиндра ход поршня", "передние тормоза",
                 "задние тормоза", "размер колес", "расход топлива смешанный",
                 "степень сжатия", "клиренс"}
    for key, value in o.items():
        if key in {"sha_id", "price", "регион", "quantity", "org_address",
                   "date"}:
            continue
        elif key in allowed_keys:
            continue
        elif key in more_cols:
            continue
        # it should be a rare event; thus 'any' is worth
        # the double loop
        elif any(key in n_v for n_v in allowed_keys) or\
                any(n_v in key for n_v in allowed_keys):
            for n_v in allowed_keys:
                if key in n_v or n_v in key and key and len(key) > 3 and\
                            len(n_v) > 3:
                    if n_v not in o or not o[n_v]:
                        to_change[n_v] = key
                    to_remove.add(key)
        else:
            print(key)
            to_remove.add(key)
    for key, value in to_change.items():
        o[key] = o[value]
    o = {k: v for k, v in o.items() if k not in to_remove}
    return o


def outputs_to_df(outputs):
    df = pd.DataFrame(outputs)
    df.to_excel("big_goszakupki.xlsx")
    # delete columns with too sparse values
    df = remove_rare_cols(df)
    # replace rare values (count < 10) from columns with None
    for col_i, col in enumerate(df.columns):
        print("\t", col_i, len(df.columns), end="\r")
        if col in {"sha_id", "date"}:
            continue
        try:
            df[col].astype(float)
            continue
        except (ValueError, TypeError) as e:
            value_counts = df[col].value_counts()
            bad_values = value_counts[value_counts < 10].index
            df[col] = df[col].apply(
                lambda x: x if x not in bad_values else None)
    # delete columns with too sparse values
    df = remove_rare_cols(df)
    df = df.dropna(how="all")
    df["price"] = df["price"].astype(str)
    df["price"] = df["price"].str.replace("-", "")
    df["price"] = df["price"].fillna("0")
    df["price"] = df["price"].apply(lambda x: re.sub(r"[^0-9\.]", "", x))
    df["price"] = df["price"].str.strip()
    df["price"] = df["price"].apply(lambda x: x if x else "0")
    df["price"] = df["price"].astype(float)
    # (df["sha_id"].str.contains("_")) &
    check_columns = {"price", "region", 'год выпуска', "марка",
                     "модель", "длина", "ширина", "объем двигателя",
                     "коробка передач"}
    df[[c for c in df.columns if c in check_columns]].dropna(how="any")
    df.to_excel("goszakupki.xlsx")
    small_df = df[(df["price"] < 100e6) &
                  (df["price"] > 100e3)]
    small_df = remove_rare_cols(small_df)
    small_df.to_excel("small_goszakupki.xlsx")
    return small_df


class YargyWorker:
    def __init__(self, files):
        self.files = files

    def parse(self):
        main_yargy(files=self.files, sql_write=None, sql_main=None)


if __name__ == "__main__":
    # main_yargy(files=files, sql_con=sql_write)

    # workers = WORKERS
    # processes = []
    # if workers > 1:
    #     for i in range(workers):
    #         files_batch = int(len(files) / workers) + 1
    #         worker_files = files[i * files_batch: (i + 1) * files_batch]
    #         yargy_scraper = YargyWorker(files=worker_files)
    #         processes.append(Process(target=yargy_scraper.parse))
    #     for p in processes:
    #         p.start()
    # else:
    #     main_yargy(files=files)
    done_files2excel()


# def lol():
#     import re
#     import os

#     from string import punctuation
#     from collections import Counter
#     from glob import glob
#     from docx import *
#     # from utils import clean_yargy_str
#     from razdel import sentenize, tokenize
#     from sklearn.model_selection import train_test_split
#     # from utils_d.web_utils import libreoffice_convert
#     color_classes = {
#         "yellow": "OPTION",
#         "red": "YEAR",  # год выпуска: 2015
#         "green": "PRICE",
#         "magenta": "MODEL",  # Patriot
#         "cyan": "BRAND",  # UAZ
#         "darkMagenta": "TRANSMISSION",
#         "darkRed": "SPECIFICATION",  # комплектация: Элеганс Плюс
#         "darkGray": "WIDTH",  # ширина: 1830
#         "lightGray": "WEIGHT",  # масса: 1485
#         "black": "LAYOUT",  # привод: (передний, задний)
#         "blue": "QUANTITY",  # количество: 1 шт.
#         "darkGreen": "CONSUMPTION",  # потребление топлива: 9.5 л
#         "darkBlue": "VOLUME",  # объем двигателя: 1.6
#         "darkYellow": "LENGTH",  # длина
#         "darkCyan": "POWER",  # мощность двигателя: 150 л.с.
#         "redFont": "MAXSPEED"  # макс скорость 140 км.ч
#     }
#     # folder = "контракты"
#     folder = "tomaev_annotation"
#     corpus_filename = "ner_corpus/tomaev_annotation_4.txt"
#     docxs = glob(f"{folder}/*.docx")
#     all_files = []
#     for file_i, file in enumerate(docxs):
#         file_dict = dict()
#         print("\t", file_i, end="\r")
#         try:
#             document = Document(file)
#         except Exception as ex:
#             print(ex, file_i)
#             continue
#         words = document.element.xpath('//w:r')

#         prev_text_class = "O"
#         for word_i, word in enumerate(words):
#             xml = word.xml
#             text = word.text
#             if not text:
#                 continue
#             text_class = "O"
#             if "highlight" in xml or 'w:color w:val="FF0000"' in xml:
#                 color = re.findall('(<w:highlight w:val=")(.*)("/>)', xml)
#                 if color:
#                     color = color[0][1]
#                 elif 'w:color w:val="FF0000"' in xml:
#                     color = "redFont"
#                 # # brand only
#                 # if color not in {"magenta", "cyan"}:
#                 #     color = "yellow"
#                 if color in color_classes:
#                     text_class = color_classes[color]
#                     text = " ".join(text.split())
#                     text = re.sub(r"(\d+)\xa0(\d+)", r"\1\2", text)
#                     text = re.sub(r"(\d+) (\d+)", r"\1\2", text)
#                     text = re.sub(r"(\d+) (\d+)", r"\1\2", text)
#                     text = re.sub(r"(\d+) (\d+)", r"\1\2", text)
#                     file_dict[text] = text_class
#                 elif color != "white":
#                     print(
#                         " ".join([w.text for w in
#                                   words[word_i - 2: word_i + 2]]))
#                     print(text, color, text_class)
#         all_files.append(file_dict)


# def done_files2excel_old():
#     all_done_files = dict()
#
#     for version in ALL_VERSIONS:
#         print(version)
#         sql_write.cur.execute(
#             f"""select
#             *
#             from goszakupki_2
#             WHERE id_sha256 ILIKE '%_{version}';
#             """)
#         done_files = sql_write.cur.fetchall()
#         sql_write.cur.execute(
#             f"""select
#             *
#             from goszakupki_2b
#             WHERE id_sha256 ILIKE '%_{version}';
#             """)
#         additional_info = sql_write.cur.fetchall()
#         additional_info = [a for a in additional_info if any(a for a in a[2:])]
#         # tuples are immutable
#         additional_info = [list(a) for a in additional_info]
#
#         additional_info = convert_add_info(additional_info)
#         # d[1] is the sha_id; d[2] is a yargy dict; d[3] is a camelot dict
#         done_files = [d for d in done_files if d[2] or d[3] or
#                       d[1] in additional_info]
#         outputs = process_done_files(done_files, additional_info)
#         # outputs = clean_outputs(outputs)
#         for o in outputs:
#             sha_id = re.sub("_.*", "", o["sha_id"])
#             o["sha_id"] = sha_id
#             if sha_id not in all_done_files:
#                 all_done_files[sha_id] = dict()
#             o = {k: v for k, v in o.items() if type(v) != bool and k not in
#                  ("числовые характеристики", "опция")}
#             # to_remove = set()
#             # to_change = dict()
#             # for k, v in o.items():
#             #     if any(k in allow_key for allow_key in allowed_keys) or\
#             #             any(allow_key in k for allow_key in allowed_keys):
#             #         for allow_key in allowed_keys:
#             #             if k in allow_key or allow_key in k:
#             #                 if allow_key not in o or not o[allow_key]:
#             #                     to_change[allow_key] = k
#             #                     to_remove.add(k)
#             # for key, value in to_change.items():
#             #     o[key] = o[value]
#             # o = {k: v for k, v in o.items() if k not in to_remove}
#             o = remove_extra_cols(o)
#             all_done_files[sha_id].update(o)
#
#     outputs = all_done_files.values()
#     # for o in outputs:
#     #     for k, v in o.items():
#
#     # outputs = [o for o in outputs if
#     #            all(type(v) != list for v in o.values())]
#     # outputs = [o for o in outputs if "_" in o["sha_id"]]
#     small_df = outputs_to_df(outputs)
#     return small_df
