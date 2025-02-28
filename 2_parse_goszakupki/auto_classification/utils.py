from typing import List
import pickle
import re
import json

import numpy as np
import pandas as pd

from sql.db_interface import SQL
from sql_config import sql_config
from utils_d.utils import text_pipeline
from characteristics_extractor import CharactericsExtractor
from yargy_cars.yargy_extractor import read_lines_file


extractor = None
numerics = read_lines_file("numeric_features", morph=False)
numerics = set(numerics)
numeric_units = read_lines_file("numeric_units", morph=False)


def flatten_auto_ru_dict(dictfile):
    cars = pickle.load(open(dictfile, "rb"))
    option_names = set()
    for c_i, c in enumerate(cars):
        if "Конфигурация" in c:
            config = c["Конфигурация"]
            config["Цена"] = re.sub("₽.*", "", config["Цена"])
            config["Цена"] = "".join(config["Цена"].split())
            config_elements = dict()
            options = dict()
            categories_to_delete = []
            for key in config.keys():
                v = config[key]
                if type(v) == list:
                    categories_to_delete.append(key)
                    for el in v:
                        if "₽\xa0" not in el:
                            config_elements[el] = True
                        else:
                            el = el.split("\xa0– ")
                            options[el[0]] = "".join(el[1].replace(
                                "\xa0₽", "").split())
                            options[el[0]] = int(options[el[0]])
                elif type(v) == dict:
                    extra_options = [o for o in v.values()]
                    for e_o in extra_options:
                        price = re.sub("₽.*", "", e_o["Цена"])
                        price = "".join(price.split())
                        price = int(price)
                        e_os = e_o["Опции"].split(", ")
                        for e_o_child in e_os:
                            if e_o_child in options:
                                if price < options[e_o_child]:
                                    options[e_o_child] = price
                            else:
                                options[e_o_child] = price
            # for option in config["Опции"]:
            config.update(config_elements)
            config.update(options)
            option_names |= config_elements.keys()
            option_names |= options.keys()
            for cat in categories_to_delete:
                del config[cat]

            cars[c_i].update(config)
            del c["Конфигурация"]
            if "Опции" in c:
                del c["Опции"]
    return cars, option_names


def remove_sparse_columns(t: pd.DataFrame) -> pd.DataFrame:
    to_drop = []
    for col in t.columns:
            col_values = [v for v in t[col] if v]
            if len(col_values) < t.shape[0] / 2:
                to_drop.append(col)
    t = t[[c for c in t.columns if c not in to_drop]]
    return t


def parse_pandas_table(t: pd.DataFrame) -> list:
    """
    t is a pandas DataFrame
    returns a list of dicts
    """

    for c in t.columns:
        t[c] = t[c].apply(lambda x: " ".join(text_pipeline(x, lemmatize=False))
                          if type(x) == str else x)
    # remove empty rows
    t = pd.DataFrame([r for r in t.values if any(l.strip() for l in r
                                                 if type(l) == str)])
    for col in t.columns:
        values = t[col].values
        prev_value = ""
        for v_i in range(1, len(values)):
            v = values[v_i]
            if values[v_i - 1]:
                # else stays the same
                prev_value = values[v_i - 1]
            # previous rows are often in repeated in docx-produced tables
            if v and prev_value:
                values[v_i] = v.replace(prev_value, "")
        t[col] = values
    for col_i in range(1, len(t.columns)):
        col = t.columns[col_i]
        prev_col = t.columns[col_i - 1]
        try:
            t[col] = t.apply(lambda x: x[col].replace(x[prev_col], ""), axis=1)
        except Exception as ex:
            print(ex)

    to_drop = []

    # first remove columns with less than two values
    column_dict = dict()
    for col in t.columns:
        col_values = [v for v in t[col] if v]
        if len(col_values) < t.shape[0] / 2:
            if len(col_values) == 2:
                column_dict[col_values[0]] = col_values[1]
                to_drop.append(col)
            elif len(col_values) <= 1:
                to_drop.append(col)
    t = t[[c for c in t.columns if c not in to_drop]]

    # shift empty cells to the left; questionable; may decrease performance
    # t = pd.DataFrame([[cell for cell in value if cell]
    #                   for value in t.values])
    t = remove_sparse_columns(t)

    if t.shape[0] == 0:
        return []
    first_row = t.values[0]
    first_row_is_index = any(
        c for c in first_row if
        c and type(c) == str and
        re.findall("название|кол-во|описание|комплектация|"
                   "технические характеристики", c)
    )
    # Длина, мм Не менее 4470 not index
    # № Название
    if first_row_is_index:
        t = t.drop([0])
        replace_columns = []
        for c_i, c in enumerate(first_row):
            if c and type(c) == str and c.strip():
                replace_columns.append(f"{c}_{c_i}")
            else:
                replace_columns.append(str(c_i))
        t.columns = replace_columns
    # check if the first column is an index
    if t.columns.shape[0] > 0:
        values = t[t.columns[0]].values
        if values.shape[0] > 0:
            first_col = values
            if all(type(c) == str for c in first_col):
                first_col_is_index = any(c for c in first_col
                                         if re.findall("^\d+\.*\d*", c))
            else:
                first_col_is_index = False
        else:
            first_col_is_index = False
    else:
        return []
    if first_col_is_index:
        t = t.drop(columns=[t.columns[0]])
    # if True:
    #     t = pd.DataFrame(t.values[1:, 1:], columns=t.values[0, 1:])

    # shift empty cells to the left; questionable; may decrease performance
    t = pd.DataFrame([[cell for cell in value if cell]
                      for value in t.values])

    t = remove_sparse_columns(t)
    t = t.fillna("")

    t_dict = t.to_dict()
    t_dicts = []
    if t.columns.size == 2:
        if t.values.shape[0] > 2:
            """
            Товар Кол-во
            Колесо 2
            Упряжь 3
            """
            t_dict = dict(zip(t[t.columns[0]].values, t[t.columns[1]].values))
            to_delete = []
            new_dict = dict()
            for key, value in t_dict.items():
                if not value and type(key) == str:
                    key_split = re.split(":|-|–|;|\t", key)
                    if len(key_split) > 1:
                        new_dict[key_split[0]] = " ".join(
                            key_split[1:]).strip()
                        to_delete.append(key)
            for key in to_delete:
                del t_dict[key]
            if new_dict:
                t_dict.update(new_dict)
        elif t.values.shape[0] == 2:
            """
            Заказчик   Поставщик
            ИНН 1231\n ИНН 13123\n
            Майкоп\n   Спб\n
            Вторый ряд состоит из ненормированных значений в одну строчку
            (по версии библиотеки camelot)
            """
            keys = list(t.values[0])
            values = list(t.values[1])
            # for v_i, v in enumerate(values):
            #     v = re.split(":|–|\t|   ", v)
            #     v = [l for l in v if l]
            #     values[v_i] = v
            t_dict = dict(zip(keys, values))
    # Most cases
    elif t.columns.size > 2 and t.shape[0] > 2:
        # костыль
        # if abs(t.shape[0] - t.shape[1]) >= 2:
        #     t = t.T
        key_col = 0
        keys = t[t.columns[key_col]].values
        if not all(type(k) == str for k in keys):
            return []
        digit_keys = len([k for k in keys if re.findall("^\d+\.*\d*$", k)])
        if digit_keys:
            key_col += 1
        keys = t[t.columns[key_col]].values
        keys = [" ".join(text_pipeline(k, lemmatize=False))
                for k in keys]
        columns_sizes = [
            t[t_i_j][pd.notna(t[t_i_j].replace("", np.nan))].size
            for t_i_j in t.columns[key_col + 1:]]
        sorted_columns = [
            t_i_j for t_i_j in
            t.columns[key_col + 1:][np.argsort(columns_sizes)[::-1]]]
        values = None
        # Unite all rows into one
        for s_i, s in enumerate(sorted_columns):
            if s_i == 0:
                values = t[s]
            else:
                try:
                    values += " " + t[s].astype(str)
                except Exception as ex:
                    print(values, t[s])
                    return []
        values = values.values
        values = [re.sub("(\d+)([А-Яа-я])", r"\1\n\2", v).strip()
                  if type(v) == str else ""
                  for v in values]
        values = [re.sub("([А-Яа-я\.])(\d+)", r"\1-\2", v) for v in values]
        values = [re.sub("(\d+) (\d+)", r"\1\2", v).strip() for v in values]
        values = [re.sub("(\d+) (\d+)", r"\1\2", v).strip() for v in values]
        values = [re.split(";|\n", v) for v in values]
        if all(len(v) == 1 for v in values):
            values = [v[0] for v in values]
            t_dict = dict(zip(keys, values))
        else:
            values = [[w.strip() for w in v] for v in values]
            values = [[re.split(":|-|–", w) for w in v]
                      for v in values]
            for v_i, v in enumerate(values):
                values[v_i] = dictify_list(v)
            values = [dict(v) for v in values]
            for v_i, v in enumerate(values):
                key = keys[v_i]
                values[v_i]["Наименование"] = key
            # !!! t_dicts
            t_dicts = values
        # """
        # 'сумма кол-во ед изм кол-во ед изм кол-во ед изм кол-во ед изм
        #  кол-во ед изм':
        #     '800000,00 1 шт 1 шт 1 шт 1 шт 1 шт           '}

        # """
        # for k_i, k in enumerate(keys):
        #     orig_key = k
        #     v = values[k_i]
        #     k = re.sub("ед\.* изм\.*", "ед.изм", k.lower())
        #     if type(v) != str:
        #         break
        #     v = re.sub(
        #         "ед\.* изм\.*", "ед.изм", v.lower()).replace(" шт", " ")
        #     k = k.split()
        #     v = v.split()
        #     if len(k) >= 6 and len(v) >= 6:
        #         for k_child_i, k_child in enumerate(k):
        #             if k_child_i < len(v):
        #                 t_dict[k_child.strip()] = v[k_child_i].strip()
        #         del t_dict[orig_key]
    elif t.columns.size > 2 and t.shape[0] == 2:
        keys = t.values[0]
        values = t.values[1]
        t_dict = dict(zip(keys, values))
    # One row table
    elif t.shape[0] == 1:
        values = t.values[0]
        values = [v for v in values if v]
        if values:
            if values[0].isdigit():
                values = values[1:]
            new_dict = dict()
            new_dict["Наименование"] = values[0]
            values = "\n".join(values[1:])
            values = re.split(";|\t|\n|\uf0b7", values)
            values = [v.strip() for v in values]
            values = [re.split("-|–| :", v) for v in values]
            values = dictify_list(values)
            values = dict(values)
            new_dict.update(values)
            t_dict = new_dict
    if not t_dict:
        t_dict = t.to_dict()
    if not t_dicts:
        t_dicts = [t_dict]
    for t_dict_i, t_dict in enumerate(t_dicts):
        t_dict = {k: v for k, v in t_dict.items() if k}
        to_delete = []
        for k, v in t_dict.items():
            if type(k) == str:
                # 098.,.212 - Numeric keys are invalid
                if re.match("^\d+[\., ]*\d*$", k.strip()):
                    to_delete.append(k)
        t_dict = {k: v for k, v in t_dict.items() if k not in to_delete}
        # {'Тип кузова': 'Кроссовер Тип кузова',
        #  'Цвет кузова': 'Белый Цвет кузова',
        for k, v in t_dict.items():
            if not v:
                v = "наличие"
            if type(v) == str:
                t_dict[k] = v.replace(k, " ").strip()
        # Flatten
        keys_to_del = set()
        new_dict = dict()
        for key, value in t_dict.items():
            if type(value) == dict and len(value) == 1:
                keys_to_del.add(key)
                new_dict.update(value)
        t_dict.update(new_dict)
        for key, value in t_dict.items():
            if type(key) == str and len(key) > 100:
                keys_to_del.add(key)
            if type(value) == str:
                if len(value) > 300:
                    keys_to_del.add(key)
                else:
                    # value = re.sub("(\w+ \w+ )(\w+ \w+ )", r"\1", value)
                    # value = re.sub("(\w+ \w+ )(\w+ \w+ )", r"\1", value)
                    # 'Кроссовер типа комфорт типа комфорт' -> 'Кроссовер'
                    value = re.sub(r"(.*?)(\w+ \w+) \2", r"\1", value).strip()
                    t_dict[key] = value
        for key in keys_to_del:
            del t_dict[key]
        t_dicts[t_dict_i] = t_dict
    if column_dict and len(t_dicts) == 1:
        t_dicts[0].update(column_dict)
    return t_dicts


def filter_pandas_tables(tables: List[pd.DataFrame]) -> List[dict]:
    global extractor
    if not extractor:
        extractor = CharactericsExtractor()
    output = []
    for t_i, t in enumerate(tables):
        t_dicts = parse_pandas_table(t)
        for table in t_dicts:
            result = extractor.filter_dict(table)
            if len(result) > 1:
                output.append(table)
    return output


def dictify_list(somelist):
    values = []
    for w_i, w in enumerate(somelist):
        # len(w) == 2 is OK
        if len(w) > 2:
            w = [w[0], " ".join(w[1:])]
        elif len(w) == 1:
            w = [w[0], ""]
        values.append(w)
    return values


def parse_goszakupki_text(text):
    # text = text.split("\n\n\n")
    # text = [p for p in text if p]
    # text = [text_pipeline(p, lemmatize=False) for p in text]
    # for p_i, p in enumerate(text):
    #     p = " ".join(p)
    #     p = re.sub("(\d+) млн\.*", r"\1 000000", p)
    #     p = re.sub("(\d+) тыс\.*", r"\1 000", p)
    #     p = re.sub("(\d+) (\d+)", r"\1\2", p)
    #     text[p_i] = p.split()
    parsed = [extractor.parse_text(p) for p in text]
    parsed = [p for p in parsed if p]
    # Hypothesis is that at lest 1 category should contain a number
    parsed = [p for p in parsed if
              any(re.findall("\d+", v) for v in p.values())]
    return parsed


def init_sql():
    sql = SQL(**sql_config)
    sql_config_write = sql_config.copy()
    sql_config_write["dbname"] = "business_dynamics_db"
    sql_write = SQL(**sql_config_write)
    return sql, sql_write, sql_config_write


def get_table_ids(sql_write):
    sql_write.cur.execute(
        """
        select * from goszakupki_2 where characteristics_pandas::text <> '{}'
        """
    )
    data = sql_write.cur.fetchall()
    data = [list(d) for d in data if d[3]]

    for d_i, d in enumerate(data):
        car = d[3]
        if type(car) == list:
            data[d_i][3] = car[0]
    data = [[d[1], d[3]] for d in data]
    data = [d for d in data if any(d[1].values())]
    ids = {re.sub("_.*", "", d[0]): json.dumps(d[1]) for d in data}
    return ids


def replacements(v):
    v = re.sub(
        "л.с.$| л$| км$|^от |л. с.|шт. |н•м | кг$|"
        "мм|^л |рублей$| руб | руб. |км ч| руб$| рубль| р уб| рубля| рублю|"
        "должен быть|"
        "не менее|не менее|не более|объем топливного бака| л |мощность|"
        "мощностью|"
        "квт|колесная база|не ранее|года| г| г$|, л$|, см³| такого|см³|",
        "", v)
    return v


def replacements_with_dict(v):
    replacement_dict = {
        "или эквивалент": "",
        "эквивалент": "",
        "бензин с октановым числом 92": "аи-92",
        "^шт$": "штуки",
        "рабочий объем": "объем",
        "багажного отделения": "багажника",
        "антиблокировочная система тормозов$": "abs",
        "антиблокировочная система тормозов abs": "abs",
        "антиблокировочная система abs": "abs",
        "антиблокировочная система$": "abs",
        "числе ндс": "ндс",
        "число ": "количество ",
        "датчики парковки": "парктроник",
        "цены контракта": "price",
        "максимальная с$": "максимальная скорость",
        "^двигателя$": "двигатель",
        "^двигателяc$": "двигатель",
        "иммобилайзера": "иммобилайзер",
        "экологический к": "экологический класс",
        "^цвет$": "цвет кузова",
        "люки": "люк",
        " л$": "",
        " нм": "",
        " об ": "",
        " мин$": "",
        " мм": "",
        " н ": "",
        " м$": "",
        " мин": "",
        "кол-в$": "количество",
        "кол-во": "количество",
        "иобилайзер": "иммобилайзер",
        "гидроусилителем руля": "гидроусилитель руля",
        "^выпуска$": "год выпуска",
        "размер обеспечения исполнения$": "размер обеспечения исполнения "
        "контракта",
        "объем двигате$": "объем двигателя",
        "бортового компьютера": "бортовой компьютер",
        "^двигат$": "двигатель",
        "дневным ходовым огнем": "дневные ходовые огни",
        "^итого$": "price",
        "количество посадочных мест": "количество мест",
        "количества товара": "количество товара",
        r"^коробка$": "коробка передач",
        "легкосплавном диске": "легкосплавные диски",
        "^наименование$": "наименование товара",
        "внешние зеркала": "наружные зеркала",
        "наружных зеркал": "наружные зеркала",
        "мин макс": "",
        ", л$": "",
        "омыватель фар": "омыватели фар",
        "опция": "опции",
        'размер обеспечения$': 'размер обеспечения заявки',
        'размер обеспечения исполнения контракта': 'размер обеспечения заявки',
        "размере обеспечения указанной заявки": 'размер обеспечения заявки',
        "регулируемая по высоте рулевая колонка": 'регулировка руля по высоте',
        'рулевого ко': 'рулевое ко',
        'рулевое управление': 'рулевое ко',
        "цена контракта": "price",
        'центральным замком': 'центральный замок',
        "числе": "количество",
        "электростеклоподъёмники передних$": "электростеклоподъемники "
        "передних дверей",
        "центрального замка": "центральный замок",
        "^привод$": "тип привода",
        r"^коробка$": "коробка передач",
        "крепления для детских сидений$": "крепления для детских "
        "сидений isofix",
        "масса автомобиля": "масса",
        "тип кузова": "кузов",
        ", кг": "",
        "полная масса": "масса",
        "разгон до 100 км / ч, с": "разгон",
        "максимальная мощность, л.с./квт при об/мин": "мощность",
        "максимальный крутящий момент, н*м при об/мин": "крутящий момент",
        "см³": "",
        "максимальная скорость, км / ч": "скорость",
        "снаряженная масса, кг": "масса",
        "цена": "price",
        "размер": "",


    }
    for key, value in replacement_dict.items():
        v = re.sub(key, value, v)
    return v


def clean_yargy_str(v, k=None):
    v = v.lower()
    v = v.replace("не менее", "")
    v = v.replace("не более", "")
    v = v.replace("предпочтительно", "")
    v = v.strip()
    if k:
        v = v.replace(k, "")
    v = v.strip()
    v = v.replace('₽', "")
    v = re.sub(r"(\d+) (\d+)", r"\1\2", v)
    v = re.sub(r"(\d+) (\d+)", r"\1\2", v)
    v = re.sub(r"(\d+) (\d+)", r"\1\2", v)
    v = replacements(v)
    v = replacements_with_dict(v)
    v = v.strip()
    if k and ("количество" in k or any(num in k for num in numerics) or
              any(num in k for num in numeric_units)):
        v = re.sub(r"[^0-9\.]", "", v)
    v = text_pipeline(v, lemmatize=False)
    v = [w for w in v if len(w) >= 3]
    v = " ".join(v)
    v = replacements(v)
    v = replacements_with_dict(v)
    v = v.strip()
    v = " ".join(text_pipeline(v, lemmatize=False))
    # only for keys
    if len(v.split()) == 1 and not k:
        v = " ".join(text_pipeline(v, lemmatize=True))
    return v


ner_dict = {
    "BRAND": "марка",
    "CONSUMPTION": "потребление топлива",
    'DLINA': "длина",
    'DOROZHNYI_PROSVET': "дорожный просвет",
    'DVIGATEL': "двигатель",
    'GOD_VYPUSKA': "год выпуска",
    'LAYOUT': "привод",
    "LENGTH": "длина",
    'KOLESNAIA_BAZA': "колесная база",
    'KOLESNAIA_FORMULA': "колесная формула",
    'KOLICHESTVO_DVEREI': "количество дверей",
    'KOLICHESTVO_MEST': "количество мест",
    'KOLICHESTVO_PEREDACH': "количество передач",
    'KOLICHESTVO_TSILINDROV': "количество цилиндров",
    'KOROBKA_PEREDACH': "коробка передач",
    'KRUTIASHCHII_MOMENT': "крутящий момент",
    'KUZOV': "кузов",
    "MAXSPEED": "максимальная скорость",
    "MODEL": "модель",
    'MOSHCHNOST': "мощность",
    'OBEM': "объем",
    'OBEM_DVIGATELIA': "объем двигателя",
    'OBEM_TOPLIVNOGO_BAKA': "объем топливного бака",
    'OPTION': "количество опций",
    "POWER": "мощность",
    "PRICE": "цена",
    'PRIVOD': "привод",
    "QUANTITY": "количество",
    'RULEVOE_UPRAVLENIE': "рулевое управление",
    'SHINY': "шины",
    'SHIRINA': "ширина",
    'SKOROST': "скорость",
    "SPECIFICATION": "комплектация",
    'TOPLIVO': "топливо",
    'TORMOZNAIA_SISTEMA': "тормозная система",
    'TRANSMISSION': "коробка передач",
    'TSVET': "цвет",
    'VOLUME': "объем двигателя",
    'VYSOTA': "высота",
    "WEIGHT": "масса",
    "WIDTH": "ширина",
    "YEAR": "год выпуска"
}


def yargy_json_to_csv():
    with open("yargy_cars/bool_features") as f:
        bools = f.readlines()
    with open("yargy_cars/measurement_units") as f:
        measurements = f.readlines()
    measurements = [m.strip() for m in measurements]
    measurements = [" ".join(text_pipeline(m, lemmatize=False))
                    for m in measurements]
    measurements = [m for m in measurements if "год" not in m]
    measures_regex = [rf"(^|\s){m}($|\s)" for m in measurements]
    measures_regex = "|".join(measures_regex)
    bools = [b.strip() for b in bools]
    sql_write.cur.execute(
        f"""select
        characteristics_yargy,
        characteristics_pandas
        from goszakupki_2;
        """)
    texts = sql_write.cur.fetchall()
    texts = [t[0] for t in texts]
    for t_i, t in enumerate(texts):
        if "числовые характеристики" in t:
            nums = t["числовые характеристики"]
            nums = {n for n in nums if not re.match("цен[а-я]", n)}
            t["числовые характеристики"] = nums
            for num in t["числовые характеристики"]:
                # there are akways numbers there because yargy rule
                value = " ".join(re.findall(r"\d+.*", num))
                key = num.replace(value, "")
                key = re.sub(measures_regex, "", key)
                t[key.strip()] = value.strip()
            t.pop('числовые характеристики')
        t = {re.sub(measures_regex, "", k): v for k, v in t.items()}
        # if "опции" in t:
        #     options = [b for b in t["опции"] if b in bools]
        #     for o in options:
        #         t[o] = True
        #     t.pop("опции")
        for k, v in t.items():
            if type(v) == list and len(v) == 1:
                t[k] = v[0]
        if "модель" in t and "марка" in t:
            model = t["модель"][0].split()
            brand = t["марка"]
            if len(brand) == 1:
                brand = brand[0]
            else:
                # a bug where brands look like ['s', 'm', 'a', 'r', 't']
                brand = "".join(brand)
            t["марка"] = brand
            candidates = [s for s in specifications
                          if brand in s["описание"] and
                          all(w in s["описание"] for w in model)]
            common_spec = dict()
            filter_keys = set()
            for c in candidates:
                keys = set(c.keys())
                keys = {k for k in keys if k in t}
                filter_keys.update(keys)
            if candidates:
                for k in filter_keys:
                    filter_candidates = [c for c in candidates if
                                         k in c and c[k] == t[k]]
                    if filter_candidates:
                        candidates = filter_candidates
                if len(candidates) == 1:
                    common_spec = candidates[0]
                else:
                    for k, v in candidates[0].items():
                        # check if the values is the same in all specifications
                        if all(v == c[k] for c in candidates[1:] if k in c):
                            common_spec[k] = v
                if common_spec:
                    # print(model, brand, common_spec)
                    texts[t_i].update(common_spec)
        t = {k: v for k, v in t.items() if k and v}
        texts[t_i] = t
    texts = [t for t in texts if t]

    # filter + to excel
    df = pd.DataFrame(texts)
    cols = [c for c in df.columns if df[c].dropna().shape[0] > 10]
    df = df[[c for c in cols]]
    cols = [df[c].dropna().shape[0] for c in df.columns]
    cols = np.argsort(cols)[::-1]
    df = df[[df.columns[c] for c in cols]]
    df.to_excel("yargy_cars/all.xlsx")

    df = df.loc[df["модель"].dropna().index]
    df.to_excel("yargy_cars/model.xlsx")


def prepare_toloka():
    toloka_f = open("toloka.csv", "a")
    toloka_f.write(
        "INPUT:text\tGOLDEN:key\tGOLDEN:value\tHINT:text\n")
    for f in files:
        sha_id, filename = get_s3_file(f)
        # text, tables = process_file(filename, None, get_tables=True,
        #                             to_delete=True,
        #                             folder_name="./s3/")
        text = process_file(filename, None, get_tables=False,
                            to_delete=True,
                            folder_name="./s3/")
        text = text.split("\n")
        # text = [t for t in text if t and "|" in t]
        yargy_out = [parse_yargy(t, to_json=False) for t in text]
        for y_i, y in enumerate(yargy_out):
            if not y:
                continue
            if "price" in y.keys():
                try:
                    prices = [re.findall(r"\d+", price)
                              for price in y["price"]]
                    prices = [float(p[0]) for p in prices if p]
                except Exception as ex:
                    pass
                if not any(p > 100000 for p in prices):
                    yargy_out[y_i] = None
                    # print(y)
        index = [i for i, o in enumerate(yargy_out) if o]
        all_lines = []
        limit = 5
        for ind_i, i in enumerate(index):
            if ind_i > 0:
                lower_limit = max(i - limit, index[ind_i - 1])
            else:
                # not to include text[-2:]
                lower_limit = max(index[ind_i - 1], 0)
            if ind_i != len(index) - 1:
                upper_limit = min(i + limit, index[ind_i + 1])
            else:
                # no need to check
                upper_limit = i + limit
            toloka_lines = text[lower_limit: i]
            if len(toloka_lines) == 1:
                toloka_lines = []
            toloka_lines += text[i + 1: upper_limit]
            if len(toloka_lines) == 1:
                continue
            all_lines += toloka_lines
        all_lines = list(set(all_lines))
        for line in all_lines:
            if line:
                line = " ".join(line.split())
                line = line.strip()
                if len(line) < 10:
                    continue
                if len(text_pipeline(line, lemmatize=False)) < 3:
                    continue
                toloka_f.write(f"{line}\t\t\t\n")


def goszakupki2yargy():
    sql_write.cur.execute("""
        select * from goszakupki_2
        where characteristics_pandas::text <> '{}'""")
    pandas_f = sql_write.cur.fetchall()
    pandas_f = [p[3] for p in pandas_f]
    pandas_f = [p[0] if type(p) == list else p for p in pandas_f]
    pandas_f = [p for p in pandas_f if p]
    df = pd.DataFrame(pandas_f)
    df = df[[c for c in df.columns if
             c not in numeric_keys and c not in numerics]]
    for c in df.columns:
        df[c] = df[c].apply(lambda x: None if type(x) == bool and x else x)
    df = df.dropna(how="all")
    for c in df.columns:
        value_counts = df[c].value_counts()
        value_counts = value_counts[value_counts > 10]
        value_counts = value_counts.index
        df[c] = df[c].apply(lambda x: x if x in value_counts else None)
    df = df[[c for c in df.columns if df[~df[c].isna()].shape[0] > 10]]
    df = df[[c for c in df.columns if df[~df[c].isna()].shape[0] > 10]]

    float_cols = []
    for c in df.columns:
        try:
            df[~df[c].isna()][c].astype(float)
            float_cols.append(c)
        except ValueError:
            pass
    print("new numerics")
    print("\n".join(float_cols))
    df = df[[c for c in df.columns if c not in float_cols]]

    with open("yargy_cars/dict_cols.json") as f:
        r = f.read()
    car_dict = json.loads(r)
    car_dict = {k: set(v) for k, v in car_dict.items()}
    for c in df.columns:
        print(c)
        new_values = set(df[c].dropna().unique())
        new_values = {n for n in new_values if n}
        if c in car_dict:
            car_dict[c] = car_dict[c].update(new_values)
        else:
            car_dict[c] = new_values
    # json doesnt accept sets
    car_dict = {k: list(v) for k, v in car_dict.items() if v}
    car_json = json.dumps(car_dict)
    with open("yargy_cars/dict_cols.json", "w") as f:
        f.write(car_json)
