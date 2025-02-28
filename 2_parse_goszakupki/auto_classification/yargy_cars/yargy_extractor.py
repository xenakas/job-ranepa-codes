import os
import json
import re
import pickle

import numpy as np
import pandas as pd
from unidecode import unidecode

from yargy.predicates import (
    dictionary)
from yargy import Parser
from yargy.interpretation import fact
from yargy import (
    rule,
    not_,
    and_,
    or_,
)
from yargy.predicates import (
    eq,
    custom
)
from yargy import predicates
from yargy.pipelines import morph_pipeline
# from yargy.tokenizer import TokenRule
from natasha.extractors import Extractor
from natasha.extractors import MoneyExtractor

from utils_d.flask_wrapper import flask_wrapper
from utils_d.utils import text_pipeline

try:
    path = os.path.dirname(os.path.abspath(__file__))
except NameError:
    path = "/home/denis/ranepa/product_characteristics/yargy_cars"


def prepare_df(df: pd.DataFrame):
    """The format should be a flat pd.Dataframe

    Args:
        df (pd.Dataframe): it should be pre-formatted; column names should be
        in the language the rules are used for (e.g. Russian)

    Returns:
        TYPE: Description
    """
    # numeric_columns = dict()  # key: (min, max) tuple with the value range
    numeric_columns = set()  # keys which have numeric values
    dict_columns = dict()  # key: list of possible values
    text_columns = set()  # key: some free text
    property_columns = set()  # keys which are also values

    filenames = ["numerics", "textuals", "properties"]

    col_order = np.array([df[col].unique().shape[0] for col in df.columns])
    col_index = np.argsort(col_order)[::-1]
    col_order = col_order[col_index]
    df = df[[df.columns[c] for c in col_index]]
    # find columns with numbers

    uncleaned_numerics = set([c for c in df.columns if re.findall("\d+", c)])
    df = df[[c for c in df.columns if c not in uncleaned_numerics]]

    uncleaned_numerics = [
        text_pipeline(c, lemmatize=False) for c in uncleaned_numerics]
    for c_i, c in enumerate(uncleaned_numerics):
        non_first_number = False
        for w_i, w in enumerate(c):
            # if a not_first_number is found, all later words are emptied
            # ['розетка', '220', 'v'] -> ['розетка', '', '']
            # ['2', 'подушки', 'безопасности'] ->
            #   ['', 'подушки', 'безопасности']
            if non_first_number:
                uncleaned_numerics[c_i][w_i] = ""
            elif re.findall("\d+", w):
                uncleaned_numerics[c_i][w_i] = ""
                if w_i != 0:
                    non_first_number = True

    uncleaned_numerics = [" ".join([w for w in c if w]).strip()
                          for c in uncleaned_numerics]
    uncleaned_numerics = set(uncleaned_numerics)
    numeric_columns.update(uncleaned_numerics)

    for col in df.columns:
        done = False
        # to filter out rare values
        col_values = df[col].dropna()
        counts = col_values.value_counts()
        mean = int(counts.mean())
        col_values = counts[counts > mean].index
        if col_values.shape[0] > 2:
            for col_type in [int, float]:
                try:
                    col_values = col_values.astype(int)
                    # numeric_columns[col] = (None, None)
                    numeric_columns.add(col)
                    done = True
                    break
                except (ValueError, TypeError):
                    pass
            if done:
                continue
        # if the number of unique values is in range [2, 50)
        # then we save all possible values in a dict
        length = df[col].dropna().unique().shape[0]
        if length <= 2:
            property_columns.add(col)
        elif length <= 50:
            col_values = counts[counts > min(mean, 1)].index
            # json cannot serialize sets
            dict_columns[col] = list(set(col_values))
        # TODO TEXTUALS
        else:
            pass
    with open("yargy_cars/dict_cols.json", 'w') as f:
        json.dump(dict_columns, f)
    filenames = ["numeric_features", "text_features", "bool_features"]
    filenames = [os.path.join("yargy_cars/", f) for f in filenames]
    for i, property_set in enumerate(
            [numeric_columns, text_columns, property_columns]):
        with open(filenames[i]) as f:
            old_properties = f.readlines()
        old_properties = set([l.strip() for l in old_properties])
        property_set.update(old_properties)
        property_set = {s.lower() for s in property_set}
        with open(filenames[i], "w") as f:
            f.write("\n".join(property_set))
    return None


def read_lines_file(filename, morph=True):
    """Summary

    Args:
        filename (TYPE): Description

    Returns:
        TYPE: Description
    """
    with open(os.path.join(path, filename)) as f:
        output = f.readlines()
    output = [l.strip().lower() for l in output]
    output = [l for l in output if l]
    output = set(output)
    # with open(os.path.join(path, filename), "w") as f:
    #     f.write("\n".join(output))
    if morph:
        output = morph_pipeline(output)
    return output


def show_matches(rule, *lines):
    """Summary

    Args:
        rule (TYPE): Description
        *lines: Description
    """
    parser = Parser(rule)
    for line in lines:
        matches = parser.findall(line)
        matches = sorted(matches, key=lambda _: _.span)
        if matches:
            facts = [_.fact for _ in matches]
            if len(facts) == 1:
                facts = facts[0]
            print(facts)


class YargyRulesBuilder:
    def __init__(self):

        with open("yargy_cars/synonyms") as f:
            synonyms = f.readlines()
        self.synonyms = [set(l.strip().split("\t")) for l in synonyms]

        self.INT = predicates.type('INT')
        self.PUNCT = or_(eq('.'), eq(','), eq('/'), eq('-'), eq("×"), eq("*"),
                        # russian and english x х
                         eq("x"), eq("х"),
                         eq("/"))
        self.RUSSIAN = predicates.type('RU')
        self.LATIN = predicates.type('LATIN')
        self.NUMBERS = read_lines_file("numeric_units")
        self.NOUN_GEN = and_(predicates.gram("NOUN"), predicates.gram('gent'))
        self.ADJ_GEN = and_(predicates.gram("ADJF"), predicates.gram('gent'))
        self.RUBLE = morph_pipeline(["рублей", "руб.", "р." "руб"])
        self.measurements = read_lines_file("measurement_units")
        measures_set = read_lines_file("measurement_units",
                                       morph=False)
        # maybe it's a bad idea to include the year in measurement units =)
        # leads to stuff like this
        measures_set = [m for m in measures_set if "год" not in m]
        measures_regex = [f"(^|\s){m}($|\s)" for m in measures_set]
        self.measures_regex = "|".join(measures_regex)
        self.money_ext = MoneyExtractor()

        self.Car = None

        specifications = pickle.load(open("auto-ru/cars.pcl", "rb"))

        self.specifications = [{k.lower(): v.lower() for k, v in c.items()
                                if type(v) == str}
                               for c in specifications]
        self.all_specification_names = set()
        for s in self.specifications:
            self.all_specification_names.update(s.keys())
        self.all_specification_names = {s for s in self.all_specification_names
                                        if s}

    def build_numeric_feature(self, num, attribute_name=None):
        if not attribute_name:
            attribute_name = num
        item_property = self.Car.__dict__[attribute_name]
        num_rule = rule(
            morph_pipeline([num]),
            self.PUNCT.optional(),
            self.measurements.optional(),
            self.INT,
            self.INT.optional(),
            self.PUNCT.optional(),
            self.INT.optional(),
            self.measurements.optional(),
            self.INT.optional(),
        ).interpretation(item_property)
        return num_rule

    def build_rules(self):
        """
        Returns:
            TYPE: Description

        """

        attributes = ['num_property', 'price',
                      'text_property', "числовые характеристики"]

        # models_rule = rule(brands, LATIN).interpretation(Car.model)

        # read stand-alone property dictionaries
        numerics = read_lines_file("numeric_features", morph=False)
        attributes += numerics

        with open(os.path.join(path, "dict_cols.json")) as f:
            dict_features = f.read()
        dict_features = json.loads(dict_features)
        for key in dict_features.keys():
            attributes.append(key)

        filenames_dict = {
            "модель": "models",
            "марка": "brands",
            "опции": "bool_features"
        }
        # open files; add them to the dict
        # where the key is propery name and the values
        # are morph_pipeline(values)
        list_features = dict()
        for property_name, filename in filenames_dict.items():
            attributes.append(property_name)
            values = read_lines_file(filename, morph=False)
            list_features[property_name] = values
        attributes = list(set(attributes))
        self.Car = fact('Car', attributes)
        price_rule = rule(self.INT, self.RUBLE).interpretation(self.Car.price)

        # properties where the text is a property by itself
        # ABS, подушка безопасности переднего пассажира
        # textual = read_lines_file("text_features")
        # textual_rule = rule(
        #     textual, self.RUSSIAN).interpretation(self.Car.text_property)

        numeric_rules = []
        # new_numerics = set()
        for num in numerics:
            num_rule = self.build_numeric_feature(num)
            numeric_rules.append(num_rule)
            num_set = [s for s in self.synonyms if num in s]
            num_set = {w for s in num_set for w in s}
            # new_numerics.update(num_set)
            for n in num_set:
                num_rule = self.build_numeric_feature(n, attribute_name=num)
                numeric_rules.append(num_rule)
        # numerics.update(new_numerics)
        # item_property = self.Car.__dict__[num]
        # numerics_rule = rule(
        #     morph_pipeline(numerics),
        #     self.PUNCT.optional(),
        #     self.measurements.optional(),
        #     self.INT,
        #     self.INT.optional(),
        #     self.PUNCT.optional(),
        #     self.INT.optional(),
        #     self.measurements.optional(),
        #     self.INT.optional(),
        # ).interpretation(item_property)

        # create dict rules from a dict
        # the dict looks like this
        # {'руль': ['левый', 'правый', 'не требует ремонта'],
        #  'привод': ['задний', 'передний', 'полный']}
        # to match you need a string like "руль левый"
        dict_rules = dict()
        for key, values in dict_features.items():
            item_property = self.Car.__dict__[key]
            new_rule = rule(key, morph_pipeline(values)).\
                interpretation(item_property)
            dict_rules[key] = new_rule

        list_rules = dict()
        # create dict rules from a dict
        # the dict looks like this
        # {'марка': {'acura',
        #            'mercedes-benz',
        #            'alpina'}
        # to match you need a string like "mercedes-benz"
        for key, values in list_features.items():
            item_property = self.Car.__dict__[key]
            new_rule = rule(
                morph_pipeline(values)).interpretation(item_property)
            list_rules[key] = new_rule

        # number_rule = rule(NUMBERS, RUSSIAN, INT, INT.optional())
        number_rule = rule(
            self.NUMBERS,  # количество
            self.ADJ_GEN.optional(),  # стальных
            self.NOUN_GEN,  # поршней
            self.NOUN_GEN.optional(),  # двигателя
            self.ADJ_GEN.optional(),  # внутреннего
            self.NOUN_GEN.optional(),  # сгорания
            self.PUNCT.optional(),  # :
            self.INT,  # 7
            self.PUNCT.optional(),  # -
            self.INT.optional()  # 17
        ).interpretation(self.Car.__dict__["числовые характеристики"])
        price_rule = rule(self.INT, self.RUBLE).interpretation(self.Car.price)

        models_rule = rule(
            morph_pipeline(list_features["марка"]),
            or_(self.LATIN,
                self.INT),
            self.LATIN.optional()).interpretation(
            self.Car.__dict__["модель"])

        CAR_OR = or_(
            models_rule,
            price_rule,
            # textual_rule,
            number_rule,
            # numerics_rule,
            or_(*dict_rules.values()),
            or_(*list_rules.values()),
            or_(*numeric_rules),
        )
        CAR = CAR_OR.interpretation(self.Car)
        extractor = Extractor(CAR)
        self.extractor = extractor
        return extractor

    def extract(self, text):
        # TODO почистить measurement_units из числовых характеристик
        # 'масса кг 3070',
        # 'мощность лошадиных сил 112',
        if not self.extractor:
            return dict()
        extractor = self.extractor
        # text = text_pipeline(text, lemmatize=False)
        # text = " ".join(text)
        matches = extractor(text)
        money_matches = self.money_ext(text)
        if matches:
            properties = matches[0].fact.__attributes__
        else:
            return dict()
        car = dict()
        for prop in properties:
            car[prop] = []
        for match in money_matches:
            fact = match.fact
            price = str(fact.integer)
            car["price"].append(price)
        for match in matches:
            fact = match.fact
            for key in fact.__dict__.keys():
                if fact.__dict__[key]:
                    if key in car:
                        car[key].append(fact.__dict__[key])
        # car["properties"] = set(car["properties"])
        # print(car)
        car = self.clean_yargy(car)
        return car

    def get_car_specs_from_catal(self, car: dict):
        # car["модель"] is a list
        contains_brand = True
        if not("марка" in car or "модель" in car):
            contains_brand = False
        if "модель" in car:
            if type(car["модель"]) == list and car["модель"]:
                model = car["модель"][0].split()
            elif type(car["модель"]) == str:
                model = car["модель"].split()
            else:
                model = []
        else:
            model = []
        if "марка" in car:
            if type(car["марка"]) == list and car["марка"]:
                brand = car["марка"][0]
            else:
                brand = car["марка"]
            # лада to lada; уаз to uaz
            if type(brand) == str:
                brand = unidecode(brand)
            else:
                brand = ""
        else:
            if model:
                brand = model[0]
            else:
                brand = ""

        brand_candidates = [s for s in self.specifications
                            if "описание" in s and
                            len(s["описание"]) >= len(brand) and
                            (brand and brand in s["описание"]) and
                            (model and
                                all(w in s["описание"] for w in model))]
        candidates = brand_candidates
        common_keys = dict()
        for c_k, c_v in car.items():
            if not c_k:
                continue
            if type(c_v) == bool:
                continue
            for k in self.all_specification_names:
                if not k:
                    continue
                if k in c_k or c_k in k:
                    common_keys[c_k] = k
                    break
        spec_candidates = []
        for car_k, spec_k in common_keys.items():
            car_value = car[car_k]
            if not car_value:
                continue
            key_cands = [s_i for s_i, s in enumerate(self.specifications)
                         if car_k in s and car_value in s[car_k]]
            # for s_i, s in enumerate(specifications):
            #     if v in s and car_value in s[v]:
            #         print(v, s[v])
            if key_cands:
                spec_candidates.append(key_cands)
        spec_candidates = [set(s_c) for s_c in spec_candidates]
        if spec_candidates:
            united_candidates = spec_candidates[0]
            for s in spec_candidates[1:]:
                united_candidates = united_candidates.intersection(s)
            candidates += [self.specifications[u] for u in united_candidates]
        else:
            united_candidates = []
        # TOFINISH
        # for key, value in car.items():
        #     if type(value) == str:
        #         pass
        #     elif type(value) == list:
        #         value = value[0]
        #     else:
        #         continue
        #     new_candidates = [c for c in candidates if key in c]
        #     to_drop = []

        #     # leave only candidates with the same info as in the car dict
        #     for c_i, c in enumerate(new_candidates):
        #         keys = [c_key for c_key in c.keys() if key in c_key]
        #         for c_key in keys:
        common_spec = dict()
        if candidates:
            if len(candidates) == 1:
                common_spec = candidates[0]
            else:
                for k, v in candidates[0].items():
                    # check if the values is the same in all specifications
                    if all(v == c[k] for c in candidates[1:] if k in c):
                        common_spec[k] = v
        car.update(common_spec)
        return car

    def clean_numerics(self, car):
        if "числовые характеристики" not in car:
            return car
        nums = car["числовые характеристики"]
        nums = {n for n in nums if not re.match("цен[а-я]", n)}
        car["числовые характеристики"] = nums
        for num in car["числовые характеристики"]:
            # there are akways numbers there because yargy rule
            value = " ".join(re.findall("\d+.*", num))
            key = num.replace(value, "")
            key = re.sub(self.measures_regex, "", key)
            car[key.strip()] = value.strip()
        car.pop('числовые характеристики')
        return car

    def clean_items_number(self, car):
        if "шт" not in car or not car["шт"]:
            return car
        number = car["шт"][0]
        if type(number) == str:
            number = re.findall("\d+", number)
            left_number = number[1:]
            number = number[0]
            car["шт"] = number
            if "price" not in car and left_number:
                left_number = "".join(left_number)
                left_number = int(left_number)
                if left_number > 100000:
                    car["price"] = left_number
        return car

    def clean_price(self, car):
        if "price" not in car or not car["price"]:
            return car
        prices = car["price"]
        if type(prices) in (list, set):
            prices = [re.findall("\d+", p) for p in prices]
            prices = [l for p in prices for l in p]
            prices = [int(p) for p in prices]
            prices = [p for p in prices if 10e4 < p < 50e6]  # 100k<x<50mil
            if prices:
                price = max(prices)
            else:
                price = None
        elif type(prices) == str:
            price = int(re.findall("\d+", prices)[0])
        else:
            price = 0
        if not price:
            car = {k: v for k, v in car.items() if k != "price"}
        else:
            car["price"] = price
        return car

    def clean_brand_model(self, car):
        # "ГАЗ" gets transformed into "газов"
        if "марка" in car:
            car["марка"] = [v for v in car["марка"] if v != "газов"]
        # take the first word of the model
        if "модель" in car and car["модель"] and not car["марка"]:
            model = car["модель"][0]
            model = model.split()[0]
            car["марка"] = [model]
        return car

    def clean_yargy(self, car):
        """
        everything should be a set
        """

        # ширина': {'ширина 1700'} -> ширина': {'1700'}
        for k, v in car.items():
            if type(v) == str:
                v = v.lower()
                v = v.replace(k.lower(), "").strip()
                car[k] = [v]
            # make unique
            if type(v) == list:
                v = list(set(v))
                car[k] = v

        car = self.clean_brand_model(car)

        # Renault Sandero -> Renault Sandero II Рестайлинг
        car = self.get_car_specs_from_catal(car)

        # remove prices from other numerics
        car = self.clean_numerics(car)

        if "опции" in car:
            for o in car["опции"]:
                car[o] = True
            car.pop("опции")

        car = self.clean_price(car)
        car = self.clean_items_number(car)

        # for k, v in car.items():
        #     try:
        #         iter(v)
        #         car[k] = set(v)
        #     except TypeError as te:
        #         if type(v) == str:

        for k, v in car.items():
            if type(v) in (list, set) and len(v) == 1:
                car[k] = list(v)[0]
        car = {k.lower(): v for k, v in car.items() if v}
        return car


text = """
    внимание цена указана с учетом всех акций 40.000 трейд-ин при сдаче старого
    автомобиля 20.000 лада финанс при покупке в кредит 20.000 выгода при
    покупке
    комплектации люкс новая лада гранта лифтбек комплектация люкс кондиционер
    подушка безопасности водителя подушка безопасности переднего пассажира
    подголовники задних сидений 3 шт крепления для детских сидений isofix
    блокировка задних дверей от открывания детьми иммобилайзер охранная
    сигнализация дневные ходовые огни противотуманные фары антиблокировочная
    система тормозов с усилителем экстренного торможения abs bas электронная
    система распределения тормозных усилий ebd электроусилитель рулевого
    управления регулируемая по высоте рулевая колонка регулировка ремней
    безопасности передних сидений по высоте воздушный фильтр салона легкая
    тонировка стекол центральный замок с дистанционным управлением
    электростеклоподъемники передних дверей электростеклоподъемники задних
    дверей
    подогрев передних сидений электропривод и обогрев наружных зеркал
    наружные зеркала
    с боковыми указателями поворота в цвет кузова наружные ручки дверей в цвет
    кузова молдинги боковых дверей вы можете приобрести автомобиль в кредит
    оформление заявки прямо у нас в автосалоне для оформление заявки требуются
    два документа паспорт и водительское заявку можно оформить по телефону
    возможность оформления без каско и первого взноса дополнительное
    оборудование
    по вашему желанию тонировка антикоррозийная обработка защита арок и
    двигателя
    сигнализация музыка и многое другое
    автомобиль в отличном техническом состоянии по кузову вся краска родная
    богатая комплектация кожаный салон диодные фонари климат-контроль круиз
    навигация камера заднего вида и много других приятных опций .птс оригинал
    куплен в ноябре 2012 г редкая и эксклюзивная комплектация match
    оригинальные литые диски o z racing комфорт бортовой компьютер
    усилитель руля датчик дождя климат-контроль обогрев сидений парктроник
    регулировка руля регулируемое по высоте сиденье водителя
    электростеклоподъемники безопасность abs антиблокировочная система esp
    система курсовой устойчивости боковые подушки безопасности водителя и
    переднего пассажира подушка безопасности водителя подушка безопасности
    переднего пассажира экстерьер литые колесные диски обогрев зеркал
    противотуманные фары тонированные стекла электрозеркала мультимедиа
    штатная аудиосистема противоугонные средства сигнализация интерьер
    кожаный салон подлокотник задний подлокотник передний крепления
    iso fix осмотр автомобиля ежедневно с 9-00 до 20-00 по адресу
    г челябинск братьев кашириных 126 автоцентр гольфстрим примем ваш
    автомобиль на комиссию на ваших условиях обратившись к нам вы можете
    быть уверены что вопросом продажи будут заниматься знающие свое дело
    люди и ваше авто вскоре обретет нового хозяина
    """
# for car_rule in CAR_RULE.rules:
#     show_matches(CAR, text)


def test():
    texts = pickle.load(open(
        "/home/denis/ranepa/product_characteristics/goszakupki/all_texts.pcl",
        "rb")
    )
    texts = [s for k, v in texts.items() for s in v]
    texts = [t for t in texts if len(t) > 10]
    builder = YargyRulesBuilder()
    builder.build_rules()
    builder.extract(texts[0])


if __name__ == "__main__":
    yargy_extractor = YargyRulesBuilder()
    yargy_extractor.build_rules()
    flask_wrapper(yargy_extractor.extract)
