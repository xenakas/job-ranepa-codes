import re
import gc
import pickle
import numpy as np
import pandas as pd

from rnnmorph.predictor import RNNMorphPredictor
from utils_d.utils import text_pipeline, list_diff, parse_rnn_morph_tags


class CharactericsExtractor:
    def __init__(self, dictfile="auto-ru/cars.pcl",
                 bigrams_file="normed_bigrams.pcl"):
        self.predictor = RNNMorphPredictor(language="ru")
        self.comparison_words = {"не", "тысяча", "тысяч", "мин", "макс"}
        self.stopwords = self.load_stoplines("stopwords.txt")
        self.stop_categories = self.load_stoplines("stop_categories.txt")
        option_names = set()
        cars = [dict(), dict(), dict(), dict()]
        if dictfile:
            cars = pickle.load(open(dictfile, "rb"))

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
        reverse_cars = dict()
        for c in cars:
            for k, v in c.items():
                if type(v) != str:
                    continue
                if "Ссылка" not in k and not v.isdigit() and\
                        len(v.split()) <= 5 and\
                        not re.findall("\d+\.\d+", v):
                    if v not in reverse_cars:
                        reverse_cars[v] = set([k])
                    else:
                        reverse_cars[v].add(k)
                        # "Мощность" not in k and\
                        # "Расход" not in k and\
                        # "Объем" not in k and\
                        # "Разгон" not in k and not\
        if self.stop_categories:
            for c_i, c in enumerate(cars):
                for s_c in self.stop_categories:
                    if s_c.lower() in c.keys():
                        del c[s_c]
            for cat in self.stop_categories:
                if cat in option_names:
                    option_names.remove(cat)
        normed_bigrams = pickle.load(open(bigrams_file, "rb"))
        normed_bigrams = [w[0] for w in normed_bigrams.most_common(600)]
        normed_bigrams = [" ".join(w.split("_")).lower()
                          for w in normed_bigrams]
        if self.stop_categories:
            for b_i, b in enumerate(normed_bigrams):
                if b in self.stop_categories:
                    normed_bigrams[b_i] = None
            normed_bigrams = [b for b in normed_bigrams if b]
        options_lower = {o.lower() for o in option_names}
        car_keys_lower = {k.lower() for c in cars for k in c.keys()}

        # For options names 'cost' is the only parameter
        non_option_names = {c for c in car_keys_lower
                            if c not in options_lower}
        # int_options = {k for c in cars for k in c.keys()
        #                if type(c[k]) == int}

        # bad_car_keys = [k for k_i, k in enumerate(car_keys_lower)
        #                 if any(k in w for w in
        #                        car_keys_lower[:k_i] +
        #                        car_keys_lower[k_i + 1:])]
        # print("Repetitions in car_keys")
        # for k in bad_car_keys:
        #     for c_k in car_keys_lower:
        #         if k in c_k:
        #             if k and c_k and k != c_k:
        #                 print(k, ";", c_k)

        normed_bigrams = [b for b in normed_bigrams
                          if not any(b == w for w in car_keys_lower)]
        # Normed bigrams have only bool parameter as well as option_names
        # Non-option names may have value parameters
        normed_bigrams = [b for b in normed_bigrams
                          if not any(b in w for w in non_option_names)]
        options_lemmed = [self.predictor.predict(o.split())
                          for o in options_lower]
        options_lemmed = [" ".join([w.normal_form for w in o])
                          for o in options_lemmed]
        # Add "2 " to remove difference cases like "крышки багажника"
        # (it should be genetive)

        bigrams_lemmed = [self.predictor.predict(('2 ' + b).split())[1:]
                          for b in normed_bigrams]
        bigrams_indices = [b_i for b_i, b in enumerate(bigrams_lemmed) if
                           b[0].pos not in ["ADP", "CONJ"] and
                           b[-1].pos not in ["ADP", "CONJ"]]
        bigrams_lemmed = [bigrams_lemmed[b_i] for b_i in bigrams_indices]
        normed_bigrams = [normed_bigrams[b_i] for b_i in bigrams_indices]
        bigrams_lemmed = [" ".join([w.normal_form for w in o])
                          for o in bigrams_lemmed]
        bad_options = set()
        for b_i, b in enumerate(normed_bigrams):
            if any(w for w in option_names if w.lower() in b):
                bad_options.add(b)
                continue
            if any(bigrams_lemmed[b_i] in o for o in options_lemmed):
                bad_options.add(b)
                continue
            if any(b in w.lower() for w in option_names):
                morphed = self.predictor.predict(('2 ' + b).split())[1:]
                nouns = False
                for m in morphed:
                    if m.pos == "NOUN":
                        nouns = True
                        break
                if not nouns:
                    continue
                tags = [parse_rnn_morph_tags(m.tag) for m in morphed]
                nominative = False
                for t in tags:
                    if "case" in t and t["case"] == "nom":
                        nominative = True
                if not nominative:
                    bad_options.add(b)
        normed_bigrams = [b for b in normed_bigrams if b not in bad_options]

        # normed_bigrams = [b for b in normed_bigrams if b]
        # print("Repetitions in bigrams")
        # for k in normed_bigrams:
        #     for c_k in car_keys_lower:
        #         if k in c_k:
        #             if k and c_k and k != c_k:
        #                 print(k, ";", c_k)
        value_categories = [
            '№ п/п',
            'наименование товара',
            'ед. изм.',
            'кол.',
            'страна происхождения',
            'характеристика товара',
            'цена, руб.',
            'сумма, руб.']

        my_value_strings = {"км", "год", "цвет ", "тыс", "тип ", "цена "}
        for b_i, b in enumerate(normed_bigrams):
            if any(m_v in b for m_v in my_value_strings):
                normed_bigrams[b_i] = None
                value_categories.append(b)

        normed_bigrams = [b for b in normed_bigrams if b]
        print(len(value_categories))
        print("Value bigrams len", value_categories)
        print("Normed bigrams len", len(normed_bigrams))
        for w in normed_bigrams:
            cars[0][w] = False

        # a clutch to get a range values for each w
        for w in value_categories:
            cars[0][w] = "100"
            cars[1][w] = "200"
        # for c_i, c in enumerate(cars):
        #     if "сидения" in c:
        #         del cars[c_i]["Спортивные передние сидения"]
        option_names |= set(normed_bigrams)
        self.option_names = option_names

        # all_names = non_option_names + option_names
        # all_names_piped = [text_pipeline(w, lemmatize=False)
        #                    for w in all_names]

        cars_df = pd.DataFrame(cars)

        value_range = [[]] * len(cars_df.columns)
        for column_i, column in enumerate(cars_df.columns):
            values = set(cars_df[pd.notna(cars_df[column])][column].values)
            if True in values:
                values.remove(True)
            value_range[column_i] = values
        self.value_range = value_range
        # str(int("1821")) == "1821"
        inverse_dict = dict()
        for v_r_i, v_r in enumerate(value_range):
            for value in v_r:
                if type(value) != int and\
                        cars_df.columns[v_r_i] not in "ОписаниеВерсия":
                    try:
                        if value != str(int(value)):
                            inverse_dict[value] = v_r_i
                    except Exception as ex:
                        inverse_dict[value] = v_r_i
        number_columns = set()
        for v_r_i, v_r in enumerate(value_range):
            if not cars_df.columns[v_r_i]:
                continue
            if cars_df.columns[v_r_i] in self.option_names:
                continue
            if len(v_r) > 2:
                if all(type(v) in [int, float] for v in v_r):
                    number_columns.add(cars_df.columns[v_r_i])
                # If all elements are numbers
                elif all(type(v) == str for v in v_r) and\
                    all(len(re.findall("^\d+\.*\d*$", v)) == 1
                        for v in v_r):
                    number_columns.add(cars_df.columns[v_r_i])
        self.number_columns = number_columns
        # ranged_values = [i for i in range(len(value_range))
        #                  if len(value_range[i]) > 1]
        self.columns = cars_df.columns.values
        # self.columns = list(self.columns) + list(reverse_cars.keys())
        # for key, value in reverse_cars.items():
        #     if len(value) > 1:
        #         self.option_names.add(key)
        self.columns_piped = [text_pipeline(w, lemmatize=False)
                              for w in self.columns]
        self.c_normed = [[w.normal_form for w in self.predictor.predict(c)]
                         for c in self.columns_piped]

        del cars_df, inverse_dict, normed_bigrams, cars
        gc.collect()

    def get_child_phrase(self, diff, text):
        additional_phrases = []
        diff_range = diff
        added = False
        add_word = None
        # Add additional word for correct pos-tagging
        if diff[0] != 0:
            diff_range = [diff[0] - 1] + diff
            added = True
        forms = self.predictor.predict([text[i] for i in diff_range])
        if added:
            forms = forms[1:]
        nouns = [f for f in forms if f.pos == "NOUN"]
        noun_tags = [parse_rnn_morph_tags(n.tag) for n in nouns]
        # last_noun = None
        if nouns:
            # last_noun = nouns[-1]
            ln_tags = noun_tags[-1]
        else:
            return [], diff, additional_phrases
        if not any(n["case"] == "nom" for n in noun_tags) and diff[0] != 0:
            diff = [diff[0] - 1] + diff
            add_word = self.predictor.predict([text[diff[0] - 1]])[0]
            add_word_tags = parse_rnn_morph_tags(add_word.tag)
            # if add_word.pos == "NOUN":
            #     ln_tags = add_word_tags
        not_stop = True
        child_phrase = []
        next_i = diff[-1] + 1
        # Look for a noun, or an dj in the same case
        prev_pos = None
        prev_word = None
        pw_tags = None
        first_noun_conjunction = False
        noun_required = False
        while not_stop:
            add_to_child_phrase = True
            if len(text) <= next_i:
                not_stop = False
                break
            next_word = self.predictor.predict([text[next_i]])[0]
            nw_tags = parse_rnn_morph_tags(next_word.tag)
            # break after adjective phrase
            if prev_pos == "ADJ":
                if next_word.pos in ["NOUN", "VERB"] and\
                        not first_noun_conjunction and not noun_required:
                    not_stop = False
                    break
                # If this and prev. word are adjectives and have different
                # cases, then break -> цвет черный [break] передних
                elif next_word.pos == "ADJ" and pw_tags and\
                    "case" in pw_tags and nw_tags and "case" in nw_tags and\
                        pw_tags["case"] != nw_tags["case"]:
                    not_stop = False
                    break
            if next_word.word in self.stopwords:
                pass
            # or not next_word.pos
            elif (next_word.pos == "NOUN"):
                noun_required = False
                if not first_noun_conjunction:
                    not_stop = False
                    first_noun_conjunction = False
                else:
                    first_noun_conjunction = False
                    add_to_child_phrase = False
                    additional_phrases.append(next_i)
                # print(prev_pos)
            elif next_word.pos == "VERB":
                if prev_pos == "ADJ":
                    not_stop = False
            # Расход: в смешанном ??? - NOUN!
            elif next_word.pos == "ADP":
                noun_required = True
            # elif "ADJ" in next_word.pos:
            #     # Last Noun
            #     if "case" in nw_tags and\
            #             nw_tags["case"] == ln_tags["case"]:
            #         not_stop = False
            elif next_word.pos in ["NUM", "ADV"] and\
                    next_word.word != "очень" and not noun_required:
                not_stop = False
            elif next_word.pos == "CONJ":
                if not child_phrase:
                    first_noun_conjunction = True
            # English words
            elif not text_pipeline(next_word.word,
                                   latin=False) and\
                    next_word.pos == "PUNCT":
                not_stop = False
            # prev_pos = next_word.pos
            if next_word.pos != "PRON" and add_to_child_phrase:
                child_phrase.append(next_i)
            # prevent cases like 'мощность': 'л'
            # prevent cases like 'мощность': 'л c не менее'
            # prevent cases like 'мощность': 'л c не выше'
            # prevent cases like 'мощность': 'л с к вт об мин'
            if not not_stop and\
                    all((len(text[w]) < 4 or text[w].endswith("е") or
                         text[w] in self.comparison_words) and
                        text[w].isalpha() for w in child_phrase):
                not_stop = True
            next_i += 1
            prev_pos = next_word.pos
            prev_word = next_word
            pw_tags = nw_tags
        return child_phrase, diff, additional_phrases

    def find_characteristics(self, text, column_id, text_normed=""):
        text_characteristics = dict()
        column_piped = self.columns_piped[column_id]
        column = self.columns[column_id]
        diff = list_diff(text_normed, self.c_normed[column_id],
                         return_intersection=True, unite_sequences=True)
        if not diff:
            return text_characteristics
        diff = diff[np.argmax([len(w) for w in diff])]
        diff_str = " ".join(text[diff[0]: diff[-1] + 1])
        # Remove stuff like "наличие наличие наличие"
        diff_str = re.sub(r"(.*? )+", r"\1", diff_str).strip()
        diff_str = re.sub(r"(.*? )+", r"\1", diff_str).strip()
        if len(diff) > len(column_piped) / 2 and\
                len(diff_str.split()) > len(column_piped) / 2:
            # Option names are values on their own
            # (ABS, Стеклоомывалка) vs Двиг.1.6л
            diff_tags = self.predictor.predict(diff_str.split())
            # to prevent cases like 'стоимости в рамках': 'в стоимость',
            # to prevent cases like 'стоимости и': 'и и',
            non_auxilary = True
            if diff_tags[0].pos in ["ADP", "CONJ"] or\
                    diff_tags[-1].pos in ["ADP", "CONJ"]:
                non_auxilary = False
            if non_auxilary:
                if column in self.option_names:
                    text_characteristics[column] = diff_str
                    # text_characteristics[column] = column
                    # text_characteristics[column] = True
                else:
                    # diff[-1] is the last word of our diff-phrase
                    if len(text) > diff[-1] + 1:
                        # # Words from diff are in found values for the column
                        # if any(text[diff[-1] + 1] in w
                        #         for w in self.value_range[column_id]):
                            child_phrase, diff,\
                                additional_phrases = self.get_child_phrase(
                                    diff, text)
                            text_characteristics[diff_str] = " ".join(
                                [text[i] for i in child_phrase])
                            for w_i in additional_phrases:
                                w = text[w_i]
                                text_characteristics[w] = " ".join(
                                    [text[i] for i in child_phrase])
        return text_characteristics

    def filter_dict(self, parsed_dict):
        result = [k for k in parsed_dict.keys() if
                  any(str(k).lower() == o for
                      o in self.columns) or
                  any(o == str(k).lower() for
                      o in self.columns)]
        return result

    def parse_text(self, text: list, to_filter=False, piped=True) -> dict:
        """
        takes a list of strings

        outputs a dict where the key is the name of a property and value
        is its value O_-
        """
        if not piped:
            text = text_pipeline(text[0], lemmatize=False)
        text_normed = [w.normal_form for w in self.predictor.predict(text)]
        car_characts = dict()
        for c_i, c in enumerate(self.columns_piped):
            if not c:
                continue
            car_characts.update(self.find_characteristics(
                text, c_i, text_normed))
        # Remove keys with the same values
        car_characts_inverse = {v: k for k, v in car_characts.items()}
        car_characts = {v: k for k, v in car_characts_inverse.items()}
        for k, v in car_characts.items():
            if k in self.number_columns and not re.match("\d+", v):
                car_characts[k] = None
        car_characts = {k: v for k, v in car_characts.items() if v}
        if to_filter:
            if not self.filter_dict(car_characts):
                car_characts = dict()
        return car_characts

    def load_stoplines(self, filename):
        stopwords = set()
        try:
            with open(filename) as f:
                stopwords = f.readlines()
                stopwords = set([l.strip() for l in stopwords])
        except FileNotFoundError:
            pass
        return stopwords
