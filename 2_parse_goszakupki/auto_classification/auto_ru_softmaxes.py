import pickle
import re
from datetime import datetime
from glob import glob

import numpy as np
from keras.models import load_model

from utils_d.ml_models import ml_pipeline
from utils_d.ml_utils import pad_text
from prepare_auto_ru_dataset_for_slot_filling import prepare_auto_ru_dataset


def train():
    now = datetime.now()
    now = "_".join([str(time) for time in (now.year, now.month, now.day)])
    df = prepare_auto_ru_dataset(seq2seq=False)
    df = df.loc[df["comment"].drop_duplicates().index]
    df["seller_name"] = df["seller_name"].fillna("").str.lower()
    companies = {
       "авто", "auto", "моторс", "motors", "мерседес", "mercedes",
       "volkswagen", "рольф",
       "cars", "авалон", "борисхоф", "империя",
       "trade in", "сервис", "рено", "renault", "inchcape", "атлантис",
       "тойота", "центр", "тц ", "лимитед", "киа ", "kia", "юг ",
       "лексус", "мкад", "car", " кар ", "-кар", "звезда"}
    filter_companies = df["seller_name"].apply(
        lambda x: not any(w in x for w in companies))
    # df[filter_companies]["seller_name"]
    # df[filter_companies].groupby(
    #     "seller_name").count().sort_values("comment").index[-100:]
    df = df[filter_companies]
    bad_cols = [
        "characteristics_госномер", "max_discount", "phone_schedule",
        "seller_name"]
    df = df[[c for c in df.columns if c not in bad_cols]]

    # rare cars; filter
    rare_values = df["марка"].value_counts() > 10
    # reduce
    rare_values = set(rare_values[rare_values].index)
    df = df[df["марка"].apply(lambda x: x in rare_values)]
    x = df["comment"]
    y = df[[c for c in df.columns if c != "comment"]]
    # print(
    #     [(c,
    #       round(y[c].dropna().unique().shape[0] /
    #             y[c].dropna().shape[0],
    #             5)
    #       )
    #      for c in y.columns])
    with open("auto_ru_columns.txt", "w") as f:
        f.write("\n".join(list(y.columns)))
    x = list(x.values)
    y = y.T.values
    y = [list(l) for l in y]
    # additional_outputs = y[1:]
    # y = y[0]
    additional_outputs = None
    y = list(df["марка"].values)

    preprocess = True
    if not preprocess:
        x = [l.lower().split() for l in x]
    print("training")
    ml_pipeline(
        x, y,
        return_data=False,
        additional_outputs=additional_outputs,
        model="cnn-lstm",
        cleaned_texts=not(preprocess),
        # fasttext=True,
        # use_pymorphy=True,
        use_elmo_untrained=True,
        layers_multiplier=1,
        name=f"auto_ru:{now}",
        use_generator=False,
        use_mp=False,
        lemmatize=False,
        sequence_len=150
    )


class InferenceModel:
    def __init__(self):
        folder = "models/cnn-lstm/auto_ru:2019_3_20"
        self.model = load_model(f"{folder}/_cnn_lstm.h5", compile=False)
        self.word_index = pickle.load(
            open(f"{folder}/_keras_word_index.pcl", "rb"))
        with open("auto_ru_columns.txt") as f:
            columns = f.readlines()
        self.columns = [c.strip() for c in columns]
        ints2y = glob(f"{folder}/_y_*_ints2y.pcl")
        self.ints2y_dict = dict()
        self.ints2y_dict[0] = pickle.load(
            open(f"{folder}/_y_ints2y.pcl", "rb"))
        for f in ints2y:
            # because due to my crappy design I have y and additional outputs
            index = re.findall("_y_\d+", f)[0][3:]
            index = int(index) + 1
            self.ints2y_dict[index] = pickle.load(open(f, "rb"))

    def inference(self, text):
        text = text.split()
        text = pad_text(text, max_len=150)[:150]
        text = [self.word_index[w] if w in self.word_index
                else self.word_index["<NULL>"]
                for w in text]
        text = np.array([text])
        preds = self.model.predict(text)
        output = []
        for o_i, o in enumerate(preds):
            max_index = np.argmax(o)
            try:
                score = o[0][max_index]
            except IndexError:
                score = o[max_index]
            value = self.ints2y_dict[o_i][max_index]
            key = self.columns[o_i]
            if value and value != 'nan':  # and score > 0.5:
                # print(key, value, score)
                output.append([key, value, score])
        return output


def test():
    inf = InferenceModel()
    inf.inference("Лада Седан цвета баклажан")
    inf.inference(
        """
        Владельцев по ПТС: 1
        Состояние: битый
        Руль: правый
        Привод: полный
        Цвет: чёрный
        Пробег: 39700 км
        Модель: Freelander
        Марка: Land Rover
        Год выпуска: 2012
        Тип кузова: внедорожник
        Тип двигателя: дизель
        Коробка передач: автомат
        Объём двигателя: 2.2
        Мощность двигателя: 150 л.с.
        Количество дверей: 5
        VIN или номер кузова: SALFA2BB*CH****63

        Адрес: Москва, МКАД, съезд 74
        Показано из

        Комплектация HSE, авто после ДТП, частично разукомплектован.
        Документы не продаю! Все вопросы по телефону.
        Находится в Зеленограде,возможна отправка транспортной компанией.
        СМСками не общаюсь, звоните.
        """)
