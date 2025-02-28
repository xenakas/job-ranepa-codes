import re
import pickle

import pandas as pd
import numpy as np
from torch.utils import data

from bpemb import BPEmb

from utils_d.ml_utils import pad_texts, pad_text
from sklearn.model_selection import train_test_split

y_dim = 1000


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, X, Y, y_dim):
        """
        Initialization
        y_dim is the bpe vector size
        """
        super(Dataset, self).__init__()
        self.y_len = Y.shape[1]
        self.y_dim = y_dim
        self.X = X
        self.Y = Y

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

    def __getitem__(self, index):
        'Generates one sample of data'
        # get label
        x = self.X[index]
        # one-hot to values
        y = np.zeros(shape=(self.y_len, self.y_dim))
        y_val = self.Y[index]
        y[np.arange(self.y_len), y_val] = 1
        y = y.flatten()
        # y = self.Y[index]
        return x, y


def prepare_auto_ru_dataset(seq2seq=True, use_pandas=False):
    # open pickled car dict
    if use_pandas:
        df = pd.read_csv("auto-ru/auto_ru_pandas.csv")
    else:
        ads = pickle.load(open("auto-ru/auto_ru_ads.pcl", "rb"))

        # it's originally a dict where links are keys
        # keys = ads.keys()
        ads = list(ads.values())

        # it flattens the dict (almost)
        df = pd.io.json.json_normalize(ads, sep='_')

    # some columns are useless
    bad_cols = [
        "_link", "_vin", "phone_number", "characteristics_двигатель", "keys"]
    # too lazy to deal with discounts now; contains list of lists
    # [['В кредит', '–\u200920 000 ₽'], ['За трейд-ин', '–\u200930 000 ₽']]
    bad_cols += ["discounts", "metro"]
    for col in df.columns:
        if any(b in col for b in bad_cols):
            df.drop(columns=[col], inplace=True, axis=1)
    # remove rows without comments (it will be our X)
    df = df[df["comment"].notna()]

    df["марка"] = df["title"].str.split().str[0]
    # remove columns which contain arrays
    for col in df.columns:
        try:
            print(col, df[col].unique().shape)
        except Exception as ex:
            if not seq2seq:
                col_set = set()
                for v in df[col].values:
                    if type(v) == list:
                        col_set.update(set(v))
                for col_s in col_set:
                    df[col_s] = df[col].apply(
                        lambda x: type(x) == list and col_s in x)
                    print(col_s, "created")
            print(col, "dropped")
            df.drop(columns=[col], inplace=True, axis=1)

    # sort columns by the number of unique elements
    col_order = np.array([df[col].unique().shape[0] for col in df.columns])
    col_index = np.argsort(col_order)[::-1]
    col_order = col_order[col_index]
    df = df[[df.columns[c] for c in col_index]]
    if not seq2seq:
        return df
    if seq2seq:
        df = df[[df.columns[c_i] for c_i, c in enumerate(col_order) if c > 50]]
        col_order = [c for c in col_order if c > 50]
        col_order = np.argsort(col_order)
        df = df[[df.columns[c] for c in col_order]]

    # some column renaming
    num_cols = ["_пробег", "_литраж", "price", "_discount", "_мощность"]

    for col in df.columns:
        if any(n_c in col for n_c in num_cols):
            df[col] = df[col].apply(
                lambda x: "".join(re.findall("\d+", x))
                if not pd.isna(x) else np.nan)
            df[col] = df[col].replace("", np.nan)
            # df[df[col] < df[col].quantile(0.005)] = np.nan

    # classification_columns = []
    # seq2seq_columns = []

    # all columns to bpe
    bpemb_ru = BPEmb(lang="ru", dim=50, vs=y_dim)

    for col in df.columns:
        df[col] = df[col].astype(str).apply(
            lambda x: bpemb_ru.encode_ids(x) if x != "nan" else np.nan)

    lengths = [df[col].astype(str).apply(lambda x: bpemb_ru.encode(x)).
               str.len().max() for col in df.columns if col != "comment"]
    print(lengths)
    # create embedding matrix for the neural network (nn)
    emb_matrx = np.array([bpemb_ru[w] for w in bpemb_ru.words])

    seq_lengths = df["comment"].str.len()
    print(seq_lengths.describe())

    df["comment"] = df["comment"].str[:1000]

    df = df[df["comment"].str.len() > 0]
    df["comment"] = pad_texts(df["comment"].values, max_len=1000,
                              padding_text=0)

    renamings = {"seller_name": "имя продавца",
                 "price": "цена",
                 "title": "название",
                 "price_without_discounts": "цена без скидок",
                 "region": "регион",
                 "phone_schedule": "когда звонить",
                 "max_discount": "максимальная скидка"}
    for col in df.columns:
        if col.startswith("characteristics_"):
            renamings[col] = col.replace("characteristics_", "")

    df = df.rename(renamings, axis="columns")

    col_bpe = [bpemb_ru.encode_ids(col) for col in df.columns]
    col_bpe = [col[:10] for col in col_bpe]
    col_bpe = pad_texts(col_bpe, padding_text=0)

    # max length of the nn output
    max_output = max(
        (df[col].str.len().quantile(0.9) for col in df.columns
         if col != "comment"))
    max_output = int(5 * (max_output % 5 + 1))
    # max_output = int(max_output) + 1

    for col in df.columns:
        if col == "comment":
            continue
        # pad and cut at the same time
        df[col] = df[col].apply(
            lambda x: pad_text(x, padding_text=0,
                               max_len=max_output)[:max_output]
            if not np.isnan(x).any() else np.nan)

    X, Y = [], []
    for col_i, col in enumerate(df.columns):
        if col == "comment":
            continue
        tmp_df = df[df[col].notna()][[col, "comment"]]
        x = tmp_df["comment"] + col_bpe[col_i]
        y = tmp_df[col]
        X += list(x)
        Y += list(y)

    X = np.array(X)
    Y = np.array(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    dataset_train = Dataset(X_train, Y_train, y_dim)
    dataset_test = Dataset(X_test, Y_test, y_dim)

    return dataset_train, dataset_test, emb_matrx
