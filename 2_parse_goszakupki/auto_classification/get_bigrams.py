import re
import pickle
import pandas as pd
from collections import Counter
from gensim.models.phrases import Phrases, Phraser
from nltk.corpus import stopwords
from rnnmorph.predictor import RNNMorphPredictor
from utils_d.utils import parse_rnn_morph_tags, text_pipeline

from characteristics_extractor import CharactericsExtractor

predictor = RNNMorphPredictor(language="ru")


normed = False
parse_cars = True


def get_bigrams(texts, filename):
    texts = [t.strip().split() for t in texts]

    phrases = Phrases(texts, min_count=10, threshold=1,
                      common_terms=stopwords.words("russian"))
    bigram = Phraser(phrases)

    counter = Counter()

    for t in texts:
        for w in t:
            counter[w] += 1

    counter = Counter({k: v for k, v in counter.items() if v > 1})

    for k in counter:
        pos = predictor.predict([k])[0].pos
        if pos != "NOUN":
            counter[k] = 0

    counter = Counter({k: v for k, v in counter.items() if v > 1})

    found_bigrams = Counter()

    for t_i, t in enumerate(texts):
        print("\t", t_i, end="\r")
        t = bigram[t]
        t = [w for w in t if "_" in w]
        for w in t:
            found_bigrams[w] += 1
    for k in found_bigrams:
        k_split = k.split("_")
        if not any(w in counter for w in k_split):
            found_bigrams[k] = 0
    found_bigrams = Counter({k: v for k, v in found_bigrams.items() if v > 1})
    bigram.save(filename + "_gensim_bigram")
    pickle.dump(found_bigrams,
                open(filename + ".pcl", "wb"))


def filter_bigrams(filename, limit=1000):
    # bigram = Phraser.load(filename)
    found_bigrams = pickle.load(open(filename + ".pcl", "rb"))
    found_bigrams = Counter({k: v for k, v in found_bigrams.items() if v > 10})
    # transition_dict = dict()
    normed_bigrams = Counter()
    for key, value in found_bigrams.items():
        # if "машин" in key or "автомобиль_" in k:
        #     continue
        key_split = key.split("_")
        if "tr" in key_split:
            continue
        key_lemma = predictor.predict(key_split)
        nominative_case = False
        tags = [parse_rnn_morph_tags(k_l.tag) for k_l in key_lemma]
        for tag in tags:
            if "case" in tag:
                if tag["case"] == "nom":
                    nominative_case = True
        if not nominative_case:
            pass
            # key_lemma = [w.normal_form for w in key_lemma]
            # normed_bigrams["_".join(key_lemma)] += found_bigrams[key]
        elif not any(k.pos == "NOUN" for k in key_lemma):
            pass
        elif len(key_lemma) < 3 and\
                any(t for t in key_lemma if t.pos == "ADP"):
            pass
        elif key_lemma[-1].pos == "ADJ" and\
                ("case" in tags[-1] and tags[-1]["case"] != "nom"):
            pass
        # elif len([k_l for k_l in key_lemma
        #           if k_l.pos == "NOUN"]) >= 2 and\
        #         len([t for t in tags if "case"
        #              in t and t["case"] == "nom"]) >= 2:
            pass
        else:
            normed_bigrams[key] += value
    for k, v in normed_bigrams.items():
        if re.findall("\d+_\w+", k) or re.findall("\w+_\d+", k):
            normed_bigrams[k] = 0
    normed_bigrams = Counter({k: v for k, v in normed_bigrams.items()
                              if v > 1})
    variants = dict()
    for key in found_bigrams:
        key = key.split("_")
        for word_i, word in enumerate(key):
            if word not in variants:
                variants[word] = Counter()
            for extra_key in [" ".join(key[:word_i]) + "_",
                              "_" + " ".join(key[word_i + 1:])]:
                if extra_key != "_":
                    variants[word][extra_key] += 1
    pickle.dump(normed_bigrams,
                open(filename + "_normed_bigrams.pcl", "wb"))


def limit_bigrams(bigram, found_bigrams):
    found_bigrams = Counter(
        {k: v for k, v in found_bigrams.items() if v > 1000})
    bigrams_split = [w.split("_") for w in found_bigrams.keys()]
    bigrams_set = [set(b) for b in bigrams_split]

    # bigrams_intersect = dict()
    for b_i, b in bigrams_set:
        print("\t", b_i, end="\r")
        intersection = [b.intersection(b_2) for b_2 in bigrams_set]
        intersection[b_i] = set()
    # texts = [bigram[t] for t in texts]
    # phrases = Phrases(texts, min_count=50, threshold=10,
    #                   common_terms=stopwords.words("russian"))

    # trigram = Phraser(phrases)

    most_common_bigrams = [w[0] for w in found_bigrams.most_common(
        int(len(found_bigrams) / 100))]

    for b_i, b in enumerate(most_common_bigrams):
        if b in most_common_bigrams[:b_i] + most_common_bigrams[b_i + 1:]:
            print(b_i, b)


def parse_texts(texts, filename):
    extractor = CharactericsExtractor(
        dictfile=None, bigrams_file=filename + "_normed_bigrams.pcl")
    characteristics = [[]] * len(texts)
    for t_i, t in enumerate(texts):
        print("\t", t_i, end="\r")
        if not t:
            continue
        output = extractor.parse_text([t], piped=False)
        characteristics[t_i] = output
    pickle.dump(
        characteristics,
        open(filename + "_characteristics.pcl", "wb"))
    return characteristics


if __name__ == "main":
    if parse_cars:
        if normed:
            with open("avito/descriptions_lemmatized.txt") as f:
                texts_normed = f.readlines()
        else:
            with open("avito/descriptions_lower.txt") as f:
                texts = f.readlines()
        get_bigrams(texts, "")
        filter_bigrams()
        limit_bigrams()
    else:
        df = pd.read_csv("cian_flat_ds_20000.csv", delimiter=";")
        texts = df[df.columns[1]]
        texts = [" ".join(text_pipeline(t, lemmatize=False)) for t in texts]
        get_bigrams(texts, "cian")
        filter_bigrams("cian")

        bigrams = pickle.load(open("ciannormed_bigrams.pcl", "rb"))
        extractor = CharactericsExtractor(
            dictfile=None, bigrams_file="ciannormed_bigrams.pcl")
        characteristics = [{}] * len(texts)
        for t_i, t in enumerate(texts):
            print("\t", t_i, end="\r")
            if not t:
                continue
            output = extractor.parse_text([t], piped=False)
            characteristics[t_i] = output
        pickle.dump(
            characteristics,
            open("cian" + "_characteristics.pcl", "wb"))
        for c_i, c in enumerate(characteristics):
            if type(c) != dict:
                characteristics[c_i] = {}
        characteristics_to_int = [{k: 1 for k, v in char.items()}
                                  for char in characteristics]
        characteristics_pd = pd.DataFrame(characteristics_to_int)
        characteristics_pd.fillna(0, inplace=True)
        characteristics_pd.to_csv("characteristics_pd.csv")
