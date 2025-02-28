import re
import os

from string import punctuation
from collections import Counter
from glob import glob
from docx import *
# from utils import clean_yargy_str
from razdel import sentenize, tokenize
from sklearn.model_selection import train_test_split
# from utils_d.web_utils import libreoffice_convert

# folder = "контракты"
folder = "tomaev_annotation"
corpus_filename = "ner_corpus/tomaev_annotation_4.txt"
docxs = glob(f"{folder}/*.docx")

if not os.path.exists(f"{folder}/transformed"):
    docs = glob(f"{folder}/*.doc")
    for doc in docxs:
        new_name = os.path.join(os.path.dirname(doc),
                                "docx_" + os.path.basename(doc))
        os.rename(doc, new_name)
    for doc in docs:
        libreoffice_convert(doc, f"./{folder}/transformed/", out_format="docx")
    docs = glob(f"{folder}/transformed/*.docx")
    for doc in docs:
        new_name = os.path.join(os.path.dirname(os.path.dirname(doc)),
                                "doc_" + os.path.basename(doc))
        os.rename(doc, new_name)

translator = str.maketrans('', '', punctuation)

# top 25 categories
categories = {
    "тип привода", "двигатель", "цвет кузова", "длина", "количество мест",
    "объем двигателя", "высота", "дорожный просвет", "максимальная скорость",
    "тип кузова", "коробка передач", "тип двигателя", "количество дверей",
    "максимальный", "ширина", "автомобиль", "комплектация", "abs",
    "год выпуска", "кондиционер", "полная масса",
    "максимальный крутящий момент",
    "количество передач", "объем багажника", "топливо", "шина",
    "легковой автомобиль", "колесная формула"
}

color_classes = {
    "yellow": "OPTION",
    "red": "YEAR",  # год выпуска: 2015
    "green": "PRICE",
    "magenta": "MODEL",  # Patriot
    "cyan": "BRAND",  # UAZ
    "darkMagenta": "TRANSMISSION",
    "darkRed": "SPECIFICATION",  # комплектация: Элеганс Плюс
    "darkGray": "WIDTH",  # ширина: 1830
    "lightGray": "WEIGHT",  # масса: 1485
    "black": "LAYOUT",  # привод: (передний, задний)
    "blue": "QUANTITY",  # количество: 1 шт.
    "darkGreen": "CONSUMPTION",  # потребление топлива: 9.5 л
    "darkBlue": "VOLUME",  # объем двигателя: 1.6
    "darkYellow": "LENGTH",  # длина
    "darkCyan": "POWER",  # мощность двигателя: 150 л.с.
    "redFont": "MAXSPEED"  # макс скорость 140 км.ч
}

corpus = []
if os.path.exists(corpus_filename):
    os.remove(corpus_filename)
corpus_file = open(corpus_filename, "a")

color_counter = Counter()
for file_i, file in enumerate(docxs):
    print("\t", file_i, end="\r")
    try:
        document = Document(file)
    except Exception as ex:
        print(ex, file_i)
        continue
    words = document.element.xpath('//w:r')

    prev_text_class = "O"
    for word_i, word in enumerate(words):
        xml = word.xml
        text = word.text
        if not text:
            continue
        text_class = "O"
        if "highlight" in xml or 'w:color w:val="FF0000"' in xml:
            color = re.findall('(<w:highlight w:val=")(.*)("/>)', xml)
            if color:
                color = color[0][1]
            elif 'w:color w:val="FF0000"' in xml:
                color = "redFont"
            # # brand only
            # if color not in {"magenta", "cyan"}:
            #     color = "yellow"
            if color in color_classes:
                text_class = color_classes[color]
                color_counter[color] += 1
            elif color != "white":
                print(
                    " ".join([w.text for w in words[word_i - 2: word_i + 2]]))
                print(text, color, text_class)
        # if not clean_text and class_corpus and color == class_corpus[-1]:
        #     print(text_corpus[-1], class_corpus[-1])
        # text_corpus.append(text)
        # class_corpus.append(text_class)
        if prev_text_class == "O" and text_class == "O":
            corpus_file.write("\n")
        prev_text_class = text_class

        sents = sentenize(text)
        text_corpus = []
        for s in sents:
            s = s.text
            words = [w.text for w in tokenize(s)]
            words = [w for w in words if w]
            for w_i, w in enumerate(words):
                if text_class != "O":
                    if w_i == 0:
                        w_class = "B-" + text_class
                    else:
                        w_class = "I-" + text_class
                else:
                    w_class = text_class
                text_corpus.append([w, w_class])
                corpus_file.write(f"{w} {w_class}\n")
            corpus.append(text_corpus)

corpus_file.close()

with open(corpus_filename) as f:
    r = f.readlines()

sents = []
sent = []
for l in r:
    if l.strip():
        sent.append(l)
    else:
        sents.append(sent)
        sent = []
sents = [s for s in sents if s]
train, test = train_test_split(sents, test_size=0.1)
train, valid = train_test_split(train, test_size=0.11)
dataset = {"train": train, "valid": valid, "test": test}

for file, data in dataset.items():
    filename = os.path.basename(corpus_filename)
    filename = filename.replace(".txt", f"_{file}.txt")
    with open(f"ner_corpus/{filename}", "w") as f:
        f.write("\n".join(["".join(sent) for sent in data]))
