import pandas as pd

from ufal import udpipe
from ufal.udpipe import Model, Pipeline, ProcessingError, Sentence
from utils_d.utils import text_pipeline
from gensim.models import FastText
from rnnmorph.predictor import RNNMorphPredictor

predictor = RNNMorphPredictor(language="ru")

# df = pd.read_csv("ItemInfo_train.csv", nrows=1000)
# df = df[df.categoryID == 9]
# df.to_csv("cars_only.csv")
df = pd.read_csv("cars_only.csv")  # , nrows=1000)
df["description"] = df.description.apply(lambda x: " ".join(x.split()))

text = df["description"].values[0]
model = udpipe.Model.load(
    "/home/denis/ranepa/embeddings/russian-syntagrus-ud-2.0-170801.udpipe")
pipe = Pipeline(
    model, "generic_tokenizer", model.DEFAULT, model.DEFAULT, model.DEFAULT)
pipe.process(text)

texts = df.description.values

# descriptions_0 - no-preprocessing
# descriptions_1 - text_pipeline(t, lemmatize=False)
# descriptions_2 - predictor.predict(t)
texts = [text_pipeline(t, lemmatize=True) for t in texts]
texts = [" ".join(t) for t in texts]
texts = "\n".join(texts)
f = open("descriptions_1.txt", "w")
f.write(texts)

w2v_model = FastText(corpus_file="descriptions_1.txt", workers=3)
w2v_model.save("w2v/w2v_1")

with open("descriptions_1.txt") as f:
    texts = f.readlines()

texts = [t.strip().split() for t in texts]

last_string = 0
with open("descriptions_2.txt") as f:
    for line in f:
        last_string += 1

texts = texts[last_string:]
f = open("descriptions_2.txt", "a")

batch_step = 100
# [batch_step: len(texts) + batch_step]
batch_range = range(batch_step, len(texts) + batch_step, batch_step)

for i in batch_range:
    print(" " * 100, end="\r")
    print("\t {:06d} {:06d} {:06d}".format(
        last_string + i - batch_step,
        last_string + i,
        len(texts) - i), end="\r")
    batch = texts[i - batch_step: i]
    batch = predictor.predict_sentences(batch)
    batch = [[w.normal_form for w in text] for text in batch]
    # Join add "\n" only between elements
    batch = "\n".join([" ".join(text) for text in batch]) + "\n"
    f.write(batch)
# for t_i, t in enumerate(texts):
#     print("\t", t_i, end="\r")
#     prediction = predictor.predict_sentences(t)
#     prediction = [p.normal_form for p in prediction]
#     prediction = " ".join(prediction) + "\n"
#     f.write(prediction)
