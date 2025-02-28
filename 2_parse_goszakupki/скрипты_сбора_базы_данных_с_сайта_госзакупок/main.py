from razdel import tokenize

from ner.corpus import Corpus
from ner.network import NER

from ranepa_flask_wrapper.flask_wrapper import flask_wrapper
from ranepa_string_utils.string_utils import convert_dataset

import argparse

parser = argparse.ArgumentParser(description='Create NER-service')
parser.add_argument('model_prefix', type=str, nargs='*',
                    help='a path to the model')
parser.add_argument('--port', type=int, nargs='*',
                    help='a port for Flask')

args = parser.parse_args()


model_params = {"filter_width": 7,
                "embeddings_dropout": True,
                "n_filters": [
                    128, 128,
                ],
                "token_embeddings_dim": 300,
                "char_embeddings_dim": 25,
                "use_batch_norm": True,
                "use_crf": True,
                "net_type": 'rnn',
                "use_capitalization": True,
                "cell_type": 'lstm'
                }

# Old DeepPavlov already trained models require loading the dataset
# Thanks God, NER datasets are tiny
# That's why this module is deprecated =)

dataset_dict = dict()
prefix = "goszakupki_ner2"
if args.model_prefix:
    prefix = " ".join(args.model_prefix)
for file in ("train", "test", "valid"):
    with open(f"{prefix}_{file}.txt") as f:
        dataset = f.readlines()
    corpus = convert_dataset(dataset)
    dataset_dict[file] = corpus
emb_path = 'embeddings/ft_native_300_ru_wiki_lenta_lower_case.vec'
corp = Corpus(dataset_dict, embeddings_file_path=emb_path)
print("corpus loaded")
nn = NER(
    corp, verbouse=True,  # it's really 'verbouse'
    pretrained_model_filepath=f'model_{prefix}/',
    **model_params)


def get_ner(text: str) -> dict:
    """
    In [41]: get_ner("Лада седан цвета баклажан двигатель 1.6 л")
    Out[41]:
    {'text': ['Лада', 'седан', 'цвета', 'баклажан', 'двигатель', '1.6', 'л'],
     'NER': ['O',
      'B-KUZOV',
      'B-TSVET',
      'I-TSVET',
      'O',
      'B-DVIGATEL',
      'I-DVIGATEL']}
    """
    text = list(tokenize(text))
    text = [word.text for word in text]
    output = dict()
    preds = nn.predict_for_token_batch([text])
    if preds:
        preds = preds[0]
    output["text"] = text
    output["NER"] = preds
    return output


if __name__ == "__main__":
    port = 5031
    if args.port:
        port = args.port[0]
    flask_wrapper(get_ner, port=port)
