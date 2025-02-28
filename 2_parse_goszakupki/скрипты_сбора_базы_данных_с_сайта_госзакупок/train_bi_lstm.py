import os
from random import shuffle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
from deepmit_ner.ner.ner.corpus import Corpus
from deepmit_ner.ner.ner.network import NER
from utils.utils import convert_dataset
# import win_unicode_console  # windows stuff
# win_unicode_console.enable()  # windows stuff

BASH = False
LOWER = False
LEMMA = False
CARS = True
CARS_ALEXANDER = False
PREFIX = "tomaev_annotation_4"

PERSONS = False
EMBEDDINGS_FILE = 3
DROPOUT = 0.2
UDPIPE = False
logging_string = ''
if BASH:
    f = open('launch_params')
    l = f.read()
    logging_string = l
    print(logging_string)
    l = l.strip().split()
    l = [int(i) for i in l]
    LOWER = bool(l[0])
    LEMMA = bool(l[1])
    PERSONS = bool(l[2])
    EMBEDDINGS_FILE = l[3]
    if len(l) > 4:
        DROPOUT = l[4]

if not logging_string:
    if not CARS:
        logging_string = '{} {} {} {}_dropout{}'.format(
            int(LOWER), int(LEMMA), int(PERSONS), EMBEDDINGS_FILE, DROPOUT)
    else:
        if PREFIX:
            logging_string = PREFIX
        else:
            logging_string = "goszakupki_ner2"

MODEL_PATH = 'model_{}/'.format(logging_string)

emb_files = {0: None,
             # Pymorphy+lemma
             1: 'embeddings/ft_native_300_ru_wiki_lenta_lemmatize.vec',
             2: 'embeddings/arraneum.txt',  # web arraneum
             3: 'embeddings/ft_native_300_ru_wiki_lenta_lower_case.vec',
             }

print(MODEL_PATH)


def open_corpora(name):
    types_string = ''
    if 'factrueval' in name:
        types_string = '_' + '_'.join([t.lower() for t in interested_types])
    name += types_string
    if LOWER:
        name += '_lower'
    if LEMMA:
        name += '_lem'
    if UDPIPE:
        name += '_udpipe'
    f = open(name)
    print(f.name)
    dataset = f.readlines()
    f.close()
    dataset = convert_dataset(dataset)
    dataset = [s for s in dataset if s[0]]
    return dataset


# f = open('factrueval_dev', 'r', encoding="utf_8_sig")  # windows
# interested_types = ['PER, 'LOC', 'ORG']
interested_types = ['PER']
corpora = ['factrueval_dev', 'factrueval_test']
persons_files = ['persons-1000_corpus', 'persons-1111-f_corpus']
cars = ["cars_ner_test.txt", "cars_ner_train.txt", "cars_ner_valid.txt"]

whole_set = []

if PERSONS:
    corpora += persons_files
for corpus in corpora:
    whole_set += open_corpora('corpora/' + corpus)
whole_set = [s for s in whole_set if s[0]]
shuffle(whole_set)

# valid = test[: int(len(test) / 2)]
# valid = test[int(len(test) / 2):]

train = whole_set[:int(len(whole_set) * 0.6)]
valid = whole_set[int(len(whole_set) * 0.6):int(len(whole_set) * 0.8)]
test = whole_set[int(len(whole_set) * 0.8):]

EMB_FILE = emb_files[EMBEDDINGS_FILE]

dataset_dict = {'train': train, 'test': test, 'valid': valid}

if CARS:
    if not PREFIX:
        prefix = "goszakupki_ner2"
    else:
        prefix = PREFIX
    for file in ('train', 'test', 'valid'):
        dataset_dict[file] = open_corpora(
            'corpora/' + "{}_{}.txt".format(prefix, file))

corp = Corpus(dataset_dict, embeddings_file_path=EMB_FILE)

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
net = NER(corp, **model_params)


learning_params = {'dropout_rate': DROPOUT,
                   'epochs': 5,
                   'learning_rate': 0.005,
                   'batch_size': 8,
                   'learning_rate_decay': 0.707}
results = net.fit(**learning_params)
f = open('bi-lstm_train_log', 'a')
f.write(logging_string + str(results))
f.close()
net.save(MODEL_PATH)
