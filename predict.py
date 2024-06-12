import math
import pickle

import pandas as pd

from collections import Counter
from hazm import *

from train_process import stop_words

# from hazmtest import stop_words


with open("sport_prob.pkl", "rb") as file:
    sport_dic3 = pickle.load(file)

with open("politic_prob.pkl", "rb") as file:
    politic_dic3 = pickle.load(file)


table = pd.read_csv("test_data.csv")
#table = pd.read_csv("G:\\Ehsan\\Ehsan\\sources\\term_4\\ai\\NLP\\concat.csv")


with open("p_total.txt", "r") as file:

    lines = file.readlines()
    float_sport = float(lines[0].strip())
    float_politic = float(lines[1].strip())

normalizer = Normalizer()
lematizer = Lemmatizer()
tokenizer = WordTokenizer()
docs = table["Text"].to_list()
all_keys = sport_dic3.keys()
predict_tags = []
for doc in docs:
    if type (doc ) ==  float:
        continue
    doc_no_punctuation = doc.translate(str.maketrans('', '', '[،,.,:,),(,*,_,0,1,2,3,4,5,6,7,8,9,-,/,ـ]'))
    normalize_doc = normalizer.normalize(doc_no_punctuation)
    tokenize_doc = tokenizer.tokenize(normalize_doc)
    lemmatize_token = [lematizer.lemmatize(w) for w in tokenize_doc]
    final_token = [w for w in lemmatize_token if w not in  stop_words]
    frequency = Counter(final_token)
    p_doc_in_sport = math.log(float_sport)
    p_doc_in_politic = math.log(float_politic)
    for key , val in frequency.items() :

        if key in all_keys :
            p_doc_in_sport += math.log(sport_dic3[key]) * val
            p_doc_in_politic += math.log(politic_dic3[key]) * val

    if p_doc_in_sport > p_doc_in_politic:
        predict_tags.append("Sport")
    else:
        predict_tags.append("Politics")

with open("predict.pkl", "wb") as predict_file:
    pickle.dump(list(predict_tags), predict_file)

