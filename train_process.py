
import pickle
from hazm import *
import pandas as pd
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

def process_document(doc):
    doc_no_punctuation = doc.translate(str.maketrans('', '', '[،,.,:,),(,*,_,0,1,2,3,4,5,6,7,8,9,-,/,ـ]'))
    normalize_doc = normalizer.normalize(doc_no_punctuation)
    tokenize_doc = tokenizer.tokenize(normalize_doc)
    lemmatize_token = [lematizer.lemmatize(w) for w in tokenize_doc]
    return [w for w in lemmatize_token if w not in stop_words]

def update_dictionary(doc, category, word_set, word_dic):
    for word in doc:
        if word in word_set:
            word_dic[word] += 1
        else:
            word_dic[word] = 1
            word_set.add(word)

normalizer = Normalizer()
lematizer = Lemmatizer()
tokenizer = WordTokenizer()

table = pd.read_csv("train_data.csv")

news = table["Text"].tolist()
tags = table["Category"].tolist()
stop_words = set(stopwords_list())

sport_dic = defaultdict(int)
sport_set = set()
politic_dic = defaultdict(int)
politic_set = set()

with ThreadPoolExecutor() as executor:
    processed_docs = list(executor.map(process_document, news))

for index, doc in enumerate(processed_docs):
    if tags[index] == "Sport":
        update_dictionary(doc, "Sport", sport_set, sport_dic)
    else:
        update_dictionary(doc, "Politics", politic_set, politic_dic)





number_sport = 0
number_politic = 0
for i in tags:
    if i == "Sport":
        number_sport+=1
    else:
        number_politic+=1

p_sport = number_sport /(len(tags))
p_politic = number_politic /(len(tags))


all_keys = sport_set | politic_set

sport_values = sport_dic.values()
politic_values = politic_dic.values()

sum_sport = 0
sum_politic = 0

for i in sport_values:
    sum_sport+=i

for i in politic_values:
    sum_politic +=i


p_sport_words = {}
p_politic_words = {}
for key in all_keys:

    if key in sport_set :
        prob = (sport_dic[key] + 1) / (sum_sport + len(all_keys))
        p_sport_words[key] = prob
    else:
        prob = (1) / (sum_sport + len(all_keys))
        p_sport_words[key] = prob


for key in all_keys:

    if key in politic_set :
        prob = (politic_dic[key] + 1) / (sum_politic + len(all_keys))
        p_politic_words[key] = prob
    else:
        prob = (1) / (sum_politic + len(all_keys))
        p_politic_words[key] = prob





with open("sport_prob.pkl", "wb") as sport_file:
    pickle.dump(dict(p_sport_words), sport_file)

with open("politic_prob.pkl", "wb") as politic_file:
    pickle.dump(dict(p_politic_words), politic_file)


with open("p_total.txt", "w") as file:

    file.write(str(p_sport))
    file.write("\n")
    file.write(str(p_politic))
    file.close()