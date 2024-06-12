import csv

import random

import pandas as pd



table = pd.read_csv("G:\\Ehsan\\Ehsan\\sources\\term_4\\ai\\project\\nlp_train.csv")

fields = ['Text', 'Category']

with open("train_data.csv", 'w') as csvfile2:

    csvwriter = csv.writer(csvfile2)
    csvwriter.writerow(fields)


with open("test_data.csv", 'w') as csvfile2:

    csvwriter = csv.writer(csvfile2)
    csvwriter.writerow(fields)


n = len(table)
for i in range(n):
    p = random.uniform(0, 1)
    if p < 0.2:
        filename = "test_data.csv"
        tag = table["Category"][i]
        row = [table["Text"][i], tag]
        with open(filename, 'a', encoding="utf-8") as csvfile2:

            csvwriter = csv.writer(csvfile2)
            csvwriter.writerow(row)
    else:
        filename = "train_data.csv"
        tag = table["Category"][i]
        row = [table["Text"][i], tag]
        with open(filename, 'a', encoding="utf-8") as csvfile2:

            csvwriter = csv.writer(csvfile2)
            csvwriter.writerow(row)