import pickle
import pandas as pd

def f1(predicted, original):
    n = len(predicted)
    politics_TP = 0
    politics_TN = 0
    politics_FP = 0
    politics_FN = 0

    sport_TP = 0
    sport_TN = 0
    sport_FP = 0
    sport_FN = 0

    for i in range(n):
        predicted_tag = predicted[i]
        original_tag = original[i]
        if predicted_tag == "Sport":
            if original_tag == "Sport":
                sport_TP = sport_TP + 1
                politics_TN = politics_TN + 1
            else:
                sport_FP = sport_FP + 1
                politics_FN = politics_FN + 1

        elif predicted_tag == "Politics":
            if original_tag == "Politics":
                sport_TN = sport_TN + 1
                politics_TP = politics_TP + 1
            else:
                sport_FN = sport_FN + 1
                politics_FP = politics_FP + 1

    sport_f1 = (2 * sport_TP) / ((2 * sport_TP) + sport_FP + sport_FN)
    politics_f1 = (2 * politics_TP) / ((2 * politics_TP) + politics_FP + politics_FN)

    return sport_f1, politics_f1


with open("predict.pkl", "rb") as file:
    predict = pickle.load(file)

#table = pd.read_csv("G:\\Ehsan\\Ehsan\\sources\\term_4\\ai\\NLP\\concat.csv")
table = pd.read_csv("test_data.csv")
original_tag = table["Category"].to_list()

print( f1(predict, original_tag) )

