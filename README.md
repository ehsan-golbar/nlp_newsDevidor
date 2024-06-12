# nlp_newsDevidor
This is an implementation of Naive Bayes for classifying the news into politics or sports categories. for two datasets the f1_score for both category were above %98 . 
1) Run split.py to split train and test data randomely. (output : test_data.csv)
2) Run train_process.py to calculate words and their frequency in each category.(output : sport_prob.pkl, politic_prob.pkl, p_total.txt)
3) Run predict.py to perform Naive Bayes algorithm.(output : predict.pkl)
4) Run calculate_f1.py to observe the scores for each category in console.


the f1_score for sports and politics on this dataset is below respectively :

![image](https://github.com/ehsan-golbar/nlp_newsDevidor/assets/102996244/75712870-8a51-441b-82ac-4bd3fe48a69c)
