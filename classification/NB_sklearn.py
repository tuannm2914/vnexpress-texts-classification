from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
import os,re
import string
from sklearn.metrics import accuracy_score
from create_dicts import *

def get_stopword(path_file):
    stopwords = []
    with open(path_file) as fp:
        for line in fp:
            line = re.sub("\n",'',line)
            stopwords.append("_".join(line.split(" ")))
    return stopwords

def preprocess(line,stopwords):
    #line = " ".join([el for el in line.split(" ") if el not in stopwords])
    input_str = line.translate(str.maketrans("","", string.punctuation))
    final = re.sub(r'\d+', '', input_str)
    final = list(map(lambda e:e.lower(),final.split(" ")))
    return " ".join(final)

def prepare_data(train_file,test_file,max_nb_words):
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    i = 0
    stopword = get_stopword("/home/tuannm/mine/vnexpress-texts-classification/data/vietnamese-stopwords.txt")
    for class_file in os.listdir(train_file):
        with open(os.path.join(train_file,class_file)) as fp:
            for line in fp:
                Y_train.append(i)
                line = preprocess(line,stopword)
                X_train.append(line)
            i += 1
    X_train,_,Y_train,_ = train_test_split(X_train,Y_train,test_size=0,random_state=234)
    vectorizer = TfidfVectorizer(analyzer='word',max_features=max_nb_words)
    vectorizer.fit(X_train)
    X_train_tfidf = vectorizer.transform(X_train)
    j = 0
    for class_file in os.listdir(test_file):
        with open(os.path.join(test_file,class_file)) as fp:
            for line in fp:
                Y_test.append(j)
                line = preprocess(line,stopword)
                X_test.append(line)
            j+= 1
    X_test_tfidf = vectorizer.transform(X_test)
   # Y_train = pd.get_dummies(Y_train).values
   # Y_test = pd.get_dummies(Y_test).values

    return X_train_tfidf,Y_train,X_test_tfidf,Y_test

if __name__ == "__main__":
    max_nb_words = 20000
    X_train_tfidf, Y_train, X_test_tfidf, Y_test = prepare_data("/home/tuannm/mine/vnexpress-texts-classification/data/tokenized_data","/home/tuannm/mine/vnexpress-texts-classification/data/tokenized_test_data",max_nb_words)
    total_doc, list_number_topic = get_freq_doc("/home/tuannm/mine/vnexpress-texts-classification/data/tokenized_data")
    class_freq = [el / total_doc for el in list_number_topic]
    cls = MultinomialNB(fit_prior=True)
   # cls = GaussianNB()
    cls.fit(X_train_tfidf,Y_train)
    y_pred = cls.predict(X_test_tfidf)
    print(accuracy_score(Y_test,y_pred))

    # 88.05% Multinomial Naive Bayes
    #