from keras import models
import pandas as pd
from keras.preprocessing.text import  Tokenizer
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense
from keras.preprocessing.sequence import  pad_sequences
from keras.callbacks import EarlyStopping
import string
import re
import os
import random
from sklearn.model_selection import train_test_split

def RNN(max_nb_words,embedding_dim,max_len):
    model = Sequential()
    model.add(Embedding(max_nb_words,embedding_dim,input_length=max_len))
    model.add(LSTM(units=embedding_dim,dropout=0.2,recurrent_dropout=0.2))
    model.add(Dense(10,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

def preprocess(line):
    input_str = line.translate(str.maketrans("","", string.punctuation))
    final = re.sub(r'\d+', '', input_str)
    final = list(map(lambda e:e.lower(),final.split(" ")))
    return final

def prepare_data(train_file,test_file,max_nb_words,max_len):
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    i = 0
    for class_file in os.listdir(train_file):
        with open(os.path.join(train_file,class_file)) as fp:
            for line in fp:
                Y_train.append(i)
                line = preprocess(line)
                X_train.append(line)
            i += 1
    X_train,_,Y_train,_ = train_test_split(X_train,Y_train,test_size=0,random_state=234)
    tokenizer = Tokenizer(num_words=max_nb_words,lower=True)
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_train = pad_sequences(X_train,maxlen=max_len)
    j = 0
    for class_file in os.listdir(test_file):
        with open(os.path.join(test_file,class_file)) as fp:
            for line in fp:
                Y_test.append(j)
                line = preprocess(line)
                X_test.append(line)
            j+= 1
    X_test = tokenizer.texts_to_sequences(X_test)
    X_test = pad_sequences(X_test,maxlen=max_len)
    Y_train = pd.get_dummies(Y_train).values
    Y_test = pd.get_dummies(Y_test).values
    print(X_train.shape)
    return X_train,Y_train,X_test,Y_test



if __name__ == "__main__":
    max_nb_words = 20000
    max_len = 300
    embedding_dim = 128
    epochs = 50
    batch_size = 64
    X_train,Y_train,X_test,Y_test = prepare_data("/home/tuannm/mine/vnexpress-texts-classification/data/tokenized_data","/home/tuannm/mine/vnexpress-texts-classification/data/tokenized_test_data",max_nb_words,max_len)
    model = RNN(max_nb_words,embedding_dim,max_len)
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
              callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
    acc = model.evaluate(X_test,Y_test)
    print(acc)

    # 86 %