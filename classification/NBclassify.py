import numpy as np
from tokenization import dict_models
from create_dicts import *
import os
<<<<<<< HEAD
from collections import Counter
import re
import string
=======
>>>>>>> origin


number_document = 33759

class tf_idf:
<<<<<<< HEAD
    def __init__(self,texts,single_bow,common_bow,idf):
        self.texts = texts
        self.single_bow = single_bow
        self.common_bow = common_bow
        self.idf = idf
        self.number_document = number_document
=======
    def __init__(self,texts,single_bow,common_bow):
        self.texts = texts
        self.single_bow = single_bow
        self.common_bow = common_bow
>>>>>>> origin

    def get_tf(self):
        total_words = sum([value for key,value in self.single_bow.items()])
        list_tf = []
        for text in self.texts:
<<<<<<< HEAD
            freq = 1 / total_words
            if text in self.single_bow:
                freq = (1+ self.single_bow[text])/ (total_words + len(self.texts))
            list_tf.append(freq)
=======
            if text not in self.single_bow:
                self.single_bow[text] = 1
            else:
                self.single_bow[text] += 1
            list_tf.append([self.single_bow[text] / (total_words + len(self.texts)) ])
>>>>>>> origin
        return list_tf

    def get_idf(self):
        list_idf = []
        for text in self.texts:
<<<<<<< HEAD
            freq = 1
            if text in self.idf:
                freq = 1 + self.idf[text]
            list_idf.append((number_document + 1) / (freq))
=======
            if text not in self.common_bow:
                self.common_bow[text] = 1
            list_idf.append((1 + number_document) / self.common_bow[text])
>>>>>>> origin
        return np.log(list_idf)

    def get_tf_idf(self):
        return [float(np.multiply(el[0], el[1])) for el in zip(self.get_tf(),self.get_idf())]

class NB_classify:
    def __init__(self,list_tf_idf,freq_of_class):
        self.tf_idf_each_class = list_tf_idf
        self.freq_each_class = freq_of_class

    def NB_denominator(self):
        deno = 0
<<<<<<< HEAD
        for i in range(len(self.freq_each_class)):
            deno += np.sum(np.log(self.tf_idf_each_class[i])) + np.log (self.freq_each_class[i])
=======
        for i in range(len(self.freq_of_class)):
            deno += np.prod(self.tf_idf_each_class[i]) * self.freq_each_class[i]
>>>>>>> origin
        return deno

    def get_list_NB_prob(self):
        list_NB_prob  = []
        deno = self.NB_denominator()
<<<<<<< HEAD
        #print(deno)
        for i in range(len(self.freq_each_class)):
            list_NB_prob.append(np.divide(np.sum(np.log(self.tf_idf_each_class[i])) +np.log(self.freq_each_class[i]),deno))
        return list_NB_prob

    def predict(self):
        list_NB_prob = self.get_list_NB_prob()
        return np.argmin(list_NB_prob)
=======
        for i in range(len(self.freq_each_class)):
            list_NB_prob.append((np.prod(self.tf_idf_each_class[i]) * self.freq_each_class[i]) / deno)
        return [el / sum(list_NB_prob) for el in list_NB_prob]

    def predict(self):
        list_NB_prob = get_list_NB_prob()
        return np.argmax(list_NB_prob)
>>>>>>> origin


def load_data(input_path):
    lm = dict_models.LongMatchingTokenizer()
    with open(input_path,'rb') as f:
        texts =  f.read()
    txt = texts.decode("utf-16")
    return lm.tokenize(txt)

<<<<<<< HEAD
def preprocess(line):
    input_str = line.translate(str.maketrans("","", string.punctuation))
    final = re.sub(r'\d+', '', input_str)
    return final

def vectorize_data(common_bow,line):
    line = preprocess(line)
    data = list(map(lambda  el: el.lower(),line.strip().split(" ")))
    return [text for text in data if text in common_bow]


def test(test_dir):
    topic_dir = "/home/tuannm/mine/vnexpress-texts-classification/data/tokenized_data"

    total_doc,list_number_topic = get_freq_doc(topic_dir)
    class_freq = [el / total_doc for el in list_number_topic]
    common_bow = get_entire_bow(topic_dir)
    idf = get_idf_common_bow(common_bow,"/home/tuannm/mine/vnexpress-texts-classification/data/tokenized_data")
    list_class_name = get_list_class_name(topic_dir)
    print(len(idf),len(common_bow))
    list_single_bow = []
    for topic in os.listdir(topic_dir):
        list_single_bow.append(final_bow(os.path.join(topic_dir,topic)))
    cnt = Counter()
    cnt2 = Counter()
    all_class_test_path = os.listdir(test_dir)
    i = 0
    j= 0
    rs = []
    for el in list_class_name:
        counter = Counter()
        rs.append([el,counter])
    for each_class_path in all_class_test_path:
        class_file_path = os.path.join(test_dir,each_class_path)
        with open(class_file_path) as fp:
            for line in fp:
                data = vectorize_data(common_bow,line)
                list_tf_idf = []
                for single_bow in list_single_bow:
                    tf_idf_obj = tf_idf(data,single_bow,common_bow,idf)
                    tf_idf_value = tf_idf_obj.get_tf_idf()
                    list_tf_idf.append(tf_idf_value)
                cls = NB_classify(list_tf_idf,class_freq)

                cnt2[cls.predict()] += 1
                if cls.predict()== i:
                    cnt["True"]+= 1
                    rs[i][1]["True"] += 1
                else:
                    cnt['Wrong'] += 1
                    rs[i][1]["False"] += 1

                j+= 1
                if j%1000==0:
                    print(cnt,end="\t")
                    print("acc : ", (cnt["True"] / (cnt["True"] + cnt["Wrong"])))
                    print(rs)
        i+= 1
    return cnt


if __name__ == "__main__":
    test("/home/tuannm/mine/vnexpress-texts-classification/data/tokenized_test_data")

# 89.6 %

=======
def test(test_dir):
    topic_dir = "/home/tuannm/mine/vnexpress-texts-classification/data/tokenized_data"
    total_doc,list_number_topic = get_freq_doc(topic_dir)
    common_bow = get_idf_bow(topic_dir)
    list_class_name = get_list_class_name(topic_dir)
    list_single_bow = []
    for topic in os.listdir(topic_dir):
        list_single_bow.append(bow_a_topic(os.path.join(topic_dir,topic)))

    all_class_test_path = os.listdir(test_dir)
    for each_class_path in all_class_test_path:
        class_path = os.path.join(test_dir,each_class_path)
        for file_path in os.listdir(class_path):
            file_path = os.path.join(class_path,file_path)
            texts = load_data(file_path)
            list_tf_idf = []
            for single_bow in list_single_bow:
                print(type(single_bow))
                tf_idf_obj = tf_idf(texts,single_bow,common_bow)
                tf_idf_value = tf_idf_obj.get_tf_idf()
                list_tf_idf.append(tf_idf_value)
            cls = NB_classify(list_tf_idf,list_number_topic)

            list_nb_prob = cls.get_list_NB_prob()
            print(list_nb_prob)
>>>>>>> origin





<<<<<<< HEAD

=======
if __name__ == "__main__":

    test("/home/tuannm/mine/vnexpress-texts-classification/data/Test_Full")
>>>>>>> origin
