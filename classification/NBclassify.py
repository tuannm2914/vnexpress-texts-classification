import numpy as np
from tokenization import dict_models
from create_dicts import *
import os


number_document = 33759

class tf_idf:
    def __init__(self,texts,single_bow,common_bow):
        self.texts = texts
        self.single_bow = single_bow
        self.common_bow = common_bow

    def get_tf(self):
        total_words = sum([value for key,value in self.single_bow.items()])
        list_tf = []
        for text in self.texts:
            if text not in self.single_bow:
                self.single_bow[text] = 1
            else:
                self.single_bow[text] += 1
            list_tf.append([self.single_bow[text] / (total_words + len(self.texts)) ])
        return list_tf

    def get_idf(self):
        list_idf = []
        for text in self.texts:
            if text not in self.common_bow:
                self.common_bow[text] = 1
            list_idf.append((1 + number_document) / self.common_bow[text])
        return np.log(list_idf)

    def get_tf_idf(self):
        return [float(np.multiply(el[0], el[1])) for el in zip(self.get_tf(),self.get_idf())]

class NB_classify:
    def __init__(self,list_tf_idf,freq_of_class):
        self.tf_idf_each_class = list_tf_idf
        self.freq_each_class = freq_of_class

    def NB_denominator(self):
        deno = 0
        for i in range(len(self.freq_of_class)):
            deno += np.prod(self.tf_idf_each_class[i]) * self.freq_each_class[i]
        return deno

    def get_list_NB_prob(self):
        list_NB_prob  = []
        deno = self.NB_denominator()
        for i in range(len(self.freq_each_class)):
            list_NB_prob.append((np.prod(self.tf_idf_each_class[i]) * self.freq_each_class[i]) / deno)
        return [el / sum(list_NB_prob) for el in list_NB_prob]

    def predict(self):
        list_NB_prob = get_list_NB_prob()
        return np.argmax(list_NB_prob)


def load_data(input_path):
    lm = dict_models.LongMatchingTokenizer()
    with open(input_path,'rb') as f:
        texts =  f.read()
    txt = texts.decode("utf-16")
    return lm.tokenize(txt)

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





if __name__ == "__main__":

    test("/home/tuannm/mine/vnexpress-texts-classification/data/Test_Full")