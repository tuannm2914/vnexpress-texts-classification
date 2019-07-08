import os
import re
import collections
import string

def bow_a_topic(topic_path):
    bag_of_words = {}
    if not os.path.exists(topic_path):
        print("path of topic does not exist !")
    else :
        with open(topic_path) as fp:
            for line in fp:
                line = preprocess(line)
                get_bow_count(line.strip().split(" "),bag_of_words)
    return bag_of_words


def preprocess(line):
    input_str = line.translate(str.maketrans("","", string.punctuation))
    final = re.sub(r'\d+', '', input_str)
    return final

def get_bow_count(words,bag_of_words):
    for word in words:
        word = word.lower()
        if word in bag_of_words:
            bag_of_words[word] += 1
        else:
            bag_of_words[word] = 1

def get_entire_bow(topic_dir):
    common_bow = {}
    if not os.path.exists(topic_dir):
        print("topic directory does not exist !")
    else:
        list_topic_path = os.listdir(topic_dir)
        for topic_path in list_topic_path:
            topic_path = os.path.join(topic_dir,topic_path)
            single_bow = final_bow(topic_path)
            get_common_bow_count(common_bow,single_bow)
    return elminate(common_bow)

def get_idf_common_bow(common_bow,topic_dir):
    idf = {}
    for topic_path in os.listdir(topic_dir):
        with open(os.path.join(topic_dir,topic_path)) as fp:
            for line in fp:
                line = preprocess(line)
                for el in set(line.split(" ")):
                    el = el.lower()
                    if el in common_bow:
                        if el in idf:
                            idf[el] += 1
                        else:
                            idf[el] = 1
    return idf

def get_common_bow_count(common_bow,single_bow):
    for word in single_bow:
        if word in common_bow:
            common_bow[word] += single_bow[word]
        else:
            common_bow[word] = single_bow[word]

def get_freq_doc(topic_dir):
    number_doc = 0
    list_number_topic = []
    topic_dir_list = os.listdir(topic_dir)
    for each_topic in topic_dir_list:
        number_doc_each_topic = 0
        topic_path = os.path.join(topic_dir,each_topic)
        with open(topic_path) as fp:
            for line in fp:
                number_doc += 1
                number_doc_each_topic += 1
        list_number_topic.append(number_doc_each_topic)
    return number_doc,list_number_topic

def get_list_class_name(topic_dir):
    list_class_name = []
    topic_dir_list = os.listdir(topic_dir)
    for each_topic in topic_dir_list:

        list_class_name.append(get_class_name(each_topic))
    return list_class_name

def get_class_name(topic_name):
    return re.sub(".txt","",topic_name).strip()


def most_common(word_dict,m):
    ordered =  collections.OrderedDict(sorted(word_dict.items(),key=lambda el:-el[1]))
    return [key for key,value in ordered.items()][:m]

def most(word_dict,m):
    ordered = collections.OrderedDict(sorted(word_dict.items(),key=lambda el:-el[1]))
    return [[key,value] for key,value in ordered.items()][:m]

def less_common(word_dict):
    ordered = collections.OrderedDict(sorted(word_dict.items()),key=lambda  el:el[1])
    return [key for key,value in ordered.items() if value in range(1,10)]

def get_stopword(path_file):
    stopwords = []
    with open(path_file) as fp:
        for line in fp:
            line = re.sub("\n",'',line)
            stopwords.append("_".join(line.split(" ")))
    return stopwords

def final_bow(topic_path):
    bow = bow_a_topic(topic_path)
    stopwords = get_stopword("/home/tuannm/mine/vnexpress-texts-classification/data/vietnamese-stopwords.txt")
    eliminate_words = []
    eliminate_words.append(most_common(bow,10))
    eliminate_words.append(stopwords)
    #eliminate_words.append(less_common(bow))
    for el in eliminate_words:
        for word in el:
            if word in bow:
                del bow[word]
    return bow

def elminate(bow):
    for el in less_common(bow):
        del bow[el]
    return bow

def create_text_dict(data):
    words_dict = {}
    for word in data:
        if word in words_dict:
            words_dict[word] += 1
        else:
            words_dict[word] = 1
    return words_dict

def test_idf(text,path):
    count = 0
    for topic in os.listdir(path):
        with open(os.path.join(path,topic)) as fp:
            for line in fp:
                if text in set(line.split(" ")):
                    count += 1
    return count

if __name__ == "__main__":
    final = final_bow("/home/tuannm/mine/vnexpress-texts-classification/data/tokenized_data/Chinh tri Xa hoi.txt")
    print(most_common(final,30))
