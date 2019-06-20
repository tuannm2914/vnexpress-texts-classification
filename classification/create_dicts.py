import os
import re

def bow_a_topic(topic_path):
    bag_of_words = {}
    if not os.path.exists(topic_path):
        print("path of topic does not exist !")
    else :
        with open(topic_path) as fp:
            for line in fp:
                get_bow_count(line.strip().split(" "),bag_of_words)
    return bag_of_words


def get_bow_count(words,bag_of_words):
    for word in words:
        if word in bag_of_words:
            bag_of_words[word] += 1
        else:
            bag_of_words[word] = 1

def get_idf_bow(topic_dir):
    common_bow = {}
    if not os.path.exists(topic_dir):
        print("topic directory does not exist !")
    else:
        list_topic_path = os.listdir(topic_dir)
        for topic_path in list_topic_path:
            topic_path = os.path.join(topic_dir,topic_path)
            single_bow = bow_a_topic(topic_path)
            get_common_bow_count(common_bow,single_bow)
    return common_bow

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


if __name__ == "__main__":
    print(get_freq_doc("/home/tuannm/mine/vnexpress-texts-classification/data/tokenized_data"))
