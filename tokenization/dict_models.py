from base_tokenizer import BaseTokenizer
from utils import load_n_grams
import ast

import os
__author__ = "Ha Cao Thanh"
__copyright__ = "Copyright 2018, DeepAI-Solutions"


class LongMatchingTokenizer(BaseTokenizer):
    def __init__(self, bi_grams_path='/home/tuannm/mine/vnexpress-texts-classification/tokenization/bi_grams.txt', tri_grams_path='/home/tuannm/mine/vnexpress-texts-classification/tokenization/tri_grams.txt'):
        """
        Initial config
        :param bi_grams_path: path to bi-grams set
        :param tri_grams_path: path to tri-grams set
        """
        self.bi_grams = load_n_grams(bi_grams_path)
        self.tri_grams = load_n_grams(tri_grams_path)

    def tokenize(self, text):
        """
        Tokenize text using long-matching algorithm
        :param text: input text
        :return: tokens
        """
        syllables = LongMatchingTokenizer.syllablize(text)
        syl_len = len(syllables)
        curr_id = 0
        word_list = []
        done = False
        while (curr_id < syl_len) and (not done):
            curr_word = syllables[curr_id]
            if curr_id >= (syl_len - 1):
                word_list.append(curr_word)
                done = True
            else:
                next_word = syllables[curr_id + 1]
                pair_word = ' '.join([curr_word.lower(), next_word.lower()])
                if curr_id >= (syl_len - 2):
                    if pair_word in self.bi_grams:
                        word_list.append('_'.join([curr_word, next_word]))
                        curr_id += 2
                    else:
                        word_list.append(curr_word)
                        curr_id += 1
                else:
                    next_next_word = syllables[curr_id + 2]
                    triple_word = ' '.join([pair_word, next_next_word.lower()])
                    if triple_word in self.tri_grams:
                        word_list.append('_'.join([curr_word, next_word, next_next_word]))
                        curr_id += 3
                    elif pair_word in self.bi_grams:
                        word_list.append('_'.join([curr_word, next_word]))
                        curr_id += 2
                    else:
                        word_list.append(curr_word)
                        curr_id += 1
        return word_list


"""Tests"""


def test():
    lm_tokenizer = LongMatchingTokenizer()
    tokens = lm_tokenizer.tokenize("Thuế thu nhập cá nhân")
    print(tokens)

def load_data(input_path):
    lm = LongMatchingTokenizer()
    with open(input_path,'rb') as f:
        texts =  f.read()
    txt = texts.decode("utf-16")
    return lm.tokenize(txt)

def tokenize_data(input_dir,output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    list_topic_dirs = os.listdir(input_dir)
    for topic_dir in list_topic_dirs:
        save_topic_path = os.path.join(output_dir,topic_dir)
        read_topic_path = os.path.join(input_dir,topic_dir)
        if not os.path.exists(save_topic_path+".txt"):
            with open(save_topic_path + ".txt",'w+') as f:
                for single_file in os.listdir(read_topic_path):
                    single_file_path = os.path.join(read_topic_path,single_file)
                    f.write(" ".join(load_data(single_file_path)))
                    f.write("\n")



if __name__ == '__main__':
    tokenize_data("/home/tuannm/mine/vnexpress-texts-classification/data/Train_Full","/home/tuannm/mine/vnexpress-texts-classification/data/tokenized_data")

