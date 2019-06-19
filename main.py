import os
import ast
from tokenization.dict_models import LongMatchingTokenizer

def load_data(input_path):
    lm = LongMatchingTokenizer()
    with open(input_path,'r') as f:
        texts =  f.read()
        texts = ast.literal_eval(texts)
    return lm.tokenize(texts)

if "__name__" == "__main__" :
    load_data("data/Train_Full/Chinh tri Xa hoi/XH_NLD_ (3672).txt")

