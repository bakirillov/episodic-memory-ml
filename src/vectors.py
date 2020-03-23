#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
from misc import *
import pandas as pd



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--vectors",
        dest="vectors",
        action="store", 
        help="Path to file with aligned vectors", 
        default="/home/bakirillov/HDD/weights/fasttext/aligned/wiki.en.align.vec"
    )
    parser.add_argument(
        "-s", "--study",
        dest="study",
        action="store", 
        help="Path to study data file", 
        default="en_study.pkl"
    )
    parser.add_argument(
        "-o", "--output",
        dest="output",
        action="store", 
        help="set the path of output file"
    )
    args = parser.parse_args()
    study = Study.load_from_file(args.study)
    word_aucs = study.compute_word_aucs()
    data = load_vectors(
        args.vectors, word_aucs.index
    )
    pd.DataFrame(data).T.join(word_aucs).to_csv(args.output)

