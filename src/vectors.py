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
        "-p", "--participant",
        dest="participant",
        action="store",
        default="all",
        help="the participant id"
    )
    parser.add_argument(
        "-o", "--output",
        dest="output",
        action="store", 
        help="set the path of output file"
    )
    parser.add_argument(
        "-w", "--what",
        dest="what",
        action="store",
        choices=["wv", "1hot"],
        default="wv",
        help="set the type of output"
    )
    args = parser.parse_args()
    study = Study.load_from_file(args.study)
    if args.participant == "all":
        word_aucs = study.compute_word_aucs()
        words = word_aucs.index
    else:
        words = study[int(args.participant)][1][2]
    if args.what == "wv":
        if "vec" in args.vectors:
            data = load_vectors(
                args.vectors, words
            )
        else:
            data = pd.read_csv(args.vectors, index_col=0).T[0:-1]
    elif args.what == "1hot":
        if "_1hot_" not in args.output:
            data = {a:Study.onehot(a) for a in words}
        else:
            data = {a:[a] for a in words}
    if args.participant == "all":
        pd.DataFrame(data).T.join(word_aucs).to_csv(args.output)
    else:
        answers = pd.DataFrame(
            {
                "answers": study[int(args.participant)][1]["answers"].values,
            }
        )
        answers.index = words.values
        pd.DataFrame(data).T.join(answers).dropna().to_csv(args.output)
    
