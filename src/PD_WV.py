#!/usr/bin/env python
# coding: utf-8

# In[1]:


import io
import json
import joblib
import argparse
from misc import *
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from copy import deepcopy
from tpot import TPOTClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset",
        dest="dataset",
        action="store", 
        help="Path to file with the dataset"
    )
    parser.add_argument(
        "-o", "--output",
        dest="output",
        action="store", 
        help="set the mask of output file"
    )
    parser.add_argument(
        "-s", "--seed",
        dest="seed",
        action="store", 
        default=7,
        help="set the seed for PRNG"
    )
    parser.add_argument(
        "-a", "--augs",
        dest="augs",
        action="store",
        default=3,
        help="set the number of augmentations"
    )
    args = parser.parse_args()
    np.random.seed(int(args.seed))
    data = pd.read_csv(args.dataset)
    X = data.values[:,1:-1]
    Y = data.values[:,-1:].reshape(-1,)
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2)
    tr_X = deepcopy(train_X).astype(np.float32)
    tr_Y = deepcopy(train_Y).astype(np.int32)
    print(tr_X.shape, tr_Y.shape)
    for a in range(int(args.augs)):
        current_aug = train_X*np.random.normal(size=train_X.shape, loc=1, scale=0.1)
        tr_X = np.concatenate([tr_X, current_aug]).astype(np.float32)
        tr_Y = np.concatenate([tr_Y, train_Y]).astype(np.int32)
    print(tr_X.shape, tr_Y.shape)
    tpot = TPOTClassifier(
        generations=20, population_size=5, verbosity=2, scoring="balanced_accuracy", cv=10,
        config_dict="TPOT light", random_state=int(args.seed)
    )
    tpot.fit(tr_X, tr_Y)
    tr_Yhat = tpot.fitted_pipeline_.predict(tr_X)
    train_Yhat = tpot.fitted_pipeline_.predict(train_X)
    test_Yhat = tpot.fitted_pipeline_.predict(test_X)
    results = {
        "train.reals": [float(a) for a in train_Y],
        "train.preds": [float(a) for a in train_Yhat],
        "train.aug.reals": [float(a) for a in tr_Y],
        "train.aug.preds": [float(a) for a in tr_Yhat],
        "test.reals": [float(a) for a in test_Y],
        "test.preds": [float(a) for a in test_Yhat],
    }
    OUT_MASK = op.join(args.output, op.split(args.dataset)[-1])
    with open(OUT_MASK+".json", "w") as oh:
        oh.write(json.dumps(results))
    tpot.export(OUT_MASK+".py")
    joblib.dump(tpot.fitted_pipeline_, OUT_MASK+".joblib")