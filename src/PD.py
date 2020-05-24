#!/usr/bin/env python
# coding: utf-8

# In[1]:


import io
import json
import torch
import joblib
import argparse
from misc import *
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from copy import deepcopy
from tpot import TPOTClassifier
from autoPyTorch import AutoNetImageClassification
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
    X = data.values[:, 1:-1]
    Y = data.values[:, -1:].reshape(-1,)
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2)
    if "_1hot_" not in args.dataset:
        tr_X = deepcopy(train_X).astype(np.float32)
        tr_Y = deepcopy(train_Y).astype(np.int32)
    else:
        tr_X = deepcopy(train_X)
        tr_Y = deepcopy(train_Y).astype(np.int32)
    print(tr_X.shape, tr_Y.shape)
    for a in range(int(args.augs)):
        current_aug = train_X*np.random.normal(
            size=train_X.shape, loc=1, scale=0.1
        )
        tr_X = np.concatenate([tr_X, current_aug]).astype(np.float32)
        tr_Y = np.concatenate([tr_Y, train_Y]).astype(np.int32)
    print(tr_X.shape, tr_Y.shape)
    if "1hot" not in args.dataset:
        tpot = TPOTClassifier(
            generations=20, population_size=5,
            verbosity=2, scoring="balanced_accuracy", cv=10,
            config_dict="TPOT light", random_state=int(args.seed)
        )
        tpot.fit(tr_X, tr_Y)
        tr_Yhat = tpot.fitted_pipeline_.predict(tr_X)
        train_Yhat = tpot.fitted_pipeline_.predict(train_X)
        test_Yhat = tpot.fitted_pipeline_.predict(test_X)
        tpot.export(args.output+".py")
        joblib.dump(tpot.fitted_pipeline_, args.output+".joblib")
    else:
        auto = AutoNetImageClassification(
            "medium_cs",
            log_level='info',
            max_runtime=100,
            min_budget=15,
            max_budget=45
        )
        tr_X = np.array([op.join("../images/", a[0]+".png") for a in tr_X])
        train_X = np.array(
            [op.join("../images/", a[0]+".png") for a in train_X]
        )
        test_X = np.array([op.join("../images/", a[0]+".png") for a in test_X])
        auto.fit(tr_X, tr_Y)
        tr_Yhat = np.argmax(auto.predict(tr_X)["Y"], 1)
        train_Yhat = np.argmax(auto.predict(train_X)["Y"], 1)
        test_Yhat = np.argmax(auto.predict(test_X)["Y"], 1)
    results = {
        "train.reals": [float(a) for a in train_Y],
        "train.preds": [float(a) for a in train_Yhat],
        "train.aug.reals": [float(a) for a in tr_Y],
        "train.aug.preds": [float(a) for a in tr_Yhat],
        "test.reals": [float(a) for a in test_Y],
        "test.preds": [float(a) for a in test_Yhat],
    }
    with open(args.output+".json", "w") as oh:
        oh.write(json.dumps(results))
