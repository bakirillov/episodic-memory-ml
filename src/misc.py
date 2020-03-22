import re
import os
import numpy as np
import pandas as pd
import os.path as op
import pickle as pkl
from copy import deepcopy
from scipy.stats import ks_2samp, median_test
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import spearmanr, pearsonr, ks_2samp, chisquare, levene


class Study():
    
    def __init__(self, fns, dfs):
        self.fns = fns
        self.dfs = dfs
        
    def __getitem__(self, key):
        return(self.fns[key], self.dfs[key])
    
    def __len__(self):
        return(len(self.fns))
        
    @classmethod
    def load_from_file(cls, fn):
        if op.splitext(fn)[-1] != ".pkl":
            fns = [fn]
            dfs = [pd.read_excel(fn, header=None).dropna()]
            for a in range(len(dfs)):
                dfs[a]["answers"] = dfs[a][4].apply(Study.get_answers)
            u = cls(fns, dfs)
        else:
            with open(fn, "rb") as ih:
                u = pkl.load(ih)
        return(u)
        
    @staticmethod
    def get_answers(x):
        return(1 if x == 1 else 0)
    
    @staticmethod
    def clamp(a, n=1):
        mna = np.mean(a)
        sta = np.std(a)
        la = np.logical_not(a < mna - n*sta)
        ga = np.logical_not(a > mna + n*sta)
        return(a[np.logical_and(la,ga)])
    
    @staticmethod
    def safe_roc_auc(y_true, y_score):
        try:
            r = roc_auc_score(y_true=y_true, y_score=y_score)
        except:
            r = 0.5
        finally:
            return(r)
    
    def compute_word_set(self):
        return(sum([list(a[2]) for a in self.dfs], []))
    
    def compute_study_aucs(self):
        return(
            [Study.safe_roc_auc(y_true=a[5], y_score=a["answers"]) for a in self.dfs]
        )
    
    def get_participant_RT(self):
        return([(a[0], list(a[1][6])) for a in self])
    
    def get_word_RT(self):
        word_set = self.compute_word_set()
        RTs = {a:[] for a in word_set}
        for _,b in self:
            current_words = b[2].values
            current_RTs = b[6].values
            for w,t in zip(current_words, current_RTs):
                RTs[w].append(t)
        return(RTs)
    
    def get_all_RT(self):
        RT = []
        for _,b in self:
            RT.extend(b[6].values)
        return(np.array(RT))
    
    def compute_word_aucs(self):
        word_set = self.compute_word_set()
        answers = {a:[] for a in word_set}
        reals = {a:[] for a in word_set}
        for _,b in self:
            current_words = b[2].values
            current_answers = b["answers"].values
            current_reals = b[5].values
            for w,a,r in zip(current_words, current_answers, current_reals):
                answers[w].append(a)
                reals[w].append(r)
        aucs = {w:[Study.safe_roc_auc(y_true=reals[w], y_score=answers[w])] for w in word_set}
        return(pd.DataFrame(aucs).T)
    
    def __sub__(self, other):
        a_a = deepcopy(self.fns)
        inds = [a_a.index(k) for k in other.fns]
        b_b = deepcopy(self.dfs)
        a = []
        b = []
        for k in range(len(a_a)):
            if k not in inds:
                a.append(a_a[k])
                b.append(b_b[k])
        return(Study(a, b))
    
    def __add__(self, other):
        a = deepcopy(self.fns)
        a.extend(other.fns)
        b = deepcopy(self.dfs)
        b.extend(other.dfs)
        return(Study(a, b))
    
    def __neg__(self):
        u = Study(self.fns, self.dfs)
        return(u)
    
    def __radd__(self, other):
        return(self)
    
    def save(self, fn):
        with open(fn, "wb") as oh:
            pkl.dump(self, oh)