# -*- coding: utf-8 -*-


# %% imports

import csv
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

# %% set seeds

import numpy as np
np.random.seed(42)
import random
random.seed(42)

# %% view example file set

with open('train.tsv', encoding='utf-8') as iFile:
    count = 0
    for x in iFile:
        print(x.strip())
        count += 1
        if count == 2:
            break

# %% import training set to lists

X_txt_train = []
y_train = []
with open('train.tsv', encoding='utf-8') as iFile:
    iCSV = csv.reader(iFile, delimiter='\t')
    for row in iCSV:
        X_txt_train.append(row[1])
        y_train.append(row[2])

# %% check train import

print(X_txt_train[:2])
print(y_train[:2])
print(len(X_txt_train), len(y_train))

# %% import test set to lists

X_txt_test = []
y_test = []

with open('test.tsv', encoding='utf-8') as iFile:
    iCSV = csv.reader(iFile, delimiter='\t')
    for row in iCSV:
        X_txt_test.append(row[1])
        y_test.append(row[2])

# %% check test import

print(X_txt_test[:10])
print(y_test[:10])
print(len(X_txt_test), len(y_test))

# %% Count -> Tfidf Vectorizer

X_train = np.array(X_txt_train)
X_test = np.array(X_txt_test)

vec = TfidfVectorizer(ngram_range = (1,2), stop_words='english', max_df=0.45)

X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

print(X_train.shape, X_test.shape)

# %% import word lists for counts

bad_words = set()
with open('bad-words.txt') as iFile:
    for row in iFile:
        bad_words.add(row.strip())

print(len(bad_words))

profanity = set()
with open('google-profanity.txt') as iFile:
    for row in iFile:
        profanity.add(row.strip())

print(len(profanity))

you_syn = {'you', 'u', 'your', 'ur', "you're", 'urs', 'yours', "u're"}

# %% count features

import re

X_train_counts = []
X_test_counts = []


for tweet in X_txt_train:
    num_bad_words = 0
    num_profanity = 0
    num_yousyn = 0
    for word in tweet.lower().split():
        if word in bad_words:
            num_bad_words += 1
        if word in profanity:
            num_profanity += 1
        if word in you_syn:
            num_yousyn += 1
    spcl = re.findall(r'[^\s\w\'\.]',tweet)
    num_spcl = len(spcl)
    num_agg = num_yousyn + num_profanity + num_bad_words
    X_train_counts.append([num_bad_words, num_profanity, 
                           num_spcl, num_yousyn, num_agg])

print(len(X_train_counts))


for tweet in X_txt_test:
    num_bad_words = 0
    num_profanity = 0
    num_yousyn = 0
    for word in tweet.lower().split():
        if word in bad_words:
            num_bad_words += 1
        if word in profanity:
            num_profanity += 1
        if word in you_syn:
            num_yousyn += 1
    spcl = re.findall(r'[^\s\w\'\.]',tweet)
    num_spcl = len(spcl)
    num_agg = num_yousyn + num_profanity + num_bad_words
    X_test_counts.append([num_bad_words, num_profanity, 
                          num_spcl, num_yousyn, num_agg])

print(len(X_test_counts))


# %% join arrays

import scipy.sparse as sp

X_train_ctarr = np.array(X_train_counts)
X_test_ctarr = np.array(X_test_counts)

X_train_w_ct = sp.hstack([X_train, X_train_ctarr])
X_test_w_ct = sp.hstack([X_test, X_test_ctarr])

print(X_train_w_ct.shape, X_test_w_ct.shape)

# %% classifier and params

svc = LinearSVC()
param = {'C': [1., 10, 100]}

clf = GridSearchCV(svc, param, cv=7, scoring='f1_macro')

# %% FIT & score

clf.fit(X_train_w_ct, y_train)

Cf1 = clf.best_score_
Cbparams = clf.best_params_
print('Validation F1: {:.4f}'.format(Cf1))
print('Best Parameters: ', Cbparams)

# %% metrics

from sklearn.metrics import precision_score

trainpreds = clf.predict(X_train_w_ct)

macrop = precision_score(y_train, trainpreds, average='macro')
microp = precision_score(y_train, trainpreds, average='micro')
wtdp = precision_score(y_train, trainpreds, average='weighted')

print('Macro Precision: ', macrop)
print('Micro Precision: ', microp)
print('Weighted Precision: ', wtdp)

acc_score = accuracy_score(y_train, trainpreds)

print('Accuracy: ', acc_score)

macrof1 = f1_score(y_train, trainpreds, average='macro')
microf1 = f1_score(y_train, trainpreds, average='micro')
wtdf1 = f1_score(y_train, trainpreds, average='weighted')

print('Macro F1: ', macrof1)
print('Micro F1: ', microf1)
print('Weighted F1: ', wtdf1)


# %% view test sample

print(X_txt_test[:5], y_test[:5])

# %% assign predictions

preds = clf.predict(X_test_w_ct)

for x in range(len(y_test)):
    y_test[x] = preds[x]

print(y_test[:5])

# %% save predictions

TeamApreds = open('TeamApreds.txt', 'w')

for entry in preds:
    TeamApreds.write(entry)
    TeamApreds.write("\n")

TeamApreds.close()