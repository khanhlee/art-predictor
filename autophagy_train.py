#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os


data_dir = 'features/'


def load_data(fpos, fneg):
    df_pos = pd.read_csv(os.path.join(data_dir, fpos), delimiter='\t')
    X_pos = df_pos.iloc[:,1:]
    y_pos = np.full(df_pos.shape[0], 1)
    
    df_neg = pd.read_csv(os.path.join(data_dir, fneg), delimiter='\t')
    X_neg = df_neg.iloc[:,1:]
    y_neg = np.full(df_neg.shape[0], 0)
    
    X = pd.concat([X_pos, X_neg], axis=0)
    y = np.concatenate((y_pos, y_neg))
    
    return X, y


X_trn_AAC, y_trn_AAC = load_data('autophagy.AAC.trn.csv', 'non-autophagy.AAC.trn.csv')
X_trn_APAAC, y_trn_APAAC = load_data('autophagy.APAAC.trn.csv', 'non-autophagy.APAAC.trn.csv')
X_trn_CKSAAGP, y_trn_CKSAAGP = load_data('autophagy.CKSAAGP.trn.csv', 'non-autophagy.CKSAAGP.trn.csv')
X_trn_CKSAAP, y_trn_CKSAAP = load_data('autophagy.CKSAAP.trn.csv', 'non-autophagy.CKSAAP.trn.csv')
X_trn_CTriad, y_trn_CTriad = load_data('autophagy.CTriad.trn.csv', 'non-autophagy.CTriad.trn.csv')
X_trn_DPC, y_trn_DPC = load_data('autophagy.DPC.trn.csv', 'non-autophagy.DPC.trn.csv')
X_trn_GAAC, y_trn_GAAC = load_data('autophagy.GAAC.trn.csv', 'non-autophagy.GAAC.trn.csv')
X_trn_GDPC, y_trn_GDPC = load_data('autophagy.GDPC.trn.csv', 'non-autophagy.GDPC.trn.csv')
X_trn_Geary, y_trn_Geary = load_data('autophagy.Geary.trn.csv', 'non-autophagy.Geary.trn.csv')
X_trn_Moran, y_trn_Moran = load_data('autophagy.Moran.trn.csv', 'non-autophagy.Moran.trn.csv')
X_trn_NMBroto, y_trn_NMBroto = load_data('autophagy.NMBroto.trn.csv', 'non-autophagy.NMBroto.trn.csv')
X_trn_PAAC, y_trn_PAAC = load_data('autophagy.PAAC.trn.csv', 'non-autophagy.PAAC.trn.csv')
X_trn_QSOrder, y_trn_QSOrder = load_data('autophagy.QSOrder.trn.csv', 'non-autophagy.QSOrder.trn.csv')


X_trn_hybrid = pd.concat([X_trn_AAC, X_trn_APAAC, X_trn_CKSAAGP, X_trn_CKSAAP, X_trn_CTriad,
                         X_trn_DPC, X_trn_GAAC, X_trn_GDPC, X_trn_Geary, X_trn_Moran,
                         X_trn_NMBroto, X_trn_PAAC, X_trn_QSOrder], axis=1)


y_trn = y_trn_AAC


feat_dict = {'AAC':X_trn_AAC, 'APAAC':X_trn_APAAC, 'CKSAAGP':X_trn_CKSAAGP, 'CKSAAP':X_trn_CKSAAP,
            'CTriad':X_trn_CTriad, 'DPC':X_trn_DPC, 'GAAC':X_trn_GAAC, 'GDPC':X_trn_GDPC,
            'Geary':X_trn_Geary, 'Moran':X_trn_Moran, 'NMBroto':X_trn_NMBroto, 'PAAC':X_trn_PAAC, 
            'QSOrder':X_trn_QSOrder}


import scipy
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

kfold = StratifiedKFold(n_splits=5, shuffle=True)
from sklearn import metrics

for key, value in feat_dict.items():
    TP = FP = TN = FN = 0
    acc_cv_scores = []
    auc_cv_scores = []
    for train, test in kfold.split(value, y_trn):
        svm_model = XGBClassifier() 
        ## evaluate the model
        svm_model.fit(value.iloc[train], y_trn[train])
        # evaluate the model
        true_labels = np.asarray(y_trn[test])
        predictions = svm_model.predict(value.iloc[test])
        acc_cv_scores.append(accuracy_score(true_labels, predictions))
        # print(confusion_matrix(true_labels, predictions))
        newTN, newFP, newFN, newTP = confusion_matrix(true_labels,predictions).ravel()
        TP += newTP
        FN += newFN
        FP += newFP
        TN += newTN
        pred_prob = svm_model.predict_proba(value.iloc[test])
        fpr, tpr, thresholds = metrics.roc_curve(true_labels, pred_prob[:,1], pos_label=1)
        auc_cv_scores.append(metrics.auc(fpr, tpr))
        # print('AUC = ', metrics.auc(fpr, tpr))
        # print('AUC = ', round(metrics.roc_auc_score(true_labels, predictions)*100,2))

    print('\nFeature: ', key)
    print('Accuracy = ', np.mean(acc_cv_scores))
    print('TP = %s, FP = %s, TN = %s, FN = %s' % (TP, FP, TN, FN))
    print('AUC = ', np.mean(auc_cv_scores))


# ## Imbalance

from imblearn.over_sampling import SMOTE
ros = SMOTE()
# Re-train
for key, value in feat_dict.items():
    TP = FP = TN = FN = 0
    acc_cv_scores = []
    auc_cv_scores = []
    for train, test in kfold.split(value, y_trn):
        train_x, train_y = value.iloc[train], y_trn[train]
        X_ros, y_ros = ros.fit_resample(train_x, train_y)
        svm_model = XGBClassifier() 
        ## evaluate the model
        svm_model.fit(X_ros, y_ros)
        # evaluate the model
        true_labels = np.asarray(y_trn[test])
        predictions = svm_model.predict(value.iloc[test])
        acc_cv_scores.append(accuracy_score(true_labels, predictions))
        # print(confusion_matrix(true_labels, predictions))
        newTN, newFP, newFN, newTP = confusion_matrix(true_labels,predictions).ravel()
        TP += newTP
        FN += newFN
        FP += newFP
        TN += newTN
        pred_prob = svm_model.predict_proba(value.iloc[test])
        fpr, tpr, thresholds = metrics.roc_curve(true_labels, pred_prob[:,1], pos_label=1)
        auc_cv_scores.append(metrics.auc(fpr, tpr))
        # print('AUC = ', metrics.auc(fpr, tpr))
        # print('AUC = ', round(metrics.roc_auc_score(true_labels, predictions)*100,2))

    print('\nFeature: ', key)
    print('Accuracy = ', np.mean(acc_cv_scores))
    print('TP: {}, FP: {}, TN: {}, FN: {}'.format(TP, FN, TN, FP))
    print('AUC: {}'.format(np.mean(auc_cv_scores)))
