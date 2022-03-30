#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 14 2022
@Author: Jingyuan Hu
"""

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import recall_score, precision_score, roc_auc_score,\
    average_precision_score, brier_score_loss, fbeta_score
from sklearn.calibration import CalibratedClassifierCV
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks


def nested_cross_validate(X, Y, estimator, c_grid, seed, index = None):
    
    ## outer cv
    train_outer = []
    test_outer = []
    outer_cv = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    
    ## 5 sets of train & test index
    for train, test in outer_cv.split(X, Y):
        train_outer.append(train)
        test_outer.append(test)
        
    ## storing lists
    best_params = []
    train_auc = []
    validation_auc = []
    auc_diffs = []
    
    holdout_with_attr_test = []
    holdout_prediction = []
    holdout_probability = []
    holdout_y = []
    holdout_accuracy = []
    holdout_recall = []
    holdout_precision = []
    holdout_roc_auc = []
    holdout_pr_auc = []
    holdout_f1 = []
    holdout_f2 = []
    holdout_brier = []
    
    ## inner cv
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    for i in range(len(train_outer)):
        
        ## subset train & test sets in inner loop
        train_x, test_x = X.iloc[train_outer[i]], X.iloc[test_outer[i]]
        train_y, test_y = Y[train_outer[i]], Y[test_outer[i]]
        
        
        ## holdout test with "race" for fairness
        holdout_with_attrs = test_x.copy()
      
        ## GridSearch: inner CV
        pipeline = Pipeline(steps=[('sampler', SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))),
                                   ('estimator', estimator)])
        
        ### Jingyuan: to specify grid on estimator, need to add 'estimator__' 
        clf = GridSearchCV(estimator=pipeline, 
                           param_grid=c_grid, 
                           scoring='average_precision',
                           cv=inner_cv, 
                           return_train_score=True).fit(train_x, train_y)        

        ## best parameter & scores
        mean_train_score = clf.cv_results_['mean_train_score']
        mean_test_score = clf.cv_results_['mean_test_score']        
        best_param = clf.best_params_
        train_auc.append(mean_train_score[np.where(mean_test_score == clf.best_score_)[0][0]])
        validation_auc.append(clf.best_score_)
        auc_diffs.append(mean_train_score[np.where(mean_test_score == clf.best_score_)[0][0]] - clf.best_score_)

        ## train model on best param
        if index == 'svm':
            best_model = CalibratedClassifierCV(clf, cv=5)
            best_model.fit(train_x, train_y)
            prob = best_model.predict_proba(test_x)[:, 1]
            holdout_pred = best_model.predict(test_x)
            holdout_acc = best_model.score(test_x, test_y)            
        else:
            prob = clf.predict_proba(test_x)[:, 1]
            holdout_pred = clf.predict(test_x)
            holdout_acc = clf.score(test_x, test_y)
        
        ## store results
        best_params.append(best_param)
        # holdout_with_attr_test.append(holdout_with_attrs)
        holdout_probability.append(prob)
        holdout_prediction.append(holdout_pred)
        holdout_y.append(test_y)
        holdout_accuracy.append(holdout_acc)
        holdout_recall.append(recall_score(test_y, holdout_pred))
        holdout_precision.append(precision_score(test_y, holdout_pred))
        holdout_roc_auc.append(roc_auc_score(test_y, prob))
        holdout_pr_auc.append(average_precision_score(test_y, prob))
        holdout_brier.append(brier_score_loss(test_y, prob))
        holdout_f1.append(fbeta_score(test_y, holdout_pred, beta = 1))
        holdout_f2.append(fbeta_score(test_y, holdout_pred, beta = 2))
        


    return {'best_param': best_params,
            'train_auc': train_auc,
            'validation_auc': validation_auc,
            'auc_diffs': auc_diffs,
            'holdout_test_accuracy': holdout_accuracy,
            'holdout_test_recall': holdout_recall,
            "holdout_test_precision": holdout_precision,
            'holdout_test_roc_auc': holdout_roc_auc,
            'holdout_test_pr_auc': holdout_pr_auc,
            "holdout_test_brier": holdout_brier,
            'holdout_test_f1': holdout_f1,
            "holdout_test_f2": holdout_f2}


