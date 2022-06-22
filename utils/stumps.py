#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 30 2022
@Author: Jingyuanhu
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, \
    roc_auc_score, average_precision_score, brier_score_loss, fbeta_score
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def create_stumps(data, columns, cutpoints):
    
    """
    @parameters:
    
    - data: featres; np.array
    - columns: feature names
    - cutpoints: cut off points used to create stumps
    
    """
    
    ## data dimension
    final_data = []
    final_names = []
    n, p = data.shape[0], data.shape[1]
    
    ## loop through features
    for i in range(len(columns)):
        ## subset feature
        feature = columns[i]
        feature_values = data[:,i]
        cutoff = cutpoints[i]
        cutoff_length = len(cutoff)
        names = []
        
        ## create stumps
        ### if the variable is 'p_current_age' or 'p_age_first_offense', then we would want to use 
        ### '<=' intervals. For other variables, we use '>=' intervals
        ### if the variable is binary, then set the cutoff point value to be 1.
        
        if ((feature == 'age_at_current_charge') | (feature == 'age_at_first_charge')):
            stumps = np.zeros([n, cutoff_length])
            for k in range(cutoff_length):
                for j in range(n):
                    if feature_values[j] <= cutoff[k]: stumps[j,k] = 1
                names.append(feature + str(cutoff[k]))
        else: 
            stumps = np.zeros([n, cutoff_length])
            for k in range(cutoff_length):
                for j in range(n):
                    if feature_values[j] >= cutoff[k]: stumps[j,k] = 1
                names.append(feature + str(cutoff[k]))
        
        ## store stumps
        final_data.append(stumps)
        final_names.append(names)
        
        ## post process
        new_data = pd.DataFrame(final_data[0], columns=final_names[0])
        for s in range(len(final_data)-1):
            a = pd.DataFrame(final_data[s+1], columns=final_names[s+1])
            new_data = pd.concat([new_data, a], axis=1)
    
    return new_data



########################
def stump_cv(X, Y, columns, c_grid, seed, class_weight=None):
    
    ## estimator
    lasso = LogisticRegression(class_weight=class_weight, solver='liblinear', 
                               random_state=seed, penalty='l1')
    
    ## rename key to adapt pipeline
    c_grid['estimator__C'] = c_grid.pop('C')
    
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
    
    holdout_with_attrs_test = []
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
        
        ## GridSearch: inner CV
        pipeline = Pipeline(steps=[('over', SMOTE(sampling_strategy=0.1)), 
                                    ('under', RandomUnderSampler(sampling_strategy=0.5)),
                                    ('estimator', lasso)])
        
        clf = GridSearchCV(estimator=pipeline, 
                            param_grid=c_grid, 
                            scoring='average_precision',
                            cv=inner_cv, 
                            return_train_score=True).fit(train_x, train_y)
        # clf = GridSearchCV(estimator=lasso, 
        #                     param_grid=c_grid, 
        #                     scoring='average_precision',
        #                     cv=inner_cv, 
        #                     return_train_score=True).fit(train_x, train_y)
    
        ## best parameter & scores        
        mean_train_score = clf.cv_results_['mean_train_score']
        mean_test_score = clf.cv_results_['mean_test_score']        
        best_param = clf.best_params_
        train_auc.append(mean_train_score[np.where(mean_test_score == clf.best_score_)[0][0]])
        validation_auc.append(clf.best_score_)
        auc_diffs.append(mean_train_score[np.where(mean_test_score == clf.best_score_)[0][0]] - clf.best_score_)
        
        ## run model with best parameter
        best_model = LogisticRegression(class_weight=class_weight, 
                                        solver='liblinear', 
                                        random_state=seed, 
                                        penalty='l1', 
                                        C=best_param['estimator__C']).fit(train_x, train_y)   
        coefs = best_model.coef_[best_model.coef_ != 0]
        features = columns[best_model.coef_[0] != 0].tolist()
        intercept = best_model.intercept_[0]
        
        ## dictionary
        lasso_dict = {}
        for j in range(len(features)):
            lasso_dict.update({features[j]: coefs[j]})
        
        ## prediction on test set
        prob = 0
        for k in features:
            test_values = test_x[k]*(lasso_dict[k])
            prob += test_values
        holdout_prob = np.exp(prob)/(1+np.exp(prob))
        holdout_pred = (holdout_prob > 0.5)
        
        ########################
        ## store results
        best_params.append(best_param)
        holdout_probability.append(holdout_prob)
        holdout_prediction.append(holdout_pred)
        holdout_y.append(test_y)
        holdout_accuracy.append(accuracy_score(test_y, holdout_pred))
        holdout_recall.append(recall_score(test_y, holdout_pred))
        holdout_precision.append(precision_score(test_y, holdout_pred))
        holdout_roc_auc.append(roc_auc_score(test_y, holdout_prob))
        holdout_pr_auc.append(average_precision_score(test_y, holdout_prob))
        holdout_brier.append(brier_score_loss(test_y, holdout_prob))
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



########################
def stump_model(X_train, Y_train, X_test, Y_test, c, columns, seed):
    
    ## remove unused feature in modeling
    X_train = X_train.drop(['person_id', 'screening_date', 'race'], axis=1)
    X_test = X_test.drop(['person_id', 'screening_date', 'race'], axis=1)
    
    ## estimator
    lasso = LogisticRegression(class_weight = 'balanced', 
                               solver='liblinear', 
                               random_state=seed, 
                               penalty='l1', 
                               C = c).fit(X_train, Y_train)
    coefs = lasso.coef_[lasso.coef_ != 0]
    features = columns[lasso.coef_[0] != 0].tolist()
    intercept = lasso.intercept_[0]
     
    ## dictionary
    lasso_dict = {}
    for i in range(len(features)):
        lasso_dict.update({features[i]: coefs[i]})
    
    ## prediction on test set
    prob = 0
    for k in features:
        test_values = X_test[k]*(lasso_dict[k])
        prob += test_values
    
    holdout_prob = np.exp(prob)/(1+np.exp(prob))
    test_auc = roc_auc_score(Y_test, holdout_prob)
    
    return {'coefs': coefs, 
            'features': features, 
            'intercept': intercept, 
            'dictionary': lasso_dict, 
            'test_auc': test_auc}

  

