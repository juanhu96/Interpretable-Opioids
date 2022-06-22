#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 14, 2022
@Author: Jingyuan Hu

Got rid of unconstrained RiskSLIM, regular CV, and unfound functions
"""

import os
import numpy as np
import pandas as pd
os.chdir('/Users/jingyuanhu/Desktop/Research/Interpretable Opioid/Code')

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import recall_score, precision_score, roc_auc_score,\
    average_precision_score, brier_score_loss, fbeta_score, accuracy_score
from sklearn.utils import shuffle

from pprint import pprint
from riskslim import load_data_from_csv, print_model
from riskslim.coefficient_set import CoefficientSet
from riskslim.lattice_cpa import run_lattice_cpa
from riskslim.lattice_cpa import setup_lattice_cpa, finish_lattice_cpa



def riskslim_prediction(X, feature_name, model_info):
    
    """
    @parameters
    
    X: test input features (np.array)
    feature_name: feature names
    model_info: output from RiskSLIM model
    
    """
    
    ## initialize parameters
    dictionary = {}
    prob = np.zeros(len(X))
    scores = np.zeros(len(X))
    
    ## prepare statistics
    subtraction_score = model_info['solution'][0]
    coefs = model_info['solution'][1:]
    index = np.where(coefs != 0)[0]
    
    nonzero_coefs = coefs[index]
    features = feature_name[index]
    X_sub = X[:,index]
    
    ## build dictionaries
    for i in range(len(features)):
        single_feature = features[i]
        coef = nonzero_coefs[i]
        dictionary.update({single_feature: coef})
        
    ## calculate probability
    for i in range(len(X_sub)):
        summation = 0
        for j in range(len(features)):
            a = X_sub[i,j]
            summation += dictionary[features[j]] * a
        scores[i] = summation
    
    prob = 1/(1+np.exp(-(scores + subtraction_score)))
    
    return prob



def riskslim_accuracy(X, Y, feature_name, model_info, threshold=0.5):
    
    prob = riskslim_prediction(X, feature_name, model_info)
    pred = np.mean((prob > threshold) == Y)
    
    return pred


###############################################################################################################################
###########################################     constrained RiskSLIM     ######################################################
###############################################################################################################################

    
def risk_slim_constrain(data, max_coefficient, max_L0_value, c0_value, max_offset, class_weight=None, max_runtime = 120, w_pos = 1):
    
    """
    @parameters:
    
    max_coefficient:  value of largest/smallest coefficient
    max_L0_value:     maximum model size (set as float(inf))
    max_offset:       maximum value of offset parameter (optional)
    c0_value:         L0-penalty parameter such that c0_value > 0; larger values -> 
                      sparser models; we set to a small value (1e-6) so that we get a model with max_L0_value terms
    max_runtime:      max algorithm running time
    w_pos:            relative weight on examples with y = +1; w_neg = 1.00 (optional)
    
    """
    
    # create coefficient set and set the value of the offset parameter
    coef_set = CoefficientSet(variable_names = data['variable_names'], lb = 0, ub = max_coefficient, sign = 0)
    # Jingyuan: get_conservative_offset is not defined
    # conservative_offset = get_conservative_offset(data, coef_set, max_L0_value)   
    # max_offset = min(max_offset, conservative_offset)
    coef_set['(Intercept)'].ub = max_offset
    coef_set['(Intercept)'].lb = -max_offset
    # coef_set.update_intercept_bounds(X = data['X'], y = data['Y'], max_offset = max_offset)
    
    constraints = {
        'L0_min': 0,
        'L0_max': max_L0_value,
        'coef_set':coef_set,
    }
    
    # Jingyuan: scale positive weight if balanced training
    if class_weight == 'balanced':
        w_pos = len(data['Y']) / sum(data['Y']==1)[0]
    
    # Set parameters
    settings = {
        # Problem Parameters
        'c0_value': c0_value,
        'w_pos': w_pos,

        # LCPA Settings
        'max_runtime': max_runtime,                         # max runtime for LCPA
        'max_tolerance': np.finfo('float').eps,             # tolerance to stop LCPA (set to 0 to return provably optimal solution)
        'display_cplex_progress': False,                    # print CPLEX progress on screen
        'loss_computation': 'lookup',                       # how to compute the loss function ('normal','fast','lookup')
        
        # LCPA Improvements
        'round_flag': False,                                # round continuous solutions with SeqRd
        'polish_flag': False,                               # polish integer feasible solutions with DCD
        'chained_updates_flag': False,                      # use chained updates
        'add_cuts_at_heuristic_solutions': True,            # add cuts at integer feasible solutions found using polishing/rounding
        
        # Initialization
        'initialization_flag': True,                        # use initialization procedure
        'init_max_runtime': 300.0,                          # max time to run CPA in initialization procedure
        'init_max_coefficient_gap': 0.49,

        # CPLEX Solver Parameters
        'cplex_randomseed': 0,                              # random seed
        'cplex_mipemphasis': 0,                             # cplex MIP strategy
    }
    

    # train model using lattice_cpa
    model_info, mip_info, lcpa_info = run_lattice_cpa(data, constraints, settings)
        
    return model_info, mip_info, lcpa_info


def risk_cv_constrain(X, 
                             Y,
                             y_label, 
                             max_coef,
                             max_coef_number,
                             c,
                             seed,
                             max_runtime=1000,
                             max_offset=100,
                             class_weight = None):

    ## set up data
    sample_weights = np.repeat(1, len(Y))

    ## set up cross validation
    outer_cv = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    inner_cv = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    
    train_auc = []
    validation_auc = []
    test_auc = []
    
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
    
    best_score = 0
    
    i = 0
    for outer_train, outer_test in outer_cv.split(X, Y):
        
        outer_train_x, outer_train_y = X.iloc[outer_train], Y[outer_train]
        outer_test_x, outer_test_y = X.iloc[outer_test], Y[outer_test]
        outer_train_sample_weight, outer_test_sample_weight = sample_weights[outer_train], sample_weights[outer_test]
        
        ## holdout test
        holdout_with_attrs = outer_test_x.copy().drop(['(Intercept)'], axis=1)            
        cols = outer_train_x.columns.tolist()
        
        
        ## inner cross validation
        for inner_train, validation in inner_cv.split(outer_train_x, outer_train_y):
            
            ## subset train data & store test data
            inner_train_x, inner_train_y = outer_train_x.iloc[inner_train].values, outer_train_y[inner_train]
            validation_x, validation_y = outer_train_x.iloc[validation].values, outer_train_y[validation]
            inner_train_sample_weight = outer_train_sample_weight[inner_train]
            validation_sample_weight = outer_train_sample_weight[validation]
            inner_train_y = inner_train_y.reshape(-1,1)
       
            ## create new data dictionary
            new_train_data = {
                'X': inner_train_x,
                'Y': inner_train_y,
                'variable_names': cols,
                'outcome_name': y_label,
                'sample_weights': inner_train_sample_weight
            }
                
            ## fit the model
            model_info, mip_info, lcpa_info = risk_slim_constrain(new_train_data, 
                                                                  max_coefficient=max_coef, 
                                                                  max_L0_value=max_coef_number, 
                                                                  c0_value=c, 
                                                                  max_runtime=max_runtime, 
                                                                  max_offset=max_offset,
                                                                  class_weight=class_weight)
            
            ## check validation auc
            validation_x = validation_x[:,1:] ## remove the first column, which is "intercept"
            validation_y[validation_y == -1] = 0 ## change -1 to 0
            validation_prob = riskslim_prediction(validation_x, np.array(cols), model_info)
            validation_auc.append(roc_auc_score(validation_y, validation_prob))
            
            ## Jingyuan: store the model_info with best performance (e.g. AUC)
            ## They didn't have this before, so the outer data is not performed on the best model of inner CV
            ## Can use other metrics
            # performance_metric = roc_auc_score(validation_y, validation_prob)
            # if performance_metric >= best_score:
            #     best_model_info = model_info
            #     best_score = performance_metric
            
        
        ## outer loop: use best model_info
        outer_train_x = outer_train_x.values
        outer_test_x = outer_test_x.values
        outer_train_y = outer_train_y.reshape(-1,1)
        new_train_data = {
            'X': outer_train_x,
            'Y': outer_train_y,
            'variable_names': cols,
            'outcome_name': y_label,
            'sample_weights': outer_train_sample_weight
        }
                
        
        ## fit the model
        # Jingyuan: confused here, shouldn't we use best model_info on outer test directly?
        model_info, mip_info, lcpa_info = risk_slim_constrain(new_train_data, 
                                                              max_coefficient=max_coef, 
                                                              max_L0_value=max_coef_number, 
                                                              c0_value=c, 
                                                              max_runtime=max_runtime, 
                                                              max_offset=max_offset,
                                                              class_weight=class_weight)
        print_model(model_info['solution'], new_train_data)  

        
        ## change data format
        outer_train_x, outer_test_x = outer_train_x[:,1:], outer_test_x[:,1:] ## remove the first column, which is "intercept"
        outer_train_y[outer_train_y == -1] = 0 ## change -1 to 0
        outer_test_y[outer_test_y == -1] = 0 ## change -1 to 0
        
        ## probability & accuracy
        outer_train_prob = riskslim_prediction(outer_train_x, np.array(cols), model_info).reshape(-1,1)
        outer_test_prob = riskslim_prediction(outer_test_x, np.array(cols), model_info)
        outer_test_pred = (outer_test_prob > 0.5)

        ########################
        ## AUC
        train_auc.append(roc_auc_score(outer_train_y, outer_train_prob))
        test_auc.append(roc_auc_score(outer_test_y, outer_test_prob)) 
        
        ########################
        ## store results
        # holdout_with_attrs_test.append(holdout_with_attrs)
        holdout_probability.append(outer_test_prob)
        holdout_prediction.append(outer_test_pred)
        holdout_y.append(outer_test_y)
        holdout_accuracy.append(accuracy_score(outer_test_y, outer_test_pred))
        holdout_recall.append(recall_score(outer_test_y, outer_test_pred))
        holdout_precision.append(precision_score(outer_test_y, outer_test_pred))
        holdout_roc_auc.append(roc_auc_score(outer_test_y, outer_test_prob))
        holdout_pr_auc.append(average_precision_score(outer_test_y, outer_test_prob))
        holdout_brier.append(brier_score_loss(outer_test_y, outer_test_prob))
        holdout_f1.append(fbeta_score(outer_test_y, outer_test_pred, beta = 1))
        holdout_f2.append(fbeta_score(outer_test_y, outer_test_pred, beta = 2))
        
        i += 1
        
    
    return {'train_auc': train_auc,
            'validation_auc': validation_auc,
            'holdout_test_auc': test_auc, 
            'holdout_test_accuracy': holdout_accuracy,
            'holdout_test_recall': holdout_recall,
            "holdout_test_precision": holdout_precision,
            'holdout_test_roc_auc': holdout_roc_auc,
            'holdout_test_pr_auc': holdout_pr_auc,
            "holdout_test_brier": holdout_brier,
            'holdout_test_f1': holdout_f1,
            "holdout_test_f2": holdout_f2}



def risk_nested_cv_constrain(X, 
                             Y,
                             y_label, 
                             max_coef,
                             max_coef_number,
                             c,
                             seed,
                             max_runtime=1000,
                             max_offset=100,
                             class_weight = None):

    """
    Implemented the nested cross-validation step for hyperparameter tuning
    For each inner_cv, pick one combination
    Find the best set of hyperparameter, fit it for outer_cv
    
    This leads to #combinations * 5 inner * 5 outer
    """
    
    
    ## set up data
    sample_weights = np.repeat(1, len(Y))

    ## set up cross validation
    outer_cv = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    inner_cv = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    
    train_auc = []
    validation_auc = []
    test_auc = []
    
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
    
    best_score = 0
    
    i = 0
    for outer_train, outer_test in outer_cv.split(X, Y):
        
        outer_train_x, outer_train_y = X.iloc[outer_train], Y[outer_train]
        outer_test_x, outer_test_y = X.iloc[outer_test], Y[outer_test]
        outer_train_sample_weight, outer_test_sample_weight = sample_weights[outer_train], sample_weights[outer_test]
        
        ## holdout test
        holdout_with_attrs = outer_test_x.copy().drop(['(Intercept)'], axis=1)            
        cols = outer_train_x.columns.tolist()
        
        j = 0
        ## inner cross validation
        for inner_train, validation in inner_cv.split(outer_train_x, outer_train_y):
            
            ## subset train data & store test data
            inner_train_x, inner_train_y = outer_train_x.iloc[inner_train].values, outer_train_y[inner_train]
            validation_x, validation_y = outer_train_x.iloc[validation].values, outer_train_y[validation]
            inner_train_sample_weight = outer_train_sample_weight[inner_train]
            validation_sample_weight = outer_train_sample_weight[validation]
            inner_train_y = inner_train_y.reshape(-1,1)
       
            ## create new data dictionary
            new_train_data = {
                'X': inner_train_x,
                'Y': inner_train_y,
                'variable_names': cols,
                'outcome_name': y_label,
                'sample_weights': inner_train_sample_weight
            }
            
            
            ## fit the model: CV for max_coef_number
            model_info, mip_info, lcpa_info = risk_slim_constrain(new_train_data, 
                                                                  max_coefficient=max_coef, 
                                                                  max_L0_value=max_coef_number[j], 
                                                                  c0_value=c, 
                                                                  max_runtime=max_runtime, 
                                                                  max_offset=max_offset,
                                                                  class_weight=class_weight)
            
            ## check validation auc
            validation_x = validation_x[:,1:] ## remove the first column, which is "intercept"
            validation_y[validation_y == -1] = 0 ## change -1 to 0
            validation_prob = riskslim_prediction(validation_x, np.array(cols), model_info)
            validation_auc.append(roc_auc_score(validation_y, validation_prob))
            
            ## Jingyuan: store the model_info with best performance (e.g. AUC)
            performance_metric = roc_auc_score(validation_y, validation_prob)
            if performance_metric >= best_score:
                best_max_coef_number = max_coef_number[j]
                best_score = performance_metric
            
            j += 1
            
        
        ## outer loop: use best model_info
        outer_train_x = outer_train_x.values
        outer_test_x = outer_test_x.values
        outer_train_y = outer_train_y.reshape(-1,1)
        new_train_data = {
            'X': outer_train_x,
            'Y': outer_train_y,
            'variable_names': cols,
            'outcome_name': y_label,
            'sample_weights': outer_train_sample_weight
        }
                
        ## fit the model
        model_info, mip_info, lcpa_info = risk_slim_constrain(new_train_data, 
                                                              max_coefficient=max_coef, 
                                                              max_L0_value=best_max_coef_number, 
                                                              c0_value=c, 
                                                              max_runtime=max_runtime, 
                                                              max_offset=max_offset,
                                                              class_weight=class_weight)
        print_model(model_info['solution'], new_train_data)  

        
        ## change data format
        outer_train_x, outer_test_x = outer_train_x[:,1:], outer_test_x[:,1:] ## remove the first column, which is "intercept"
        outer_train_y[outer_train_y == -1] = 0 ## change -1 to 0
        outer_test_y[outer_test_y == -1] = 0 ## change -1 to 0
        
        ## probability & accuracy
        outer_train_prob = riskslim_prediction(outer_train_x, np.array(cols), model_info).reshape(-1,1)
        outer_test_prob = riskslim_prediction(outer_test_x, np.array(cols), model_info)
        outer_test_pred = (outer_test_prob > 0.5)

        ########################
        ## AUC
        train_auc.append(roc_auc_score(outer_train_y, outer_train_prob))
        test_auc.append(roc_auc_score(outer_test_y, outer_test_prob)) 
        
        ########################
        ## store results
        # holdout_with_attrs_test.append(holdout_with_attrs)
        holdout_probability.append(outer_test_prob)
        holdout_prediction.append(outer_test_pred)
        holdout_y.append(outer_test_y)
        holdout_accuracy.append(accuracy_score(outer_test_y, outer_test_pred))
        holdout_recall.append(recall_score(outer_test_y, outer_test_pred))
        holdout_precision.append(precision_score(outer_test_y, outer_test_pred))
        holdout_roc_auc.append(roc_auc_score(outer_test_y, outer_test_prob))
        holdout_pr_auc.append(average_precision_score(outer_test_y, outer_test_prob))
        holdout_brier.append(brier_score_loss(outer_test_y, outer_test_prob))
        holdout_f1.append(fbeta_score(outer_test_y, outer_test_pred, beta = 1))
        holdout_f2.append(fbeta_score(outer_test_y, outer_test_pred, beta = 2))
        
        i += 1
        
    
    return {'train_auc': train_auc,
            'validation_auc': validation_auc,
            'holdout_test_auc': test_auc, 
            'holdout_test_accuracy': holdout_accuracy,
            'holdout_test_recall': holdout_recall,
            "holdout_test_precision": holdout_precision,
            'holdout_test_roc_auc': holdout_roc_auc,
            'holdout_test_pr_auc': holdout_pr_auc,
            "holdout_test_brier": holdout_brier,
            'holdout_test_f1': holdout_f1,
            "holdout_test_f2": holdout_f2}


