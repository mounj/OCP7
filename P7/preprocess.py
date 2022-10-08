############# LIBRAIRY #############
import time
from datetime import date
import numpy as np
import os
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import missingno as msno
from contextlib import contextmanager
from P7 import params, utils

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import manifold, decomposition
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import RandomForestClassifier


# Travail sur un jeu de donnée non équilibré
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import sys
sys.path.insert(0, "../data")
sys.path.insert(0, "../models")

import pickle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def loadData():
    df = pd.read_csv(params.PATH_OUTPUT + 'new_train_application.csv')
    return df

def PreprocessData(df):
    # Create train labels
    train_labels = df['TARGET'].astype(int)
    train = df.drop(columns='TARGET').copy()

    # Drop the IDs and record them for later
    train_ids = train['SK_ID_CURR']
    train.drop(columns = ['SK_ID_CURR'], inplace = True)

    train.replace([np.inf, -np.inf], np.nan, inplace=True)
    train.dropna(axis=1, how='all', inplace=True)

    # Feature names
    features = list(train.columns)

    # Median imputation of missing values
    imputer = SimpleImputer(missing_values=np.nan,
                            strategy='median')

    # Scale each feature to 0-1
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit on the training data
    imputer.fit(train)

    # Transform both training and testing data
    train = imputer.transform(train)

    # Repeat with the scaler
    scaler.fit(train)
    train = scaler.transform(train)

    # Recreate a data frame from the training dataset
    df_final = pd.DataFrame(train, columns=features)

    df_final['SK_ID_CURR'] = train_ids
    df_final['TARGET'] = train_labels
    df_final.to_csv(params.PATH_OUTPUT + 'new_trainScaled_application.csv',
                    index=False)

    # train = df_final.drop(columns=['SK_ID_CURR'])

    # train_labels = df_final_sample['TARGET'].astype(int)
    train = np.asarray(train)

    # Création d'un jeu de donnée X_train, X_valid
    X_train, X_valid, y_train, y_valid = train_test_split(train,
                                                    train_labels,
                                                    test_size=0.3,
                                                    random_state=42)

    smote_df = SMOTETomek(sampling_strategy="auto", random_state=42)
    X_smoted, y_smoted = smote_df.fit_resample(X_train, y_train)

    feature_names = features
    forest = RandomForestClassifier(random_state=0)
    forest.fit(X_smoted, y_smoted)

    forest_importances = pd.Series(
        forest.feature_importances_,
        index=feature_names).sort_values(ascending=False)

    pd.DataFrame(forest_importances,
                 columns=['importance'
                          ]).reset_index().rename(columns={'index': 'feature'}).to_csv(params.PATH_OUTPUT + 'feature_selection_application.csv', index=False)

    return df_final, list(forest_importances[:10].index), train_labels

def LinearRegression(X_train, y_train, X_test, y_test):
    ############ Linear Regression #############
    # print('LinearRegression')
    # Define model
    lr = LogisticRegression(random_state=42)

    # Fit model
    log_reg_tuned = lr.fit(X_train, y_train)

    # Create prediction and get score
    test_score = fbeta_score(y_test, log_reg_tuned.predict(X_test), beta=2)
    train_score = fbeta_score(y_train, log_reg_tuned.predict(X_train), beta=2)

    # # Serialize model
    pickle.dump(log_reg_tuned, open(params.PATH_MODEL + 'LRModel1.pkl', 'wb'))

    return {
        'model': 'Logistic Regression standard',
        'score': 'F-Beta',
        'train_score': train_score,
        'test_score': test_score
    }

def SmotedLinearRegression(X_train, y_train, X_test, y_test, smoteFunction='SMOTE', f_beta=None):
    # ############ LinearRegression with SMOTE #############
    # print('LinearRegression with SMOTE')
    # Define model
    lr_smote = LogisticRegression(random_state=42)

    # Define pipeline
    if smoteFunction=='SMOTETOMEK':
        smote_df = SMOTETomek(sampling_strategy="auto", random_state=42)
        steps = [('smote_df', smote_df), ('model', lr_smote)]
    elif smoteFunction=='SMOTEENN':
        smote_enn = SMOTEENN(sampling_strategy="auto", random_state = 42)
        steps = [('smote_enn', smote_enn), ('model', lr_smote)]
    else:
        over = SMOTE(sampling_strategy=0.5)
        under = RandomUnderSampler(sampling_strategy=1)
        steps = [('over', over), ('under', under), ('model', lr_smote)]

    pipeline = Pipeline(steps=steps)

    # # Parameter grid
    parameters = {'model__penalty': ['l1', 'l2'], 'model__C': [10, 1, 0.1]}

    # # Hyperparameter Tuning
    log_reg_search = GridSearchCV(estimator=pipeline,
                                  param_grid=parameters,
                                  cv=4,
                                  verbose=0,
                                  n_jobs=-1,
                                  scoring=f_beta)

    # # Fit model
    log_reg_tuned = log_reg_search.fit(X_train, y_train)

    # Create prediction and get score
    test_score = fbeta_score(y_test, log_reg_tuned.predict(X_test), beta=2)
    train_score = fbeta_score(y_train, log_reg_tuned.predict(X_train), beta=2)

    # Serialize model
    pickle.dump(log_reg_tuned,
                open(params.PATH_MODEL + 'LR_{}.pkl'.format(smoteFunction), 'wb'))

    return {
        'model': 'Logistic Regression {}'.format(smoteFunction),
        'score': 'F-Beta',
        'train_score': train_score,
        'test_score': test_score
    }

# def SMOTETOMEKLinearRegression(X_train, y_train, X_test, y_test, f_beta):
#     ############ LinearRegression with SMOTETOMEK #############
#     print('LinearRegression with SMOTETOMEK')

#     # Define model
#     lr_smote = LogisticRegression(random_state=42)

#     # Define pipeline
#     smote_df = SMOTETomek(sampling_strategy="auto", random_state=42)
#     steps = [('smote_df', smote_df), ('model', lr_smote)]
#     pipeline = Pipeline(steps=steps)

#     # Parameter grid
#     parameters = {'model__penalty': ['l1', 'l2'], 'model__C': [10, 1, 0.1]}

#     # Hyperparameter Tuning
#     log_reg_search = GridSearchCV(estimator=pipeline,
#                                   param_grid=parameters,
#                                   cv=4,
#                                   verbose=0,
#                                   n_jobs=-1,
#                                   scoring=f_beta)

#     # Fit model
#     log_reg_tuned = log_reg_search.fit(X_train, y_train)

#     # Create prediction and get score
#     test_score = fbeta_score(y_test, log_reg_tuned.predict(X_test), beta=2)
#     train_score = fbeta_score(y_train, log_reg_tuned.predict(X_train), beta=2)

#     # # Serialize model
#     pickle.dump(log_reg_tuned, open(params.PATH_MODEL + 'LRModel3.pkl', 'wb'))

#     return {
#         'model': 'Logistic Regression standard',
#         'score': 'F-Beta',
#         'train_score': train_score,
#         'test_score': test_score
#     }

# def SMOTEENNLinearRegression(X_train, y_train, X_test, y_test, f_beta):
#     ############ LinearRegression with SMOTEENN #############
#     print('LinearRegression with SMOTEENN')
#     # Define model
#     model = LogisticRegression(random_state = 42)

#     # Define pipeline
#     smote_enn = SMOTEENN(sampling_strategy="auto", random_state = 42)
#     steps = [('smote_enn', smote_enn), ('model', model)]
#     pipeline = Pipeline(steps=steps)

#     # Parameter grid
#     parameters = {'model__penalty': ['l1', 'l2'],
#                 'model__C': [10,1,0.1]
#                 }

#     # Hyperparameter Tuning
#     log_reg_search = GridSearchCV(estimator = pipeline,
#                                 param_grid = parameters,
#                                 cv = 4,
#                                 verbose = 0,
#                                 n_jobs = -1,
#                                 scoring = f_beta
#                                 )

#     # Fit model
#     log_reg_tuned = log_reg_search.fit(X_train, y_train)

#     # Create prediction and get score
#     test_score = fbeta_score(y_test, log_reg_tuned.predict(X_test), beta = 2)
#     train_score = fbeta_score(y_train, log_reg_tuned.predict(X_train), beta = 2)

#     # # Serialize model
#     pickle.dump(log_reg_tuned, open(params.PATH_MODEL + 'LRModel4.pkl', 'wb'))

#     return {
#         'model': 'Logistic Regression standard',
#         'score': 'F-Beta',
#         'train_score': train_score,
#         'test_score': test_score
#     }

def SmotedXGB(X_train, y_train, X_test, y_test, smoteFunction='SMOTE', f_beta=None):
    ############ XGB SMOTE #############
    # print('XGB SMOTE')
    # Define model
    xgb = XGBClassifier(objective='binary:logistic', random_state=42)

    # Define pipeline
    if smoteFunction == 'SMOTETOMEK':
        smote_df = SMOTETomek(sampling_strategy="auto", random_state=42)
        steps = [('smote_df', smote_df), ('model', xgb)]
    elif smoteFunction == 'SMOTEENN':
        smote_enn = SMOTEENN(sampling_strategy="auto", random_state=42)
        steps = [('smote_enn', smote_enn), ('model', xgb)]
    else:
        over = SMOTE(sampling_strategy=0.5)
        under = RandomUnderSampler(sampling_strategy=1)
        steps = [('over', over), ('under', under), ('model', xgb)]

    pipeline = Pipeline(steps=steps)

    # Parameter grid
    parameters = {
        'model__max_depth': [4, 5, 6],
        'model__min_child_weight': [1, 3]
    }

    # Hyperparameter Tuning
    xgb_search = GridSearchCV(estimator=pipeline,
                              param_grid=parameters,
                              cv=4,
                              verbose=0,
                              n_jobs=-1,
                              scoring=f_beta)

    #Variable 1
    xgbtuned = xgb_search.fit(X_train, y_train)

    # Create prediction and get score
    test_score = fbeta_score(y_test, xgbtuned.predict(X_test), beta=2)
    train_score = fbeta_score(y_train, xgbtuned.predict(X_train), beta=2)

    # Serialize model
    pickle.dump(
        xgbtuned,
        open(params.PATH_MODEL + 'xgb_{}.pkl'.format(smoteFunction), 'wb'))

    return {
        'model': 'xgb {}'.format(smoteFunction),
        'score': 'F-Beta',
        'train_score': train_score,
        'test_score': test_score
    }

    # ############ XGB SMOTETOMEK #############
    # print('XGB SMOTETOMEK')
    # # Define model
    # xgb = XGBClassifier(objective='binary:logistic', random_state = 42)

    # # Define pipeline
    # # over = SMOTE(sampling_strategy=0.3)
    # # under = RandomUnderSampler(sampling_strategy=1)
    # # steps = [('over', over), ('under', under), ('model', xgb)]
    # # X_smoted, y_smoted = smote_df.fit_resample(X_train, y_train)
    # smote_df = SMOTETomek(sampling_strategy="auto", random_state=42)
    # steps = [('smote_df', smote_df), ('model', xgb)]
    # pipeline = Pipeline(steps=steps)

    # # Parameter grid
    # parameters = {'model__max_depth': [4,5,6],
    #             'model__min_child_weight': [1,3]
    #             }

    # # Hyperparameter Tuning
    # xgb_search = GridSearchCV(estimator = pipeline,
    #                             param_grid = parameters,
    #                             cv = 4,
    #                             verbose = 0,
    #                             n_jobs = -1,
    #                             scoring = f_beta
    #                             )

    # #Variable 1
    # xgbtuned = xgb_search.fit(X_train, y_train)

    # # Create prediction and get score
    # test_score = fbeta_score(y_test, xgbtuned.predict(X_test), beta = 2)
    # train_score = fbeta_score(y_train, xgbtuned.predict(X_train), beta = 2)

    # # Append results to table
    # results = results.append({'model': 'XG Boost SMOTETOMEK',
    #                         'score': 'F-Beta',
    #                         'train_score': train_score,
    #                         'test_score': test_score
    #                         }, ignore_index = True)

    # # Serialize model
    # pickle.dump(log_reg_tuned, open(path + 'models/XGBModel2.obj', 'wb'))

def SmoteLGBM(X_train, y_train, X_test, y_test, smoteFunction='SMOTE', f_beta=None):
    ############ LGBM SMOTE #############
    # print('LGBM SMOTE')
    # Define model
    lgbm = LGBMClassifier(objective = 'binary', random_state = 42)

    # Define pipeline
    over = SMOTE(sampling_strategy=0.5)
    under = RandomUnderSampler(sampling_strategy=1)
    steps = [('over', over), ('under', under), ('model', lgbm)]
    pipeline = Pipeline(steps=steps)

    # Parameter grid
    parameters = {'model__is_unbalance': [True, False],
                'model__boosting_type': ['gbdt', 'goss']
                }

    # Hyperparameter Tuning
    lgbm_search = GridSearchCV(estimator = pipeline,
                                param_grid = parameters,
                                cv = 4,
                                verbose = 0,
                                n_jobs = -1,
                                scoring = f_beta
                                )

    # Fit model
    lgbm_tuned = lgbm_search.fit(X_train, y_train)

    # Create prediction and get score
    test_score = fbeta_score(y_test, lgbm_tuned.predict(X_test), beta = 2)
    train_score = fbeta_score(y_train, lgbm_tuned.predict(X_train), beta = 2)

    # Serialize model
    pickle.dump(
        lgbm_tuned,
        open(params.PATH_MODEL + 'lgbm_{}.pkl'.format(smoteFunction), 'wb'))

    return {
        'model': 'LGBM {}'.format(smoteFunction),
        'score': 'F-Beta',
        'train_score': train_score,
        'test_score': test_score
    }

def rfc(X_train, y_train, X_test, y_test, smoteFunction='SMOTE', f_beta=None):
    ############ Ramdom Forest #############
    print('Ramdom Forest')
    # Define model
    rfc = RandomForestClassifier(random_state = 42)

    # Define pipeline
    over = SMOTE(sampling_strategy=0.5)
    under = RandomUnderSampler(sampling_strategy=1)
    steps = [('over', over), ('under', under), ('model', rfc)]
    pipeline = Pipeline(steps=steps)

    # Parameter grid
    parameters = {'model__min_samples_split': [2,4]
                }

    # Hyperparameter Tuning
    rfc_search = GridSearchCV(estimator = pipeline,
                                param_grid = parameters,
                                cv = 4,
                                verbose = 0,
                                #n_jobs =-1,
                                scoring = f_beta
                                )

    #Variable 1
    rfc_tuned = rfc_search.fit(X_train, y_train)

    # Create prediction and get score
    test_score = fbeta_score(y_test, rfc_tuned.predict(X_test), beta = 2)
    train_score = fbeta_score(y_train, rfc_tuned.predict(X_train), beta = 2)

    # Serialize model
    pickle.dump(
        rfc_tuned,
        open(params.PATH_MODEL + 'rfc_{}.pkl'.format(smoteFunction), 'wb'))

    return {
        'model': 'rfc {}'.format(smoteFunction),
        'score': 'F-Beta',
        'train_score': train_score,
        'test_score': test_score
    }

def ModelSelection(df, top_feat, train_labels):
    train = df.drop(columns=['SK_ID_CURR'])[top_feat[:10]]
    # train_labels = df['TARGET'].astype(int)
    # train = np.asarray(train)

    # Création d'un jeu de donnée X_train, X_valid
    X_train, X_valid, y_train, y_valid = train_test_split(train,
                                                          train_labels,
                                                          test_size=0.3,
                                                          random_state=42)

    # Define a dataframe to store the results of our different models
    results = pd.DataFrame(
        columns=['model', 'score', 'train_score', 'test_score'])

    # Create F-Beta scorer
    f_beta = make_scorer(fbeta_score, beta=2)

    # Define the two scoring techniques for our models
    scoring = {'F-Beta': f_beta, 'ROC AUC': 'roc_auc'}

    with utils.timer("Linear Regression"):
        results = results.append(LinearRegression(X_train, y_train, X_valid,
                                                  y_valid),
                                 ignore_index=True)

    with utils.timer("Linear Regression"):
        results = results.append(SmotedLinearRegression(
            X_train, y_train, X_valid, y_valid, 'SMOTE', f_beta),
                                 ignore_index=True)

    with utils.timer("Linear Regression"):
        results = results.append(SmotedLinearRegression(
            X_train, y_train, X_valid, y_valid, 'SMOTETOMEK', f_beta),
                                 ignore_index=True)

    with utils.timer("Linear Regression"):
        results = results.append(SmotedLinearRegression(
            X_train, y_train, X_valid, y_valid, 'SMOTEENN', f_beta),
                                 ignore_index=True)

    with utils.timer("XGBSMOTE"):
        results = results.append(SmotedXGB(X_train, y_train, X_valid, y_valid,
                                           'SMOTE', f_beta),
                                 ignore_index=True)

    with utils.timer("SmoteLGBM"):
        results = results.append(SmoteLGBM(X_train, y_train, X_valid, y_valid,
                                           'SMOTE', f_beta),
                                 ignore_index=True)

    with utils.timer("rfc"):
        results = results.append(rfc(X_train, y_train, X_valid, y_valid,
                                     'SMOTE', f_beta),
                                 ignore_index=True)

    results.to_csv(params.PATH_OUTPUT + 'perf_model.csv', index=False)

    return None

def main():
    with utils.timer("open new_application file"):
        df = loadData()

    with utils.timer("Process feature importance"):
        df_scaled, top_feat_import, train_labels = PreprocessData(df)

    with utils.timer("Model Selection"):
        ModelSelection(df_scaled, top_feat_import, train_labels)

if __name__ == "__main__":
    with utils.timer("Full model run"):
        main()
