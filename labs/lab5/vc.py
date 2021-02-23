import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
import os
import seaborn as sns
# from IPython.display import display

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import Lasso, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import cross_val_score, ShuffleSplit, train_test_split, GridSearchCV

if __name__ == '__main__':
    X_train = pd.read_csv('./data/X_train.csv')
    y_train = pd.read_csv('./data/y_train.csv')
    X_test = pd.read_csv('./data/X_test.csv')

    # Concatenation
    X = pd.concat([X_train, X_test])
    print(f'X Train Shape: {X_train.shape}')
    print(f'X Test Shape: {X_test.shape}')
    print(f'Concate X Shape: {X.shape}')
    X.head()

    # Standardize
    sc_X = StandardScaler()
    sc_X.fit(X.values)
    X_std = pd.DataFrame(sc_X.fit_transform(X.values), columns=X.columns)
    X_train_std = pd.DataFrame(sc_X.fit_transform(X_train.values), columns=X.columns)
    X_test_std = pd.DataFrame(sc_X.fit_transform(X_test.values), columns=X.columns)

    X_train_std = X_train_std[['SHOT_DIST', 'GAME_CLOCK', 'SHOT_CLOCK']]
    X_test_std = X_test_std[['SHOT_DIST', 'GAME_CLOCK', 'SHOT_CLOCK']]

    parallel_procs = -1

    vc = VotingClassifier(estimators=[
                    ('rf', load('models/forest.joblib')), 
                    ('gnb', load('models/gaussian.joblib')), ('knn', load('models/knn.joblib')), 
                    ('mlp', load('models/mlp.joblib')), ('svc', load('models/svc.joblib'))],
                     voting='soft', n_jobs=parallel_procs)
    cv_score = cross_val_score(vc, X_train_std.values, y_train.values.ravel(), cv=5)
    print(f'Mean {sum(cv_score) / len(cv_score)}')
    print(f'{cv_score}')
    dump(vc, f'models/vc-soft.joblib')

    vc = VotingClassifier(estimators=[
                    ('rc', load('models/ridge.joblib')), ('rf', load('models/forest.joblib')), 
                    ('gnb', load('models/gaussian.joblib')), ('knn', load('models/knn.joblib')), 
                    ('mlp', load('models/mlp.joblib')), ('svc', load('models/svc.joblib'))],
                     voting='hard', n_jobs=parallel_procs)
    cv_score = cross_val_score(vc, X_train_std.values, y_train.values.ravel(), cv=5)
    print(f'Mean {sum(cv_score) / len(cv_score)}')
    print(f'{cv_score}')
    dump(vc, f'models/vc-hard.joblib')

    # vc = clone(vc)
    # vc.fit(X_train_std.values, y_train.values.ravel())
    # print(f'Training Acc')
    # print(f'{accuracy_score(y_train, vc.predict(X_train_std.values))}')
    # print(f'CV 5 folds Acc')