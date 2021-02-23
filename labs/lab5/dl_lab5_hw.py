#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
import os
import seaborn as sns
from IPython.display import display

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import cross_val_score, ShuffleSplit, train_test_split, GridSearchCV


# In[4]:


X_train = pd.read_csv('./data/X_train.csv')
y_train = pd.read_csv('./data/y_train.csv')

X_test = pd.read_csv('./data/X_test.csv')


# In[5]:


print(X_train.shape)
print(X_train.columns)
print(y_train.columns)


# In[6]:


X_train.head()


# In[7]:


# Concatenation
X = pd.concat([X_train, X_test])
print(f'X Train Shape: {X_train.shape}')
print(f'X Test Shape: {X_test.shape}')
print(f'Concate X Shape: {X.shape}')
X.head()


# In[8]:


parallel_procs = -1


# In[9]:


# Standardize
sc_X = StandardScaler()
sc_X.fit(X.values)
X_std = pd.DataFrame(sc_X.fit_transform(X.values), columns=X.columns)
X_train_std = pd.DataFrame(sc_X.fit_transform(X_train.values), columns=X.columns)
X_test_std = pd.DataFrame(sc_X.fit_transform(X_test.values), columns=X.columns)


# In[10]:


# row_num = 3
# col_num = 3
# _, subplot_arr = plt.subplots(col_num, row_num, figsize=(20, 20))

# for idx, x_var in enumerate(X_train.columns):
#     col_idx = idx % col_num
#     row_idx = idx // col_num

#     subplot_arr[row_idx, col_idx].scatter(X_train[x_var], y_train)
#     subplot_arr[row_idx, col_idx].set_xlabel(x_var)
# plt.show()


# # Random Forest Feature Importance
# 
# Sort the feature in descending order of importance. Keep the order in indices_f.

# In[11]:


forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=200,
                                random_state=1,
                                n_jobs=parallel_procs)
forest.fit(X_train_std, y_train.values.ravel())
importances = forest.feature_importances_
indices_f = np.argsort(importances)[::-1]

plt.title('Feature Importances')
plt.bar(range(importances.shape[0]), 
        importances[indices_f], 
        align='center', 
        alpha=0.5)
plt.xticks(range(importances.shape[0]),
           X_train.columns.values[indices_f],
           rotation=90)
plt.tight_layout()
plt.show()

print(f'TOP 3 Fetures: {X_train.columns.values[indices_f[0]]}, {X_train.columns.values[indices_f[1]]}, {X_train.columns.values[indices_f[2]]}')


# # LASSO Feature Selection
# 
# Sort the feature in descending order of cofficients. Keep the order in indices_l.

# In[12]:


# alpha = 10
alpha_arr = np.arange(0.01, 1, 0.05)
feature_num = X.columns.values.shape[0]
coef = np.zeros((alpha_arr.shape[0], feature_num))

for idx in range(alpha_arr.shape[0]):
    lasso = Lasso(alpha=alpha_arr[idx])
    lasso.fit(X_train, y_train)
    coef[idx, :] = lasso.coef_.reshape(1, -1)

for coef_idx in range(lasso.coef_.shape[0]):
    plt.plot(alpha_arr, coef[:, coef_idx], label=X_train.columns.values[coef_idx])

plt.legend()
plt.ylabel('Coefficients')
plt.xlabel('Alpha')
plt.tight_layout()
plt.show()

indices_l = np.argsort(abs(coef[0]))[::-1]
print(f'TOP 3 Fetures: {X_train.columns.values[indices_l[0]]}, {X_train.columns.values[indices_l[1]]}, {X_train.columns.values[indices_l[2]]}')


# # Variance of Features
# 
# Sort the feature in descending order of variance. Keep the order in indices_var.

# In[13]:


var_attributes = np.var(X, axis=0)
indices_var = np.argsort(var_attributes)[::-1]
display(var_attributes[indices_var])

plt.title('Variance of Attibutes')
plt.bar(range(var_attributes.shape[0]), 
        var_attributes[indices_var], 
        align='center', 
        alpha=0.5)
plt.xticks(range(var_attributes.shape[0]),
           X_train.columns.values[indices_var],
           rotation=90)
plt.tight_layout()
plt.show()
print(f'TOP 3 Fetures: {X_train.columns.values[indices_var[0]]}, {X_train.columns.values[indices_var[1]]}, {X_train.columns.values[indices_var[2]]}')


# # Correlation Heat Map of Attirbutes

# In[14]:


# Z-normalize data
# sc = StandardScaler()
# Z = sc.fit_transform(X)
# Estimate the correlation matrix
R = np.dot(X_std.T, X_std) / X_std.shape[0]

sns.set(font_scale=1.0)

ticklabels = [s for s in X.columns]

hm = sns.heatmap(
    R,
    cbar = True,
    square = True,
    yticklabels = ticklabels,
    xticklabels = ticklabels
)

plt.figure(figsize=(10, 8))
plt.tight_layout()
# plt.savefig('./output/fig-wine-corr.png', dpi = 300)
plt.show()

sns.reset_orig()


# # Select Features
# 
# The table shows the result of each feature selection. Number '5' in the column of PERIOD of first row means PERIOD is 6th (important) feature in descending order. Here I sum all the number through the same column in the last row 'sum'. I select the features which have smallest 3 number as output.

# In[15]:


indices_all = pd.DataFrame(index=['indices_f', 'indices_l', 'indices_var'], columns=X.columns)

for idx, idx_pack in enumerate(zip(indices_f, indices_l, indices_var)):
    idx_f, idx_l, idx_var = idx_pack
    indices_all.loc['indices_f'][idx_f] = idx
    indices_all.loc['indices_l'][idx_l] = idx
    indices_all.loc['indices_var'][idx_var] = idx
    
# display(indices_all)
indices_all.loc['indeice_sum'] = indices_all.sum(axis=0)
display(indices_all)

indices_sort = np.argsort(indices_all.loc['indeice_sum'].values)
columns_sel = indices_all.columns.values[indices_sort][:3]
print(f'Selected 3 features: {columns_sel}')

X_train_std = X_train_std[columns_sel]
X_test_std = X_test_std[columns_sel]


# # Split Validation Set
# 
# Here I use module 'gridSearchCV' which has implemented K-Fold algorithm. I use 5 fold cross validation and the scoring is accuracy(correct samples/ validation set).

# In[16]:


# X_train, X_vali, y_train, y_vali = train_test_split(X_train, y_train, test_size=0.1, random_state=0)
# X_train_std = sc_X.fit_transform(X_train)
# X_vali_std = sc_X.fit_transform(X_vali)
# X_test_std = sc_X.fit_transform(X_test)

# print('X_train')
# display(X_train)
# print('X_vali')
# display(X_vali)

# print('X_train_std')
# # display(pd.DataFrame(X_train_std, columns=X_train.columns))
# print('X_vali_std')
# # display(pd.DataFrame(X_vali_std, columns=X_train.columns))

# print('y_train')
# display(y_train)
# print('y_vali')
# display(y_vali)


# # GridSearchCV
# 
# Here are some utilty functions. 'grid_tune' function apply the model and parameters, search for the best paramter set for the model and, show the score of cross validation in 5 folds. 
# 
# 'record' function store the current model and compare with the performance of the best model. 
# 
# 'best_predict' function load the best model and predict the label according to the testing data.

# In[17]:


global_best_score = 0
global_best_model = None
global_best_cv_results = None
global_best_model_name = 'None'

def grid_tune(clf, params, scoring=None, is_show=True):
    gs = GridSearchCV(clf, params, scoring=scoring, cv=5, n_jobs=parallel_procs, return_train_score=True)
    gs.fit(X_train_std.values, y_train.values.ravel())
    
    if(is_show):
        display(pd.DataFrame(gs.cv_results_))
        print(f'Mean Validation Score of Best: {gs.best_score_}')
        print(f'Params of Best: {gs.best_params_}')
#     best_model = gs.best_estimator_
    best_model = clone(gs.best_estimator_)
    best_model.fit(X_train_std.values, y_train.values.ravel())
    
    return best_model, gs.best_score_, gs.best_params_, gs.cv_results_

def record(model, score, cv_results, name):
    print(f'Recording Model: {name}')
    global global_best_score
    global global_best_model
    global global_best_cv_results
    global global_best_model_name
    
    if(global_best_model == None and global_best_score == 0):
        print(f'Init Best Model: {name}, CV: {score}')
        global_best_score = score
        global_best_model = model
        global_best_cv_results = pd.DataFrame(cv_results)
        global_best_model_name = name
        
        dump(model, f'models/{name}.joblib')
        dump(model, f'models/best.joblib')
        pd.DataFrame(cv_results).to_csv('cvs/best-cv.csv')
        pd.DataFrame(cv_results).to_csv(f'cvs/{name}-cv.csv')
        
    else:
        if(global_best_score > score):
            print(f'Keep Best Model: {global_best_model_name}, CV: {global_best_score}')
            dump(model, f'models/{name}.joblib')
            pd.DataFrame(cv_results).to_csv(f'cvs/{name}-cv.csv')
            
        else:
            print(f'Update Best Model: {name}, CV: {score}')
            global_best_score = score
            global_best_model = model
            global_best_cv_results = pd.DataFrame(cv_results)
            global_best_model_name = name
            
            dump(model, f'models/{name}.joblib')
            dump(model, f'models/best.joblib')
            pd.DataFrame(cv_results).to_csv('cvs/best-cv.csv')
            pd.DataFrame(cv_results).to_csv(f'cvs/{name}-cv.csv')

# Pair: (name, model, params)
def model_search(model_param_list):
    for pair in model_param_list:
        model, score, param = grid_tune(pair[1], pair[2])
        record(model, score, pair[0])
        
    best_predict()
        
def best_predict():
    best = load('models/best.joblib')
    print(f'Best Model')
    print(f'{type(best).__name__}')
    print(f'Parameters')
    print(best.get_params())
    print(f'Training Acc')
    print(f'{accuracy_score(y_train, best.predict(X_train_std.values))}')
    print(f'CV 5 folds Acc')
    cv_score = cross_val_score(best, X_train_std.values, y_train.values.ravel(), cv=5)
    print(f'Mean {sum(cv_score) / len(cv_score)}')
    print(f'{cv_score}')
    
    pred_best = pd.DataFrame(best.predict(X_test_std))
    pred_best.to_csv('models/best.csv', index=False)
    
    return pred_best


# # Ridge Classifier

# In[22]:


params = [{
              'alpha': [100000, 50000, 20000, 10000, 5000, 1000, 500, 100, 50, 10, 5, 1, 0.1, 0.001, 1e-9],
              'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
          }]
model, score, param, cv_results = grid_tune(RidgeClassifier(), params)
record(model, score, cv_results, 'ridge')

# search_model_list = [('ridge', RidgeClassifier(), params)]
# model_search(search_model_list)

params[0]['solver'] = [param['solver']]
model, score, param, cv_results = grid_tune(RidgeClassifier(), params, is_show=False)
record(model, score, cv_results, 'ridge')
cv_results = pd.DataFrame(cv_results)

plt.plot(cv_results.loc[:, ['mean_test_score']], color='red', marker='^', label='test')
plt.plot(cv_results.loc[:, ['mean_train_score']], color='blue', marker='o', label='train')
plt.xlabel('L1 Regularlize Coffiicient')
plt.ylabel('mean score of CV 5 fold')
plt.legend()
    
plt.xticks(range(len(params[0]['alpha'])), params[0]['alpha'], rotation=90)
plt.tight_layout()
plt.savefig('fig/ridge-error-curve.png')
plt.show()


# # Random Forest Classifier

# In[ ]:


params = [{
              'n_estimators': [100, 200, 500], 
              'criterion': ['gini', 'entropy'],
              'n_jobs': [parallel_procs]
          }]

model, score, param, cv_results = grid_tune(RandomForestClassifier(), params)
record(model, score, cv_results, 'forest')


# # Gaussian Naive Bayes

# In[56]:


params = [{
              'var_smoothing': [1e-9, 1e-6, 1e-3, 1, 10]
         }]

model, score, param, cv_results = grid_tune(GaussianNB(), params)
record(model, score, cv_results, 'gaussian')


# # KNN

# In[ ]:


params = [{
              'n_neighbors': [5, 100, 200, 500, 1000],
              'weights': ['uniform', 'distance'],
              'n_jobs': [parallel_procs]
         }]

model, score, param, cv_results = grid_tune(KNeighborsClassifier(), params)
record(model, score, cv_results, 'knn')


# # MLPClassifier

# In[ ]:


params = [{
              'hidden_layer_sizes': [(100), (100, 100), (100, 100, 100)], 
              'solver': ['adam'],
              'learning_rate': ['adaptive'],
              'learning_rate_init': [0.001, 0.1, 1]
          }]

model, score, param, cv_results = grid_tune(MLPClassifier(), params)
record(model, score, cv_results, 'mlp')


# # SVC

# In[ ]:


params = [
          {
              'C': [1, 5, 10], 
              'kernel': ['rbf'], 
          },
          {
              'C': [1, 5, 10], 
              'kernel': ['poly'],
              'degree': [3, 5]
          }
]

model, score, param, cv_results = grid_tune(SVC(), params)
record(model, score, cv_results, 'svc')


# In[ ]:


pred_best = best_predict()
# display(pred_best)


# In[ ]:


# depth = 3
# width = 4
# params = [{
#               'hidden_layer_sizes': [(50), (100), (200), (300), (50, 50), (100, 100), (200, 200), (300, 300), (50, 50, 50), (100, 100, 100), (200, 200, 200), (300, 300, 300)], 
#               'solver': ['adam'],
#               'learning_rate': ['adaptive'],
#           }]



# model, score, param = grid_tune(MLPClassifier(), params)
# record(model, score, 'mlp')
# params = [{
#               'alpha': [0.001, 0.01, 0.1, 1, 5, 10, 20, 50, 100, 200, 500, 1000],
# #               'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
#           }]
# model, score, param, cv_results = grid_tune(RidgeClassifier(), params)
# record(model, score, cv_results, 'ridge')

# cv_results = pd.DataFrame(cv_results)
# _, subplot_arr = plt.subplots(1, depth, figsize=(20, 20))

# for d in range(depth):
#     col_idx = d % depth
#     row_idx = d // depth
    
# #     print(type(cv_results))
# #     print(cv_results.loc[0:4, ['mean_test_score']])
# #     print(cv_results.loc[d * width : (d+1) * width, ['mean_test_score']])
# #     print(cv_results[d * width : (d+1) * width]['mean_test_score'])
#     subplot_arr[col_idx].plot(cv_results.loc[d * width : (d+1) * width, ['mean_test_score']], color='red', marker='^', label='test')
#     subplot_arr[col_idx].plot(cv_results.loc[d * width : (d+1) * width, ['mean_train_score']], color='blue', marker='o', label='train')
#     subplot_arr[col_idx].set_xlabel('model complexity')
#     subplot_arr[col_idx].set_ylabel('mean CV score in 5 fold')
#     subplot_arr[col_idx].legend()
    
# #     plt.xticks(range(params[0]['hidden_layer_sizes'].shape[0]),
# #            params[0]['hidden_layer_sizes'],
# #            rotation=90)
# plt.tight_layout()
# plt.show()

