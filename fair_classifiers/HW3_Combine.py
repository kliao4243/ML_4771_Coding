import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import tensorflow as tf

from sklearn.preprocessing   import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics  import accuracy_score
from sklearn.tree     import DecisionTreeClassifier

plt.rc('font', size=14)
sns.set(style='white')

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# 1. Data Preparation

os.chdir('/Users/kunjian/Documents/ml_hw3')

data_dev = pd.read_csv('train.csv', header=0)
data_tst = pd.read_csv('test_no_income.csv', header=0)

data_dev['Id'] = data_dev.index
data_tst['Id'] = data_tst.index
data_dev.rename({'native-country': 'org', 'education-num': 'edu', 'capital gain': 'gain', 'capital loss': 'loss',
                 'marital-status': 'mar', 'hours per week': 'hrs'}, axis=1, inplace=True)
data_tst.rename({'native-country': 'org', 'education-num': 'edu', 'capital gain': 'gain', 'capital loss': 'loss',
                 'marital-status': 'mar', 'hours per week': 'hrs'}, axis=1, inplace=True)


allvars = data_dev.dtypes
numvars = list(allvars.index[allvars.values != 'object'])
catvars = list(allvars.index[allvars.values == 'object'])
numvars.remove('Id')
print(numvars)
print(catvars)

def univariate_num_var(df, vars):
    res = pd.DataFrame()
    for var in vars:
        sum_stat = df[var].describe().transpose().round(3)
        sum_stat["variable"] = var
        res = res.append(sum_stat, ignore_index=True)
    return res[['variable', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]

df_class_0 = data_dev[data_dev['gender'] == 0]
df_class_1 = data_dev[data_dev['gender'] == 1]

univariate_num_var(df_class_0[numvars], numvars)

univariate_num_var(df_class_1[numvars], numvars)

# 2. Decision Tree

def label_encode(df):
    le = LabelEncoder()
    for v in df.columns:
        if df[v].dtype == object:
            df[v] = le.fit_transform(df[v]).astype(object)
    return df


def one_hot_encode(df, vars):
    dummies = pd.get_dummies(df[vars], prefix_sep='')
    return pd.concat([df.drop(vars, axis=1), dummies], axis=1)


def train_decision_tree(x_trn, x_tst, y_trn, y_tst, dp, lf, weight):
    x_trn, x_tst = x_trn.copy(), x_tst.copy()
    y_trn, y_tst = y_trn.copy(), y_tst.copy()

    print('Max_depth: ', dp)
    print('Min_leafs: ', lf)

    # Train decision tree
    dt = DecisionTreeClassifier(criterion='gini', random_state=123, max_depth=dp, min_samples_leaf=lf)
    dt.fit(x_trn.drop([weight], axis=1), y_trn, sample_weight=x_trn[weight])

    # Measure performance
    acu_trn = accuracy_score(y_trn, dt.predict(x_trn.drop([weight], axis=1)), sample_weight=x_trn[weight])
    acu_tst = accuracy_score(y_tst, dt.predict(x_tst.drop([weight], axis=1)), sample_weight=x_tst[weight])
    print('Accuracy (trn): {0:.3f}'.format(acu_trn))
    print('Accuracy (val): {0:.3f}'.format(acu_tst))

    # Calculate Final Score
    x_trn['y_hat'] = dt.predict(x_trn.drop([weight], axis=1))
    x_tst['y_hat'] = dt.predict(x_tst.drop([weight], axis=1))
    x_tst['y_wgt'] = x_tst['y_hat'] * x_tst['fnlwgt']
    pct = x_tst.loc[:, 'y_wgt'].sum() / x_tst.loc[:, 'fnlwgt'].sum()
    print('Predicted Percent: {0:.3f}'.format(pct))
    print()

    return dt, pct, pd.concat([x_trn, y_trn], axis=1), pd.concat([x_tst, y_tst], axis=1)


# 2.1. Split the Sample by Gender

df_m = data_dev[data_dev['gender'] == 0].copy()

df_y, df_x = df_m.loc[:, 'income'], df_m.drop(['income', 'Id'], axis=1)
df_x = one_hot_encode(df_x, catvars)
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.20, random_state=123)

# 2.2 Train Gender = 0 Model

depths = [3, 4, 5, 6, 7, 8, 9, 10]
leafs  = [20, 30, 40, 50]
for depth in depths:
    for leaf in leafs:
        train_decision_tree(X_train, X_test, y_train, y_test, depth, leaf, 'fnlwgt')

tree_m, pct_m, train_m, test_m = train_decision_tree(X_train, X_test, y_train, y_test, 10, 40, 'fnlwgt')

# 2.3. Split the Sample by Gender

df_f = data_dev[data_dev['gender'] == 1].copy()

df_y, df_x = df_f.loc[:, 'income'], df_f.drop(['income', 'Id'], axis=1)
df_x = one_hot_encode(df_x, catvars)
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.20, random_state=123)

# 2.4 Train Gender = 1 Model

depths = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
leafs  = [20, 30, 40, 50]
for depth in depths:
    for leaf in leafs:
        train_decision_tree(X_train, X_test, y_train, y_test, depth, leaf, 'fnlwgt')

tree_f, pct_f, train_f, test_f = train_decision_tree(X_train, X_test, y_train, y_test, 7, 20, 'fnlwgt')

# 2.5 Comparison

bias = 0
if abs(pct_m - pct_f) > 0.1:
    bias = 7 ** (abs(pct_m - pct_f) - 0.1) - 1

print('Percent (G = 0): {0:.3f}'.format(pct_m))
print('Percent (G = 1): {0:.3f}'.format(pct_f))
print('Bias: {0:.3f}'.format(bias))

df_scored = pd.concat([test_m[['y_hat', 'income', 'fnlwgt']], test_f[['y_hat', 'income', 'fnlwgt']]], axis=0)
accuracy = accuracy_score(df_scored['income'], df_scored['y_hat'], sample_weight=df_scored['fnlwgt'])
score = accuracy - bias
print('Accuracy: {0:.3f}'.format(accuracy))
print('Final Score: {0:.3f}'.format(score))

# 3. Gradient Boosting


def train_gradient_boosting(x_trn, x_tst, y_trn, y_tst, lr, ne, weight):

    x_trn, x_tst = x_trn.copy(), x_tst.copy()
    y_trn, y_tst = y_trn.copy(), y_tst.copy()

    print('Learning Rate: ', lr)
    print('No Estimators: ', ne)

    # Train Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=ne, learning_rate=lr, max_features=5, random_state=123)
    gb.fit(x_trn.drop([weight], axis=1), y_trn, sample_weight=x_trn[weight])

    # Measure performance
    acu_trn = accuracy_score(y_trn, gb.predict(x_trn.drop([weight], axis=1)), sample_weight=x_trn[weight])
    acu_tst = accuracy_score(y_tst, gb.predict(x_tst.drop([weight], axis=1)), sample_weight=x_tst[weight])
    print('Accuracy (trn): {0:.3f}'.format(acu_trn))
    print('Accuracy (val): {0:.3f}'.format(acu_tst))

    # Calculate Final Score
    x_trn['y_hat'] = gb.predict(x_trn.drop([weight], axis=1))
    x_tst['y_hat'] = gb.predict(x_tst.drop([weight], axis=1))
    x_tst['y_wgt'] = x_tst['y_hat'] * x_tst['fnlwgt']
    pct = x_tst.loc[:, 'y_wgt'].sum() / x_tst.loc[:, 'fnlwgt'].sum()
    print('Predicted Percent: {0:.3f}'.format(pct))
    print()

    return gb, pct, pd.concat([x_trn, y_trn], axis=1), pd.concat([x_tst, y_tst], axis=1)


# 1. Split the Sample by Gender

df_m = data_dev[data_dev['gender'] == 0].copy()

df_y, df_x = df_m.loc[:, 'income'], df_m.drop(['income', 'Id'], axis=1)
df_x = one_hot_encode(df_x, catvars)
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.20, random_state=123)

# 2. Train Gender = 0 Model

learning_rates = [0.01, 0.015, 0.02, 0.05, 0.1, 0.15, 0.2]
no_estimators  = [50, 100, 200, 250, 300, 500]

for l in learning_rates:
    for n in no_estimators:
        train_gradient_boosting(X_train, X_test, y_train, y_test, l, n, 'fnlwgt')

gb_m, pct_m, train_m, test_m = train_gradient_boosting(X_train, X_test, y_train, y_test, 0.02, 200, 'fnlwgt')


# 1. Split the Sample by Gender

df_m = data_dev[data_dev['gender'] == 1].copy()

df_y, df_x = df_m.loc[:, 'income'], df_m.drop(['income', 'Id'], axis=1)
df_x = one_hot_encode(df_x, catvars)
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.20, random_state=123)

# 2. Train Gender = 0 Model

learning_rates = [0.01, 0.05, 0.1, 0.15, 0.5, 0.8, 1]
no_estimators  = [300, 500, 700, 800, 1000]

for l in learning_rates:
    for n in no_estimators:
        train_gradient_boosting(X_train, X_test, y_train, y_test, l, n, 'fnlwgt')

gb_f, pct_f, train_f, test_f = train_gradient_boosting(X_train, X_test, y_train, y_test, 0.8, 800, 'fnlwgt')


bias = 0
if abs(pct_m - pct_f) > 0.1:
    bias = 7 ** (abs(pct_m - pct_f) - 0.1) - 1

print('Predicted Percent (G = 0): {0:.3f}'.format(pct_m))
print('Predicted Percent (G = 1): {0:.3f}'.format(pct_f))
print('Bias: {0:.3f}'.format(bias))

df_scored = pd.concat([test_m[['y_hat', 'income', 'fnlwgt']], test_f[['y_hat', 'income', 'fnlwgt']]], axis=0)
accuracy = accuracy_score(df_scored['income'], df_scored['y_hat'], sample_weight=df_scored['fnlwgt'])
score = accuracy - bias
print('Accuracy: {0:.3f}'.format(accuracy))
print('Final Score: {0:.3f}'.format(score))

# 4. Random Forest

def train_random_forest(x_trn, x_tst, y_trn, y_tst, no, dp, weight):

    x_trn, x_tst = x_trn.copy(), x_tst.copy()
    y_trn, y_tst = y_trn.copy(), y_tst.copy()

    print('Max_depth: ', dp)
    print('No Estimators: ', no)

    # Train decision tree
    ft = RandomForestClassifier(n_estimators=no, random_state=123, max_depth=dp, n_jobs=-1)
    ft.fit(x_trn.drop([weight], axis=1), y_trn, sample_weight=x_trn[weight])

    # Measure performance
    acu_trn = accuracy_score(y_trn, ft.predict(x_trn.drop([weight], axis=1)), sample_weight=x_trn[weight])
    acu_tst = accuracy_score(y_tst, ft.predict(x_tst.drop([weight], axis=1)), sample_weight=x_tst[weight])
    print('Accuracy (trn): {0:.3f}'.format(acu_trn))
    print('Accuracy (val): {0:.3f}'.format(acu_tst))

    # Calculate Final Score
    x_trn['y_hat'] = ft.predict(x_trn.drop([weight], axis=1))
    x_tst['y_hat'] = ft.predict(x_tst.drop([weight], axis=1))
    x_tst['y_wgt'] = x_tst['y_hat'] * x_tst['fnlwgt']
    pct = x_tst.loc[:, 'y_wgt'].sum() / x_tst.loc[:, 'fnlwgt'].sum()
    print('Predicted Percent: {0:.3f}'.format(pct))
    print()

    return ft, pct, pd.concat([x_trn, y_trn], axis=1), pd.concat([x_tst, y_tst], axis=1)


# 1. Split the Sample by Gender

df_m = data_dev[data_dev['gender'] == 0].copy()

df_y, df_x = df_m.loc[:, 'income'], df_m.drop(['income', 'Id'], axis=1)
df_x = one_hot_encode(df_x, catvars)
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.20, random_state=123)

# 2. Train Gender = 0 Model

depths = [3, 4, 5, 6, 7, 8, 9, 10]
no_est = [20, 30, 40, 50]
for depth in depths:
    for no in no_est:
        train_random_forest(X_train, X_test, y_train, y_test, no, depth, 'fnlwgt')

forest_m, pct_m, train_m, test_m = train_random_forest(X_train, X_test, y_train, y_test, 50, 4, 'fnlwgt')

# 1. Split the Sample by Gender

df_m = data_dev[data_dev['gender'] == 1].copy()

df_y, df_x = df_m.loc[:, 'income'], df_m.drop(['income', 'Id'], axis=1)
df_x = one_hot_encode(df_x, catvars)
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.20, random_state=123)

# 2. Train Gender = 0 Model

depths = [5, 10, 15, 18, 20, 30, 50, 60]
no_est = [20, 30, 40, 50, 80, 100, 200, 300]
for depth in depths:
    for no in no_est:
        train_random_forest(X_train, X_test, y_train, y_test, no, depth, 'fnlwgt')

forest_f, pct_f, train_f, test_f = train_random_forest(X_train, X_test, y_train, y_test, 40, 30, 'fnlwgt')

bias = 0
if abs(pct_m - pct_f) > 0.1:
    bias = 7 ** (abs(pct_m - pct_f) - 0.1) - 1

print('Predicted Percent (G = 0): {0:.3f}'.format(pct_m))
print('Predicted Percent (G = 1): {0:.3f}'.format(pct_f))
print('Bias: {0:.3f}'.format(bias))

df_scored = pd.concat([test_m[['y_hat', 'income', 'fnlwgt']], test_f[['y_hat', 'income', 'fnlwgt']]], axis=0)
accuracy = accuracy_score(df_scored['income'], df_scored['y_hat'], sample_weight=df_scored['fnlwgt'])
score = accuracy - bias
print('Accuracy: {0:.3f}'.format(accuracy))
print('Final Score: {0:.3f}'.format(score))




