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

# 5.Neural Network

def label_encode(df):
    le = LabelEncoder()
    for v in df.columns:
        if df[v].dtype == object:
            df[v] = le.fit_transform(df[v]).astype(object)
    return df

def one_hot_encode(df, vars):
    dummies = pd.get_dummies(df[vars], prefix_sep='')
    return pd.concat([df.drop(vars, axis=1), dummies], axis=1)


def train_neural_network(x_trn, x_tst, y_trn, y_tst, h1, h2, weight):

    x_trn, x_tst = x_trn.copy(), x_tst.copy()
    y_trn, y_tst = y_trn.copy(), y_tst.copy()

    print('Hidden Layer 1: ', h1)
    print('Hidden Layer 2: ', h2)

    # Train Neural Network
    feat_columns = [tf.feature_column.numeric_column(k) for k in x_trn.drop([weight], axis=1).columns]
    x_trn_t = x_trn.drop([weight], axis=1).to_dict('series')
    x_tst_t = x_tst.drop([weight], axis=1).to_dict('series')
    dnn = tf.contrib.learn.DNNClassifier(feature_columns=feat_columns,
                                         hidden_units=[h1, h2],
                                         n_classes=2,
                                         activation_fn=tf.nn.sigmoid,
                                         optimizer=tf.train.AdamOptimizer(learning_rate=0.001))
    dnn = tf.contrib.learn.SKCompat(dnn)
    dnn.fit(x=x_trn_t, y=y_trn, batch_size=400, steps=40000)

    # Measure performance
    acu_trn = accuracy_score(y_trn, dnn.predict(x_trn_t)['classes'], sample_weight=x_trn[weight])
    acu_tst = accuracy_score(y_tst, dnn.predict(x_tst_t)['classes'], sample_weight=x_tst[weight])
    print('Accuracy (trn): {0:.3f}'.format(acu_trn))
    print('Accuracy (val): {0:.3f}'.format(acu_tst))

    # Calculate Final Score
    x_trn['y_hat'] = dnn.predict(x_trn_t)['classes']
    x_tst['y_hat'] = dnn.predict(x_tst_t)['classes']
    x_tst['y_wgt'] = x_tst['y_hat'] * x_tst['fnlwgt']
    pct = x_tst.loc[:, 'y_wgt'].sum() / x_tst.loc[:, 'fnlwgt'].sum()
    print('Predicted Percent: {0:.3f}'.format(pct))
    print()

    return dnn, pct, pd.concat([x_trn, y_trn], axis=1), pd.concat([x_tst, y_tst], axis=1)


# 1. Split the Sample by Gender

df_m = data_dev[data_dev['gender'] == 0].copy()

df_y, df_x = df_m.loc[:, 'income'], df_m.drop(['income', 'Id'], axis=1)
df_x = label_encode(df_x)
df_x = one_hot_encode(df_x, catvars)
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.20, random_state=123)

# 2. Train Gender = 0 Model

layers = {'t1': [300, 100], 't2': [400, 200], 't3': [800, 200]}

nn_m, pct_m, train_m, test_m = train_neural_network(X_train, X_test, y_train, y_test, 300, 100, 'fnlwgt')


# 1. Split the Sample by Gender

df_m = data_dev[data_dev['gender'] == 1].copy()

df_y, df_x = df_m.loc[:, 'income'], df_m.drop(['income', 'Id'], axis=1)
df_x = label_encode(df_x)
df_x = one_hot_encode(df_x, catvars)
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.20, random_state=123)

# 2. Train Gender = 1 Model

layers = {'t1': [300, 100], 't2': [400, 200], 't3': [800, 200]}

nn_f, pct_f, train_f, test_f = train_neural_network(X_train, X_test, y_train, y_test, 300, 100, 'fnlwgt')


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
