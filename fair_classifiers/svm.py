import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# 1. Data Preparation

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

def one_hot_encode(df, vars):
    dummies = pd.get_dummies(df[vars], prefix_sep='')
    return pd.concat([df.drop(vars, axis=1), dummies], axis=1)

def train_support_vector_machine(x_trn, x_tst, y_trn, y_tst, kernel, weight):

    x_trn, x_tst = x_trn.copy(), x_tst.copy()
    y_trn, y_tst = y_trn.copy(), y_tst.copy()

    print('Kernel Type: ', kernel)

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(x_trn.drop([weight], axis=1))
    x_dev = scaling.transform(x_trn.drop([weight], axis=1).astype(float))
    x_val = scaling.transform(x_tst.drop([weight], axis=1).astype(float))
    
    # Train decision tree
    dt = SVC(kernel=kernel, C=1, random_state=123, cache_size=10000)
    dt.fit(x_dev, y_trn, sample_weight=x_trn[weight])

    # Measure performance
    acu_trn = accuracy_score(y_trn, dt.predict(x_dev), sample_weight=x_trn[weight])
    acu_tst = accuracy_score(y_tst, dt.predict(x_val), sample_weight=x_tst[weight])
    print('Accuracy (trn): {0:.3f}'.format(acu_trn))
    print('Accuracy (val): {0:.3f}'.format(acu_tst))

    # Calculate Final Score
    x_trn['y_hat'] = dt.predict(x_dev)
    x_tst['y_hat'] = dt.predict(x_val)
    x_tst['y_wgt'] = x_tst['y_hat'] * x_tst['fnlwgt']
    pct = x_tst.loc[:, 'y_wgt'].sum() / x_tst.loc[:, 'fnlwgt'].sum()
    print('Predicted Percent: {0:.3f}'.format(pct))
    print()

    return dt, pct


# 1. Split the Sample by Gender

df_m = data_dev[data_dev['gender'] == 0].copy()

df_y, df_x = df_m.loc[:, 'income'], df_m.drop(['income', 'Id'], axis=1)
df_x = one_hot_encode(df_x, catvars)
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.20, random_state=123)

# 2. Train Gender = 0 Model

kernels = ['linear', 'poly', 'rbf', 'sigmoid']

for k in kernels:
    train_support_vector_machine(X_train, X_test, y_train, y_test, k, 'fnlwgt')



# 1. Split the Sample by Gender

df_m = data_dev[data_dev['gender'] == 1].copy()

df_y, df_x = df_m.loc[:, 'income'], df_m.drop(['income', 'Id'], axis=1)
df_x = one_hot_encode(df_x, catvars)
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.20, random_state=123)

# 2. Train Gender = 1 Model

kernels = ['linear', 'poly', 'rbf', 'sigmoid']

for k in kernels:
    train_support_vector_machine(X_train, X_test, y_train, y_test, k, 'fnlwgt')
