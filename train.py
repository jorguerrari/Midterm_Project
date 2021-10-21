#!/usr/bin/env python
# coding: utf-8
import pickle

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score



# parameters

n_estimators = 150
max_depth = 15

output_file = 'model.bin'


# data preparation

df = pd.read_csv('Employee.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')



from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.leaveornot.values
y_val = df_val.leaveornot.values
y_test = df_test.leaveornot.values

del df_train['leaveornot']
del df_val['leaveornot']
del df_test['leaveornot']

df_full_train = df_full_train.reset_index(drop=True)
y_full_train = (df_full_train.leaveornot).values
del df_full_train['leaveornot']

dicts_full_train = df_full_train.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)

dicts_test = df_test.to_dict(orient='records')
X_test = dv.transform(dicts_test)

rf = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=1)
rf.fit(X_full_train, y_full_train)


# Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, rf), f_out)

print(f'the model is saved to {output_file}')
