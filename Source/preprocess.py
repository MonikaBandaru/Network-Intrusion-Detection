import pandas as pd
import joblib

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

import os

os.makedirs("Models",exist_ok=True)


columns = [
'duration','protocol_type','service','flag','src_bytes','dst_bytes',
'land','wrong_fragment','urgent','hot','num_failed_logins',
'logged_in','num_compromised','root_shell','su_attempted','num_root',
'num_file_creations','num_shells','num_access_files',
'num_outbound_cmds','is_host_login','is_guest_login','count',
'srv_count','serror_rate','srv_serror_rate','rerror_rate',
'srv_rerror_rate','same_srv_rate','diff_srv_rate',
'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
'dst_host_same_srv_rate','dst_host_diff_srv_rate',
'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
'dst_host_serror_rate','dst_host_srv_serror_rate',
'dst_host_rerror_rate','dst_host_srv_rerror_rate',
'label','difficulty'
]

train = pd.read_csv(
    "Datasets/KDDTrain+.txt",
    names=columns
)

test = pd.read_csv(
    "Datasets/KDDTest+.txt",
    names=columns
)

train['label'] = train['label'].apply(
    lambda x: "normal" if x=="normal" else "attack"
)

test['label'] = test['label'].apply(
    lambda x: "normal" if x=="normal" else "attack"
)

X_train = train.drop(['label','difficulty'], axis=1)
X_test = test.drop(['label','difficulty'], axis=1)

y_train = train['label']
y_test = test['label']

categorical = [
'protocol_type',
'service',
'flag'
]

encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

train_cat = encoder.fit_transform(
    X_train[categorical]
)

test_cat = encoder.transform(
    X_test[categorical]
)

train_cat = pd.DataFrame(
    train_cat,
    columns=encoder.get_feature_names_out(categorical)
)

test_cat = pd.DataFrame(
    test_cat,
    columns=encoder.get_feature_names_out(categorical)
)

X_train = X_train.drop(categorical, axis=1).reset_index(drop=True)
X_test = X_test.drop(categorical, axis=1).reset_index(drop=True)

X_train = pd.concat([X_train, train_cat], axis=1)
X_test = pd.concat([X_test, test_cat], axis=1)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(
    scaler,
    "Models/scaler.pkl"
)

joblib.dump(
    encoder,
    "Models/encoder.pkl"
)

joblib.dump(
    train_cat.columns.tolist(),
    "Models/feature_columns.pkl"
)

joblib.dump(
    (X_train,y_train,X_test,y_test),
    "Models/preprocessed.pkl"
)

print("Preprocessing Completed")