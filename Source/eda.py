import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("Images",exist_ok=True)

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

df = pd.read_csv(
    "Datasets/KDDTrain+.txt",
    names=columns
)

print(df.head())

print("\nShape")
print(df.shape)

print("\nMissing Values")
print(df.isnull().sum())

print("\nClass Distribution")
print(df['label'].value_counts())

plt.figure(figsize=(12,6))
sns.countplot(
    y=df['label'],
    order=df['label'].value_counts().index
)
plt.title("Attack Distribution")
plt.tight_layout()
plt.savefig("Images/Attack Distribution.png")

numeric = df.select_dtypes(include='number')

plt.figure(figsize=(14,10))
sns.heatmap(
    numeric.corr(),
    cmap="coolwarm"
)
plt.title("Correlation Matrix")
plt.savefig("Images/Correlation Map.png")