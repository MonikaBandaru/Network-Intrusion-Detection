import pandas as pd
import joblib

# Load saved model and preprocessing objects
model = joblib.load("models/model.pkl")
encoder = joblib.load("models/encoder.pkl")
scaler = joblib.load("models/scaler.pkl")

#new network connection
sample = {
    "duration": 0,
    "protocol_type": "tcp",
    "service": "http",
    "flag": "S0",
    "src_bytes": 0,
    "dst_bytes": 0,
    "land": 0,
    "wrong_fragment": 0,
    "urgent": 0,
    "hot": 0,
    "num_failed_logins": 0,
    "logged_in": 0,
    "num_compromised": 0,
    "root_shell": 0,
    "su_attempted": 0,
    "num_root": 0,
    "num_file_creations": 0,
    "num_shells": 0,
    "num_access_files": 0,
    "num_outbound_cmds": 0,
    "is_host_login": 0,
    "is_guest_login": 0,
    "count": 511,
    "srv_count": 511,
    "serror_rate": 1.0,
    "srv_serror_rate": 1.0,
    "rerror_rate": 0.0,
    "srv_rerror_rate": 0.0,
    "same_srv_rate": 1.0,
    "diff_srv_rate": 0.0,
    "srv_diff_host_rate": 0.0,
    "dst_host_count": 255,
    "dst_host_srv_count": 255,
    "dst_host_same_srv_rate": 1.0,
    "dst_host_diff_srv_rate": 0.0,
    "dst_host_same_src_port_rate": 1.0,
    "dst_host_srv_diff_host_rate": 0.0,
    "dst_host_serror_rate": 1.0,
    "dst_host_srv_serror_rate": 1.0,
    "dst_host_rerror_rate": 0.0,
    "dst_host_srv_rerror_rate": 0.0
}
    
valid_protocols = ["tcp", "udp", "icmp"]

if sample["protocol_type"] not in valid_protocols:
    print("❌ Invalid protocol type!")
    exit()

# Convert to DataFrame
X = pd.DataFrame([sample])

# Encode categorical features
categorical = ["protocol_type", "service", "flag"]

encoded = encoder.transform(X[categorical])

encoded = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(categorical)
)

# Remove original categorical columns
X = X.drop(columns=categorical).reset_index(drop=True)

# Combine numerical and encoded features
X = pd.concat([X, encoded], axis=1)

# Scale features
X = scaler.transform(X)

# Predict
prediction = model.predict(X)[0]

if prediction == "attack":
    print("\nPrediction : ATTACK")
    print("⚠️ Warning: Intrusion detected! This network connection appears malicious.")
    print("Please investigate the traffic and take appropriate security measures.")
else:
    print("\nPrediction : NORMAL")
    print("✅ This network connection appears to be legitimate.")