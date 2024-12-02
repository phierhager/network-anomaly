import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Load the pre-trained Random Forest model
MODEL_PATH = (
    Path(__file__).parent.parent.parent
    / "models/RandomForest_UNSW-NB15_model.pkl"
)
model = joblib.load(MODEL_PATH)

# Feature columns from the dataset
FEATURE_COLUMNS = [
    "dur",
    "proto",
    "service",
    "state",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "is_sm_ips_ports",
]


def classify_packet(packet_features):
    """Classify a packet based on extracted features."""
    # Convert packet features into a DataFrame with the same structure as the model's training set
    feature_df = pd.DataFrame([packet_features], columns=FEATURE_COLUMNS)

    # Predict using the model
    prediction = model.predict(feature_df)
    return "malicious" if prediction[0] == 1 else "normal"
