import sys
import pandas as pd
import joblib

import json

import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)




with open('public_cases.json', 'r') as f:
    data_json = json.load(f)

df = pd.DataFrame(data_json)

# --- Feature Engineering Function ---
def feature_engineering(df):
    df['miles_per_day'] = df['miles'] / df['days'].replace(0, 1)
    df['spending_per_day'] = df['receipts'] / df['days'].replace(0, 1)
    df['sweet_spot'] = ((df['days'] == 5) & 
                        (df['miles_per_day'] >= 180) & (df['miles_per_day'] <= 220) & 
                        (df['spending_per_day'] <= 100)).astype(int)
    df['vacation_penalty'] = ((df['days'] >= 8) & (df['spending_per_day'] > 90)).astype(int)
    return df

# --- Load model ---
model = joblib.load('reimbursement_model.pkl')

# --- Parse command-line arguments ---
if len(sys.argv) != 4:
    print("Usage: python3 calculate_reimbursement.py <days> <miles> <receipts>")
    sys.exit(1)

try:
    days = int(sys.argv[1])
    miles = float(sys.argv[2])
    receipts = float(sys.argv[3])
except ValueError:
    print("Invalid input: all inputs must be numeric")
    sys.exit(1)

# --- Prepare input ---
sample = pd.DataFrame({
    'days': [days],
    'miles': [miles],
    'receipts': [receipts]
})
sample = feature_engineering(sample)
features = ['days', 'miles', 'receipts', 'miles_per_day', 'spending_per_day', 'sweet_spot', 'vacation_penalty']

# --- Predict and print result ---
prediction = model.predict(sample[features])[0]
print(f"{prediction:.2f}")
