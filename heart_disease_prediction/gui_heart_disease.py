import os
import pickle
import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
import traceback

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Filenames (saved next to this script)
MODEL_FILE = 'heart_model.pkl'
SCALER_FILE = 'heart_scaler.pkl'
DATA_FILE = 'heart_disease_data.csv'

classifier = None
scaler = None

# Try to load model and scaler if they exist
if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
    try:
        with open(MODEL_FILE, 'rb') as f:
            classifier = pickle.load(f)
        with open(SCALER_FILE, 'rb') as f:
            scaler = pickle.load(f)
        print('Loaded existing model and scaler')
    except Exception as e:
        print('Failed to load pickles:', e)
        classifier = None
        scaler = None

# If model/scaler missing, train from CSV
if classifier is None or scaler is None:
    try:
        print('Training RandomForest model from CSV (this may take a few seconds)')
        df = pd.read_csv(DATA_FILE)
        X = df.drop(columns='target', axis=1)
        Y = df['target']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=2)

        classifier = RandomForestClassifier(n_estimators=200, random_state=42)
        classifier.fit(X_train, Y_train)

        # Save for reuse
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(classifier, f)
        with open(SCALER_FILE, 'wb') as f:
            pickle.dump(scaler, f)

        print('Training complete — model and scaler saved')
    except Exception as e:
        print('Failed to train model automatically:', e)

# Features order (matches CSV header)
FEATURE_NAMES = [
    'age','sex','cp','trestbps','chol','fbs','restecg',
    'thalach','exang','oldpeak','slope','ca','thal'
]

root = tk.Tk()
root.title('Heart Disease Predictor')

entries = {}

frame = tk.Frame(root, padx=10, pady=10)
frame.pack()

for i, name in enumerate(FEATURE_NAMES):
    lbl = tk.Label(frame, text=name+':')
    lbl.grid(row=i, column=0, sticky='e', pady=3)
    ent = tk.Entry(frame, width=20)
    ent.grid(row=i, column=1, pady=3)
    entries[name] = ent

result_label = tk.Label(root, text='', font=('Helvetica', 12, 'bold'))
result_label.pack(pady=8)


def predict_from_entries():
    try:
        vals = []
        for name in FEATURE_NAMES:
            txt = entries[name].get().strip()
            if txt == '':
                raise ValueError(f'Please enter value for {name}')
            vals.append(float(txt))

        arr = np.asarray(vals).reshape(1, -1)

        if scaler is None or classifier is None:
            messagebox.showerror('Model missing', 'Model or scaler not available.')
            return
        # StandardScaler was originally fitted on a DataFrame with column names.
        # Convert the input row to a DataFrame with matching column names to avoid warnings
        # and ensure correct feature ordering.
        arr_df = pd.DataFrame(arr, columns=FEATURE_NAMES)
        arr_scaled = scaler.transform(arr_df)
        pred = classifier.predict(arr_scaled)[0]
        # if classifier supports predict_proba, show confidence
        try:
            prob = classifier.predict_proba(arr_scaled)[0]
            conf = max(prob) * 100
            conf_text = f' — Confidence {conf:.1f}%'
        except Exception:
            conf_text = ''

        if pred == 0:
            result_text = f'Prediction: NO HEART DISEASE (0){conf_text}'
        else:
            result_text = f'Prediction: HEART DISEASE (1){conf_text}'

        result_label.config(text=result_text)
    except Exception as e:
        # Show message to user and print traceback to console for debugging
        messagebox.showerror('Input error', str(e))
        print('Error during prediction:')
        traceback.print_exc()


def fill_example_positive():
    # Example positive case (first row of CSV is a positive example)
    sample = [63,1,3,145,233,1,0,150,0,2.3,0,0,1]
    for name, val in zip(FEATURE_NAMES, sample):
        entries[name].delete(0, tk.END)
        entries[name].insert(0, str(val))


def fill_example_negative():
    # A negative example (from CSV later where target==0)
    sample = [67,1,0,160,286,0,0,108,1,1.5,1,3,2]
    for name, val in zip(FEATURE_NAMES, sample):
        entries[name].delete(0, tk.END)
        entries[name].insert(0, str(val))

btn_frame = tk.Frame(root)
btn_frame.pack(pady=6)

predict_btn = tk.Button(btn_frame, text='Predict', command=predict_from_entries, width=12)
predict_btn.grid(row=0, column=0, padx=5)

example_pos_btn = tk.Button(btn_frame, text='Example: Positive', command=fill_example_positive, width=15)
example_pos_btn.grid(row=0, column=1, padx=5)

example_neg_btn = tk.Button(btn_frame, text='Example: Negative', command=fill_example_negative, width=15)
example_neg_btn.grid(row=0, column=2, padx=5)

quit_btn = tk.Button(btn_frame, text='Quit', command=root.destroy, width=12)
quit_btn.grid(row=0, column=3, padx=5)

root.mainloop()
