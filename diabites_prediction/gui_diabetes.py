import os
import pickle
import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd

# Try to load model and scaler, otherwise train quickly from diabetes.csv
MODEL_FILE = 'diabetes_model.pkl'
SCALER_FILE = 'scaler.pkl'

classifier = None
scaler = None

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

# If not available, train a simple SVM classifier (same feature flow as the notebook)
if classifier is None or scaler is None:
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn import svm
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        print('Training model from diabetes.csv (this may take a few seconds)')
        df = pd.read_csv('data/diabetes.csv')
        X = df.drop(columns='Outcome', axis=1)
        Y = df['Outcome']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)

        classifier = svm.SVC(kernel='linear', probability=True)
        classifier.fit(X_train, Y_train)

        # Optionally save
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(classifier, f)
        with open(SCALER_FILE, 'wb') as f:
            pickle.dump(scaler, f)

        print('Training complete and model saved')
    except Exception as e:
        print('Failed to train model automatically:', e)

# GUI
FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

root = tk.Tk()
root.title('Diabetes Predictor (SVM)')

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
    # Validate inputs
    try:
        vals = []
        for name in FEATURE_NAMES:
            txt = entries[name].get().strip()
            if txt == '':
                raise ValueError(f'Please enter value for {name}')
            vals.append(float(txt))

        arr = np.asarray(vals).reshape(1, -1)

        if scaler is None or classifier is None:
            messagebox.showerror('Model missing', 'Model/scaler not available.')
            return

        arr_scaled = scaler.transform(arr)
        pred = classifier.predict(arr_scaled)[0]
        prob = classifier.predict_proba(arr_scaled)[0]

        if pred == 0:
            result_text = f'Prediction: NON-DIABETIC (0) — Confidence {prob[0]*100:.1f}%'
        else:
            result_text = f'Prediction: DIABETIC (1) — Confidence {prob[1]*100:.1f}%'

        result_label.config(text=result_text)
    except Exception as e:
        messagebox.showerror('Input error', str(e))


def fill_sample():
    sample = [6,148,72,35,0,33.6,0.627,50]
    for name, val in zip(FEATURE_NAMES, sample):
        entries[name].delete(0, tk.END)
        entries[name].insert(0, str(val))

btn_frame = tk.Frame(root)
btn_frame.pack(pady=6)

predict_btn = tk.Button(btn_frame, text='Predict', command=predict_from_entries, width=12)
predict_btn.grid(row=0, column=0, padx=5)

sample_btn = tk.Button(btn_frame, text='Fill Sample', command=fill_sample, width=12)
sample_btn.grid(row=0, column=1, padx=5)

quit_btn = tk.Button(btn_frame, text='Quit', command=root.destroy, width=12)
quit_btn.grid(row=0, column=2, padx=5)

root.mainloop()
