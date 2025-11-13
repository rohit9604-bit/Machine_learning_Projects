import os
import pickle
import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd

MODEL_FILE = 'wine_quality_model.pkl'

model = None

# Try to load existing model
if os.path.exists(MODEL_FILE):
    try:
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        print('Loaded model from', MODEL_FILE)
    except Exception as e:
        print('Failed to load model:', e)
        model = None

# If model not found, train quickly from CSV
if model is None:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        print('Training model from CSV (winequality-red.csv)')
        df = pd.read_csv(r"C:\Users\rohit\OneDrive\Desktop\mach_learning\Wine_prediction\winequality-red.csv")
        X = df.drop('quality', axis=1)
        Y = df['quality'].apply(lambda y: 1 if y >= 7 else 0)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
        model = RandomForestClassifier()
        model.fit(X_train, Y_train)

        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model, f)
        print('Training complete and model saved to', MODEL_FILE)
    except Exception as e:
        print('Training failed:', e)

# Feature list for red wine dataset
FEATURES = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
    'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
]

# Build GUI
root = tk.Tk()
root.title('Wine Quality Predictor (Good vs Bad)')
root.geometry('480x620')

title = tk.Label(root, text='Wine Quality Predictor', font=('Helvetica', 16, 'bold'))
title.pack(pady=8)

info = tk.Label(root, text='Enter the 11 chemical features below.\nPrediction: Good (1) if quality>=7 else Bad (0)', font=('Helvetica', 9))
info.pack(pady=4)

frame = tk.Frame(root)
frame.pack(padx=10, pady=6)

entries = {}
for i, feat in enumerate(FEATURES):
    lbl = tk.Label(frame, text=f'{feat}:', anchor='w')
    lbl.grid(row=i, column=0, sticky='w', pady=6)
    ent = tk.Entry(frame, width=24)
    ent.grid(row=i, column=1, padx=6)
    entries[feat] = ent

result_label = tk.Label(root, text='', font=('Helvetica', 12, 'bold'))
result_label.pack(pady=10)


def predict():
    try:
        values = []
        for feat in FEATURES:
            val = entries[feat].get().strip()
            if val == '':
                raise ValueError(f'Please enter value for {feat}')
            values.append(float(val))

        arr = np.asarray(values).reshape(1, -1)
        if model is None:
            messagebox.showerror('Model missing', 'Model not available.')
            return
        pred = model.predict(arr)[0]
        prob = model.predict_proba(arr)[0]

        if pred == 1:
            result_label.config(text=f'Prediction: GOOD (1) — Confidence {prob[1]*100:.1f}%', fg='darkgreen')
        else:
            result_label.config(text=f'Prediction: BAD (0) — Confidence {prob[0]*100:.1f}%', fg='darkred')
    except ValueError as e:
        messagebox.showerror('Input error', str(e))
    except Exception as e:
        messagebox.showerror('Error', f'Prediction failed: {e}')


def fill_sample():
    # median-ish sample from common distributions
    sample = [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]
    for feat, val in zip(FEATURES, sample):
        entries[feat].delete(0, tk.END)
        entries[feat].insert(0, str(val))


def clear_all():
    for feat in FEATURES:
        entries[feat].delete(0, tk.END)
    result_label.config(text='')

btn_frame = tk.Frame(root)
btn_frame.pack(pady=8)

predict_btn = tk.Button(btn_frame, text='Predict', command=predict, bg='#4CAF50', fg='white', width=12)
predict_btn.grid(row=0, column=0, padx=6)

sample_btn = tk.Button(btn_frame, text='Fill Sample', command=fill_sample, bg='#2196F3', fg='white', width=12)
sample_btn.grid(row=0, column=1, padx=6)

clear_btn = tk.Button(btn_frame, text='Clear', command=clear_all, bg='#FF9800', fg='white', width=12)
clear_btn.grid(row=1, column=0, columnspan=2, pady=6)

quit_btn = tk.Button(btn_frame, text='Quit', command=root.destroy, bg='#f44336', fg='white', width=26)
quit_btn.grid(row=2, column=0, columnspan=2, pady=6)

root.mainloop()
