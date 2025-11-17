import os
import pickle
import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd

MODEL_FILE = 'gold_price_model.pkl'
CSV_PATH = r"C:\Users\rohit\OneDrive\Desktop\mach_learning\Gold_Price_Prediction\gld_price_data.csv"

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
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split

        print('Training model from CSV (gld_price_data.csv)')
        df = pd.read_csv(CSV_PATH)
        # Convert Date if present
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Features used in notebook: drop Date and GLD
        FEATURES = [c for c in df.columns if c not in ['Date', 'GLD']]
        X = df[FEATURES].copy()
        Y = df['GLD']

        # Some CSVs may contain commas or missing values; drop NA rows for training
        data = pd.concat([X, Y], axis=1).dropna()
        X = data[FEATURES]
        Y = data['GLD']

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
        model = RandomForestRegressor(n_estimators=100, random_state=2)
        model.fit(X_train, Y_train)

        # Save model for future runs
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model, f)
        print('Training complete and model saved to', MODEL_FILE)
    except Exception as e:
        print('Training failed:', e)

# If for any reason features couldn't be inferred from training block, fall back to defaults
DEFAULT_FEATURES = ['SPX', 'USO', 'SLV', 'EUR/USD']

try:
    # If we trained above, FEATURES is defined; otherwise use defaults if present in CSV
    FEATURES
except NameError:
    try:
        df = pd.read_csv(CSV_PATH, nrows=1)
        FEATURES = [c for c in df.columns if c not in ['Date', 'GLD']]
        if not FEATURES:
            FEATURES = DEFAULT_FEATURES
    except Exception:
        FEATURES = DEFAULT_FEATURES

# Build GUI
root = tk.Tk()
root.title('Gold Price Predictor (GLD)')
root.geometry('520x420')

title = tk.Label(root, text='Gold Price Predictor', font=('Helvetica', 16, 'bold'))
title.pack(pady=8)

info = tk.Label(root, text='Enter the features below to predict GLD (Gold Price)', font=('Helvetica', 9))
info.pack(pady=4)

frame = tk.Frame(root)
frame.pack(padx=10, pady=6)

entries = {}
for i, feat in enumerate(FEATURES):
    lbl = tk.Label(frame, text=f'{feat}:', anchor='w')
    lbl.grid(row=i, column=0, sticky='w', pady=8)
    ent = tk.Entry(frame, width=28)
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
            messagebox.showerror('Model missing', 'Model not available. Unable to predict.')
            return
        pred = model.predict(arr)[0]

        result_label.config(text=f'Predicted GLD: {pred:.4f}', fg='darkblue')
    except ValueError as e:
        messagebox.showerror('Input error', str(e))
    except Exception as e:
        messagebox.showerror('Error', f'Prediction failed: {e}')


def fill_sample():
    # Use a recent-ish sample or medians from training set if available
    sample = None
    try:
        df = pd.read_csv(CSV_PATH)
        # compute medians for features
        if set(FEATURES).issubset(set(df.columns)):
            med = df[FEATURES].median()
            sample = med.tolist()
    except Exception:
        sample = None

    if sample is None:
        # fallback example values
        sample = [1400.0, 75.0, 16.0, 1.45]

    for feat, val in zip(FEATURES, sample):
        entries[feat].delete(0, tk.END)
        entries[feat].insert(0, str(round(float(val), 6)))


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
clear_btn.grid(row=0, column=2, padx=6)

quit_btn = tk.Button(root, text='Quit', command=root.destroy, bg='#f44336', fg='white', width=26)
quit_btn.pack(pady=10)

if __name__ == '__main__':
    root.mainloop()
