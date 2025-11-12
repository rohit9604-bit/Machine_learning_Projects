import os
import pickle
import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Try to load model, otherwise train from housing.csv
MODEL_FILE = os.path.join(SCRIPT_DIR, 'house_price_model.pkl')
CSV_FILE = os.path.join(SCRIPT_DIR, 'housing.csv')

model = None

if os.path.exists(MODEL_FILE):
    try:
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        print('Loaded existing model')
    except Exception as e:
        print('Failed to load model:', e)
        model = None

# If model not available, train from housing data
if model is None:
    try:
        from sklearn.model_selection import train_test_split
        from xgboost import XGBRegressor

        print('Training model from housing.csv')
        column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
        data = pd.read_csv(CSV_FILE, header=None, delimiter=r"\s+", names=column_names)
        data = data.rename(columns={'MEDV': 'Price'})
        
        X = data.drop('Price', axis=1)
        Y = data['Price']

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

        model = XGBRegressor()
        model.fit(X_train, Y_train)

        # Save model
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model, f)

        print('Training complete and model saved')
    except Exception as e:
        print('Failed to train model automatically:', e)

# GUI Setup
root = tk.Tk()
root.title('House Price Predictor')
root.geometry('500x700')

# Title
title_label = tk.Label(root, text='🏠 House Price Predictor', font=('Helvetica', 16, 'bold'))
title_label.pack(pady=10)

# Instructions
info_label = tk.Label(root, text='Enter housing features to predict price (in $1000s)', 
                      font=('Helvetica', 9), justify='center')
info_label.pack(pady=5)

# Scrollable frame for inputs
canvas = tk.Canvas(root, height=450, width=480)
scrollbar = tk.Scrollbar(root, orient='vertical', command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True, padx=10, pady=5)
scrollbar.pack(side="right", fill="y")

# Feature definitions
FEATURES = [
    ('CRIM', 'Crime Rate'),
    ('ZN', 'Proportion of Residential Land'),
    ('INDUS', 'Proportion of Non-Retail Business'),
    ('CHAS', 'Charles River (1 if bounds, 0 else)'),
    ('NOX', 'Nitric Oxide Concentration'),
    ('RM', 'Average Rooms per Dwelling'),
    ('AGE', 'Proportion of Old Buildings (%)'),
    ('DIS', 'Distance to Employment Centers'),
    ('RAD', 'Accessibility to Highways'),
    ('TAX', 'Property Tax Rate'),
    ('PTRATIO', 'Pupil-Teacher Ratio'),
    ('B', 'Black Population Proportion'),
    ('LSTAT', 'Lower Status Population (%)')
]

entries = {}
for i, (code, name) in enumerate(FEATURES):
    lbl = tk.Label(scrollable_frame, text=f'{name} ({code}):', font=('Helvetica', 9))
    lbl.grid(row=i, column=0, sticky='e', pady=3, padx=5)
    ent = tk.Entry(scrollable_frame, width=20)
    ent.grid(row=i, column=1, pady=3, padx=5)
    entries[code] = ent

# Result label
result_label = tk.Label(root, text='', font=('Helvetica', 12, 'bold'), fg='darkgreen')
result_label.pack(pady=8)


def predict_from_entries():
    try:
        # Get all 13 input values
        vals = []
        for code, name in FEATURES:
            txt = entries[code].get().strip()
            if txt == '':
                raise ValueError(f'Please enter value for {name}')
            vals.append(float(txt))

        if model is None:
            messagebox.showerror('Model missing', 'Model not available.')
            return

        # Make prediction
        arr = np.asarray(vals).reshape(1, -1)
        pred = model.predict(arr)[0]

        result_label.config(text=f'🏠 Predicted Price: ${pred*1000:,.2f}')
    except ValueError as e:
        messagebox.showerror('Input error', str(e))
    except Exception as e:
        messagebox.showerror('Error', f'Prediction failed: {str(e)}')


def fill_sample():
    sample = [0.1, 12.5, 7.07, 0, 0.538, 6.5, 60.0, 4.0, 4, 300.0, 18.5, 396.0, 10.0]
    for (code, name), val in zip(FEATURES, sample):
        entries[code].delete(0, tk.END)
        entries[code].insert(0, str(val))


def clear_fields():
    for code, name in FEATURES:
        entries[code].delete(0, tk.END)
    result_label.config(text='')


# Button frame
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

predict_btn = tk.Button(btn_frame, text='Predict Price', command=predict_from_entries, 
                       bg='#4CAF50', fg='white', font=('Helvetica', 10, 'bold'), width=12)
predict_btn.grid(row=0, column=0, padx=5)

sample_btn = tk.Button(btn_frame, text='Fill Sample', command=fill_sample, 
                      bg='#2196F3', fg='white', font=('Helvetica', 10, 'bold'), width=12)
sample_btn.grid(row=0, column=1, padx=5)

clear_btn = tk.Button(btn_frame, text='Clear', command=clear_fields, 
                     bg='#FF9800', fg='white', font=('Helvetica', 10, 'bold'), width=12)
clear_btn.grid(row=1, column=0, columnspan=2, pady=5)

quit_btn = tk.Button(btn_frame, text='Quit', command=root.destroy, 
                    bg='#f44336', fg='white', font=('Helvetica', 10, 'bold'), width=26)
quit_btn.grid(row=2, column=0, columnspan=2, pady=5)

root.mainloop()
