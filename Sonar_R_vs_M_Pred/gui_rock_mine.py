import os
import pickle
import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd

# Try to load model, otherwise train from sonar_dataset.csv
MODEL_FILE = 'rock_mine_model.pkl'

model = None

if os.path.exists(MODEL_FILE):
    try:
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        print('Loaded existing model')
    except Exception as e:
        print('Failed to load model:', e)
        model = None

# If model not available, train from sonar data
if model is None:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        print('Training model from sonar_dataset.csv')
        sonar_dataset = pd.read_csv('C:/Users/rohit/OneDrive/Desktop/mach_learning/data/sonar_dataset.csv', header=None)
        X = sonar_dataset.drop(columns=60, axis=1)
        Y = sonar_dataset[60]

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, Y_train)

        # Save model
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model, f)

        print('Training complete and model saved')
    except Exception as e:
        print('Failed to train model automatically:', e)

# GUI Setup
root = tk.Tk()
root.title('Rock vs Mine Predictor')
root.geometry('400x350')

# Title
title_label = tk.Label(root, text='🏔️ Rock vs Mine Predictor 💣', font=('Helvetica', 14, 'bold'))
title_label.pack(pady=10)

# Instructions
info_label = tk.Label(root, text='Enter 5 sonar frequency values (0-1):\n(Remaining 55 values will be auto-filled)', 
                      font=('Helvetica', 9), justify='center')
info_label.pack(pady=5)

# Input frame
frame = tk.Frame(root, padx=10, pady=10)
frame.pack()

entries = {}
for i in range(1, 6):
    lbl = tk.Label(frame, text=f'Frequency {i}:')
    lbl.grid(row=i-1, column=0, sticky='e', pady=5)
    ent = tk.Entry(frame, width=25)
    ent.grid(row=i-1, column=1, pady=5)
    entries[f'freq_{i}'] = ent

# Result label
result_label = tk.Label(root, text='', font=('Helvetica', 12, 'bold'), fg='darkblue')
result_label.pack(pady=10)


def predict_from_entries():
    try:
        # Get 5 input values
        vals = []
        for i in range(1, 6):
            txt = entries[f'freq_{i}'].get().strip()
            if txt == '':
                raise ValueError(f'Please enter value for Frequency {i}')
            val = float(txt)
            if not (0 <= val <= 1):
                raise ValueError(f'Frequency {i} must be between 0 and 1')
            vals.append(val)

        # Fill remaining 55 with default value
        default_values = [0.02] * 55
        vals.extend(default_values)

        if model is None:
            messagebox.showerror('Model missing', 'Model not available.')
            return

        # Make prediction
        arr = np.asarray(vals).reshape(1, -1)
        pred = model.predict(arr)[0]
        prob = model.predict_proba(arr)[0]

        if pred == 'R':
            result_text = f'🏔️ ROCK\nConfidence: {prob[0]*100:.1f}%'
        else:
            result_text = f'💣 MINE\nConfidence: {prob[1]*100:.1f}%'

        result_label.config(text=result_text)
    except ValueError as e:
        messagebox.showerror('Input error', str(e))
    except Exception as e:
        messagebox.showerror('Error', f'Prediction failed: {str(e)}')


def fill_sample():
    sample = [0.02, 0.0371, 0.0428, 0.0207, 0.0954]
    for i in range(1, 6):
        entries[f'freq_{i}'].delete(0, tk.END)
        entries[f'freq_{i}'].insert(0, str(sample[i-1]))


def clear_fields():
    for i in range(1, 6):
        entries[f'freq_{i}'].delete(0, tk.END)
    result_label.config(text='')


# Button frame
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

predict_btn = tk.Button(btn_frame, text='Predict', command=predict_from_entries, 
                       bg='#4CAF50', fg='white', font=('Helvetica', 10, 'bold'), width=10)
predict_btn.grid(row=0, column=0, padx=5)

sample_btn = tk.Button(btn_frame, text='Fill Sample', command=fill_sample, 
                      bg='#2196F3', fg='white', font=('Helvetica', 10, 'bold'), width=10)
sample_btn.grid(row=0, column=1, padx=5)

clear_btn = tk.Button(btn_frame, text='Clear', command=clear_fields, 
                     bg='#FF9800', fg='white', font=('Helvetica', 10, 'bold'), width=10)
clear_btn.grid(row=0, column=2, padx=5)

quit_btn = tk.Button(btn_frame, text='Quit', command=root.destroy, 
                    bg='#f44336', fg='white', font=('Helvetica', 10, 'bold'), width=10)
quit_btn.grid(row=1, column=0, columnspan=3, pady=10)

root.mainloop()
