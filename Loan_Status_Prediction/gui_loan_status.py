import os
import pickle
import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Try to load model, otherwise train from loan data
MODEL_FILE = os.path.join(SCRIPT_DIR, 'loan_status_model.pkl')
CSV_FILE = os.path.join(SCRIPT_DIR, 'train_u6lujuX_CVtuZ9i (1).csv')

classifier = None

if os.path.exists(MODEL_FILE):
    try:
        with open(MODEL_FILE, 'rb') as f:
            classifier = pickle.load(f)
        print('Loaded existing model')
    except Exception as e:
        print('Failed to load model:', e)
        classifier = None

# If model not available, train from loan data
if classifier is None:
    try:
        from sklearn import svm
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        print('Training model from loan dataset')
        loan_status_data = pd.read_csv(CSV_FILE)
        
        # Data preprocessing
        loan_status_data = loan_status_data.dropna()
        loan_status_data.replace({"Loan_Status": {'N': 0, 'Y': 1}}, inplace=True)
        loan_status_data = loan_status_data.replace(to_replace='3+', value=4)
        loan_status_data.replace({"Married": {'No': 0, 'Yes': 1}}, inplace=True)
        loan_status_data.replace({"Gender": {'Male': 1, 'Female': 0}}, inplace=True)
        loan_status_data.replace({"Self_Employed": {'No': 0, 'Yes': 1}}, inplace=True)
        loan_status_data.replace({"Education": {'Graduate': 1, 'Not Graduate': 0}}, inplace=True)
        loan_status_data.replace({"Property_Area": {'Urban': 2, 'Semiurban': 1, 'Rural': 0}}, inplace=True)
        
        X = loan_status_data.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
        Y = loan_status_data['Loan_Status']

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)

        classifier = svm.SVC(kernel='linear', probability=True)
        classifier.fit(X_train, Y_train)

        # Save model
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(classifier, f)

        print('Training complete and model saved')
    except Exception as e:
        print('Failed to train model automatically:', e)

# GUI Setup
root = tk.Tk()
root.title('Loan Status Predictor')
root.geometry('500x650')

# Title
title_label = tk.Label(root, text='💰 Loan Status Predictor', font=('Helvetica', 16, 'bold'))
title_label.pack(pady=10)

# Instructions
info_label = tk.Label(root, text='Enter loan details to predict approval status', 
                      font=('Helvetica', 9), justify='center')
info_label.pack(pady=5)

# Scrollable frame for inputs
canvas = tk.Canvas(root, height=400, width=480)
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

# Feature definitions with expected ranges/values
FEATURES = [
    ('Gender', 'Gender (0=Female, 1=Male)', 'combo', [0, 1]),
    ('Married', 'Married (0=No, 1=Yes)', 'combo', [0, 1]),
    ('Dependents', 'Dependents (0-4)', 'combo', [0, 1, 2, 3, 4]),
    ('Education', 'Education (0=Not Graduate, 1=Graduate)', 'combo', [0, 1]),
    ('Self_Employed', 'Self Employed (0=No, 1=Yes)', 'combo', [0, 1]),
    ('ApplicantIncome', 'Applicant Income (numeric)', 'entry', None),
    ('CoapplicantIncome', 'Coapplicant Income (numeric)', 'entry', None),
    ('LoanAmount', 'Loan Amount (in thousands)', 'entry', None),
    ('Loan_Amount_Term', 'Loan Amount Term (months)', 'entry', None),
    ('Credit_History', 'Credit History (0 or 1)', 'combo', [0, 1]),
    ('Property_Area', 'Property Area (0=Rural, 1=Semiurban, 2=Urban)', 'combo', [0, 1, 2])
]

entries = {}
for i, (code, name, input_type, values) in enumerate(FEATURES):
    lbl = tk.Label(scrollable_frame, text=f'{name}:', font=('Helvetica', 9))
    lbl.grid(row=i, column=0, sticky='e', pady=3, padx=5)
    
    if input_type == 'combo':
        var = tk.StringVar()
        combo = tk.OptionMenu(scrollable_frame, var, *[str(v) for v in values])
        combo.grid(row=i, column=1, pady=3, padx=5, sticky='w')
        entries[code] = var
    else:
        ent = tk.Entry(scrollable_frame, width=20)
        ent.grid(row=i, column=1, pady=3, padx=5)
        entries[code] = ent

# Result label
result_label = tk.Label(root, text='', font=('Helvetica', 12, 'bold'), fg='darkgreen')
result_label.pack(pady=8)


def predict_from_entries():
    try:
        # Get all input values
        vals = []
        feature_order = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
                        'Credit_History', 'Property_Area']
        
        for code in feature_order:
            if isinstance(entries[code], tk.StringVar):
                txt = entries[code].get()
            else:
                txt = entries[code].get().strip()
            
            if txt == '' or txt == '':
                raise ValueError(f'Please enter value for {code}')
            vals.append(float(txt))

        if classifier is None:
            messagebox.showerror('Model missing', 'Model not available.')
            return

        # Make prediction
        arr = np.asarray(vals).reshape(1, -1)
        pred = classifier.predict(arr)[0]
        prob = classifier.predict_proba(arr)[0]

        if pred == 1:
            result_text = f'✅ LOAN APPROVED\nConfidence: {prob[1]*100:.1f}%'
            result_label.config(fg='darkgreen')
        else:
            result_text = f'❌ LOAN REJECTED\nConfidence: {prob[0]*100:.1f}%'
            result_label.config(fg='darkred')

        result_label.config(text=result_text)
    except ValueError as e:
        messagebox.showerror('Input error', str(e))
    except Exception as e:
        messagebox.showerror('Error', f'Prediction failed: {str(e)}')


def fill_sample():
    sample = {
        'Gender': '1',
        'Married': '1',
        'Dependents': '0',
        'Education': '1',
        'Self_Employed': '0',
        'ApplicantIncome': '5000',
        'CoapplicantIncome': '1500',
        'LoanAmount': '110',
        'Loan_Amount_Term': '360',
        'Credit_History': '1',
        'Property_Area': '2'
    }
    
    for code, value in sample.items():
        if isinstance(entries[code], tk.StringVar):
            entries[code].set(value)
        else:
            entries[code].delete(0, tk.END)
            entries[code].insert(0, value)


def clear_fields():
    for code in entries:
        if isinstance(entries[code], tk.StringVar):
            entries[code].set('')
        else:
            entries[code].delete(0, tk.END)
    result_label.config(text='')


# Button frame
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

predict_btn = tk.Button(btn_frame, text='Predict', command=predict_from_entries, 
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
