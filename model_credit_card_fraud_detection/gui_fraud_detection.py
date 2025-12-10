import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import os

class FraudDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Credit Card Fraud Detection System")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Load the model
        self.model = None
        self.load_model()
        
        # Set style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Create GUI
        self.create_widgets()
        
    def load_model(self):
        """Load the saved fraud detection model"""
        try:
            model_path = os.path.join(os.path.dirname(__file__), "fraud_detection_model.pkl")
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                print("Model loaded successfully!")
            else:
                messagebox.showerror("Error", "Model file not found. Please train the model first.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def create_widgets(self):
        """Create GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Credit Card Fraud Detection System", 
                                font=("Arial", 18, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # Tab 1: Single Transaction
        self.create_single_transaction_tab(notebook)
        
        # Tab 2: Batch File Upload
        self.create_batch_upload_tab(notebook)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
    
    def create_single_transaction_tab(self, notebook):
        """Create tab for single transaction prediction"""
        frame = ttk.Frame(notebook, padding="15")
        notebook.add(frame, text="Single Transaction")
        
        # Info label
        info_label = ttk.Label(frame, text="Enter transaction details below to check for fraud:", 
                              font=("Arial", 10))
        info_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Create input fields
        self.single_entries = {}
        fields = ["Time", "Amount", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", 
                  "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", 
                  "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"]
        
        # Create scrollable frame
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Sample data
        sample_data = {
            "Time": "0", "Amount": "149.62", "V1": "-1.36", "V2": "-0.07", "V3": "2.54",
            "V4": "1.39", "V5": "-0.31", "V6": "-0.62", "V7": "-0.99", "V8": "-0.24",
            "V9": "1.80", "V10": "0.79", "V11": "0.66", "V12": "-0.69", "V13": "-0.71",
            "V14": "-4.71", "V15": "3.53", "V16": "0.61", "V17": "0.26", "V18": "-0.57",
            "V19": "-3.04", "V20": "1.80", "V21": "0.64", "V22": "-1.66", "V23": "-0.22",
            "V24": "0.06", "V25": "0.23", "V26": "-0.64", "V27": "0.33", "V28": "0.17"
        }
        
        # Add input fields to scrollable frame
        for idx, field in enumerate(fields):
            label = ttk.Label(scrollable_frame, text=f"{field}:")
            label.grid(row=idx, column=0, sticky="w", padx=5, pady=5)
            
            entry = ttk.Entry(scrollable_frame, width=30)
            entry.insert(0, sample_data.get(field, ""))  # Insert sample data
            entry.grid(row=idx, column=1, sticky="w", padx=5, pady=5)
            self.single_entries[field] = entry
        
        canvas.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=1, column=2, sticky=(tk.N, tk.S))
        
        # Buttons frame
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=15)
        
        predict_btn = ttk.Button(button_frame, text="Predict", command=self.predict_single)
        predict_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = ttk.Button(button_frame, text="Clear", command=self.clear_single_fields)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Result frame
        result_frame = ttk.LabelFrame(frame, text="Prediction Result", padding="10")
        result_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        self.result_label = ttk.Label(result_frame, text="Result will appear here...", 
                                      font=("Arial", 11), wraplength=400)
        self.result_label.pack()
        
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
    
    def create_batch_upload_tab(self, notebook):
        """Create tab for batch file upload"""
        frame = ttk.Frame(notebook, padding="15")
        notebook.add(frame, text="Batch Prediction")
        
        # Info label
        info_label = ttk.Label(frame, text="Upload a CSV file with transaction data for batch prediction.", 
                              font=("Arial", 10))
        info_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # File selection
        file_label = ttk.Label(frame, text="CSV File:", font=("Arial", 10, "bold"))
        file_label.grid(row=1, column=0, sticky="w", padx=5, pady=10)
        
        self.file_path = tk.StringVar()
        file_entry = ttk.Entry(frame, textvariable=self.file_path, width=50, state="readonly")
        file_entry.grid(row=1, column=1, padx=5, pady=10)
        
        browse_btn = ttk.Button(frame, text="Browse", command=self.browse_file)
        browse_btn.grid(row=1, column=2, padx=5, pady=10)
        
        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=15)
        
        predict_btn = ttk.Button(button_frame, text="Predict on Batch", command=self.predict_batch)
        predict_btn.pack(side=tk.LEFT, padx=5)
        
        # Results display
        result_label = ttk.Label(frame, text="Prediction Results:", font=("Arial", 11, "bold"))
        result_label.grid(row=3, column=0, columnspan=3, sticky="w", pady=(10, 5))
        
        # Treeview for results
        self.batch_tree = ttk.Treeview(frame, columns=("Index", "Prediction", "Probability"), 
                                       height=15, show="headings")
        self.batch_tree.column("Index", width=50)
        self.batch_tree.column("Prediction", width=150)
        self.batch_tree.column("Probability", width=150)
        
        self.batch_tree.heading("Index", text="Index")
        self.batch_tree.heading("Prediction", text="Prediction")
        self.batch_tree.heading("Probability", text="Probability")
        
        self.batch_tree.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.batch_tree.yview)
        self.batch_tree.configure(yscroll=scrollbar.set)
        scrollbar.grid(row=4, column=3, sticky=(tk.N, tk.S))
        
        # Summary
        self.summary_label = ttk.Label(frame, text="", font=("Arial", 10))
        self.summary_label.grid(row=5, column=0, columnspan=3, sticky="w", pady=10)
        
        # Export button
        export_btn = ttk.Button(frame, text="Export Results as CSV", command=self.export_results)
        export_btn.grid(row=6, column=0, columnspan=3, pady=10)
        
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(4, weight=1)
    
    def predict_single(self):
        """Predict fraud for single transaction"""
        try:
            if self.model is None:
                messagebox.showerror("Error", "Model not loaded!")
                return
            
            # Collect input values
            input_data = []
            fields = ["Time", "Amount", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", 
                     "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", 
                     "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"]
            
            for field in fields:
                value = self.single_entries[field].get()
                if not value:
                    messagebox.showerror("Error", f"Please enter a value for {field}")
                    return
                try:
                    input_data.append(float(value))
                except ValueError:
                    messagebox.showerror("Error", f"{field} must be a number")
                    return
            
            # Create DataFrame
            input_df = pd.DataFrame([input_data], columns=fields)
            
            # Make prediction
            prediction = self.model.predict(input_df)[0]
            probability = self.model.predict_proba(input_df)[0]
            
            # Display result
            if prediction == 0:
                result_text = f"✓ LEGITIMATE TRANSACTION\n\nConfidence: {probability[0]*100:.2f}%"
                color = "green"
            else:
                result_text = f"⚠ FRAUDULENT TRANSACTION\n\nConfidence: {probability[1]*100:.2f}%"
                color = "red"
            
            self.result_label.config(text=result_text, foreground=color)
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def clear_single_fields(self):
        """Clear all input fields"""
        for entry in self.single_entries.values():
            entry.delete(0, tk.END)
        self.result_label.config(text="Result will appear here...", foreground="black")
    
    def browse_file(self):
        """Browse for CSV file"""
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if filename:
            self.file_path.set(filename)
    
    def predict_batch(self):
        """Predict fraud for batch file"""
        try:
            if self.model is None:
                messagebox.showerror("Error", "Model not loaded!")
                return
            
            file_path = self.file_path.get()
            if not file_path:
                messagebox.showerror("Error", "Please select a CSV file")
                return
            
            # Read CSV
            df = pd.read_csv(file_path)
            
            # Make predictions
            predictions = self.model.predict(df)
            probabilities = self.model.predict_proba(df)
            
            # Clear previous results
            for item in self.batch_tree.get_children():
                self.batch_tree.delete(item)
            
            # Add results to treeview
            fraud_count = 0
            for idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
                pred_text = "Fraudulent" if pred == 1 else "Legitimate"
                prob_text = f"{prob[1]*100:.2f}%" if pred == 1 else f"{prob[0]*100:.2f}%"
                
                if pred == 1:
                    fraud_count += 1
                
                self.batch_tree.insert("", "end", values=(idx, pred_text, prob_text))
            
            # Update summary
            total = len(predictions)
            legitimate = total - fraud_count
            summary = f"Total Transactions: {total} | Legitimate: {legitimate} | Fraudulent: {fraud_count} ({fraud_count/total*100:.2f}%)"
            self.summary_label.config(text=summary)
            
            messagebox.showinfo("Success", f"Prediction completed!\nFraudulent cases: {fraud_count}/{total}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Batch prediction failed: {str(e)}")
    
    def export_results(self):
        """Export results as CSV"""
        try:
            if not self.batch_tree.get_children():
                messagebox.showwarning("Warning", "No results to export")
                return
            
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", 
                                                    filetypes=[("CSV files", "*.csv")])
            if file_path:
                # Extract data from treeview
                data = []
                for item in self.batch_tree.get_children():
                    values = self.batch_tree.item(item)['values']
                    data.append(values)
                
                df = pd.DataFrame(data, columns=["Index", "Prediction", "Probability"])
                df.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Results exported to {file_path}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")


def main():
    root = tk.Tk()
    gui = FraudDetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
