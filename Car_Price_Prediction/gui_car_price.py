import os
import pickle
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Car details v3.csv")
MODEL_PATH = os.path.join(BASE_DIR, "car_price_model.pkl")

# Categorical mappings (from notebook run)
FUEL_MAP = {"CNG": 0, "Diesel": 1, "LPG": 2, "Petrol": 3}
SELLER_MAP = {"Dealer": 0, "Individual": 1, "Trustmark Dealer": 2}
TRANS_MAP = {"Automatic": 0, "Manual": 1}
OWNER_MAP = {"First Owner": 0, "Fourth & Above Owner": 1, "Second Owner": 2, "Test Drive Car": 3, "Third Owner": 4}

# Preprocessing helpers
def extract_numeric(series):
    return series.astype(str).str.extract(r"(\d+\.?\d*)")[0]


def load_and_prepare_data(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)

    # Extract numeric parts and convert where necessary
    for col in ["mileage", "engine", "max_power", "torque"]:
        if col in df.columns:
            df[col] = extract_numeric(df[col])
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)

    # seats
    if "seats" in df.columns:
        df["seats"].fillna(df["seats"].mode()[0], inplace=True)

    # Map categorical columns using fixed maps; if unknown value, map to -1
    if "fuel" in df.columns:
        df["fuel_encoded"] = df["fuel"].map(FUEL_MAP).fillna(-1).astype(int)
    if "seller_type" in df.columns:
        df["seller_type_encoded"] = df["seller_type"].map(SELLER_MAP).fillna(-1).astype(int)
    if "transmission" in df.columns:
        df["transmission_encoded"] = df["transmission"].map(TRANS_MAP).fillna(-1).astype(int)
    if "owner" in df.columns:
        df["owner_encoded"] = df["owner"].map(OWNER_MAP).fillna(-1).astype(int)

    # Drop columns we won't use
    drop_cols = [c for c in ["name", "fuel", "seller_type", "transmission", "owner"] if c in df.columns]
    df = df.drop(drop_cols, axis=1)

    return df


def train_or_load_model(df, model_path=MODEL_PATH):
    features = [
        "year",
        "km_driven",
        "mileage",
        "engine",
        "max_power",
        "torque",
        "seats",
        "fuel_encoded",
        "seller_type_encoded",
        "transmission_encoded",
        "owner_encoded",
    ]

    # Ensure features exist
    features = [f for f in features if f in df.columns]

    X = df[features]
    y = df["selling_price"]

    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            return model, features
        except Exception:
            pass

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    try:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    except Exception:
        print("Warning: could not save model")

    return model, features


# Build GUI
class CarPriceGUI(tk.Tk):
    def __init__(self, model, features):
        super().__init__()
        self.title("Car Price Predictor")
        self.geometry("520x560")
        self.model = model
        self.features = features

        # Input fields
        frame = ttk.Frame(self, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        # Numeric inputs
        self.inputs = {}
        numeric_fields = ["year", "km_driven", "mileage", "engine", "max_power", "torque", "seats"]
        row = 0
        for f in numeric_fields:
            if f in features:
                ttk.Label(frame, text=f.capitalize() + ":").grid(column=0, row=row, sticky=tk.W, pady=6)
                ent = ttk.Entry(frame)
                ent.grid(column=1, row=row, sticky=tk.W, pady=6)
                self.inputs[f] = ent
                row += 1

        # Categorical comboboxes
        if "fuel_encoded" in features:
            ttk.Label(frame, text="Fuel:").grid(column=0, row=row, sticky=tk.W, pady=6)
            fuel_cb = ttk.Combobox(frame, values=list(FUEL_MAP.keys()), state="readonly")
            fuel_cb.current(0)
            fuel_cb.grid(column=1, row=row, sticky=tk.W, pady=6)
            self.inputs["fuel_encoded"] = fuel_cb
            row += 1

        if "seller_type_encoded" in features:
            ttk.Label(frame, text="Seller Type:").grid(column=0, row=row, sticky=tk.W, pady=6)
            seller_cb = ttk.Combobox(frame, values=list(SELLER_MAP.keys()), state="readonly")
            seller_cb.current(0)
            seller_cb.grid(column=1, row=row, sticky=tk.W, pady=6)
            self.inputs["seller_type_encoded"] = seller_cb
            row += 1

        if "transmission_encoded" in features:
            ttk.Label(frame, text="Transmission:").grid(column=0, row=row, sticky=tk.W, pady=6)
            trans_cb = ttk.Combobox(frame, values=list(TRANS_MAP.keys()), state="readonly")
            trans_cb.current(0)
            trans_cb.grid(column=1, row=row, sticky=tk.W, pady=6)
            self.inputs["transmission_encoded"] = trans_cb
            row += 1

        if "owner_encoded" in features:
            ttk.Label(frame, text="Owner:").grid(column=0, row=row, sticky=tk.W, pady=6)
            owner_cb = ttk.Combobox(frame, values=list(OWNER_MAP.keys()), state="readonly")
            owner_cb.current(0)
            owner_cb.grid(column=1, row=row, sticky=tk.W, pady=6)
            self.inputs["owner_encoded"] = owner_cb
            row += 1

        # Predict button and Example button (side-by-side)
        predict_btn = ttk.Button(frame, text="Predict Price", command=self.predict_price)
        predict_btn.grid(column=0, row=row, sticky=tk.EW, padx=(0, 6), pady=16)

        example_btn = ttk.Button(frame, text="Load Example", command=self.load_example)
        example_btn.grid(column=1, row=row, sticky=tk.EW, padx=(6, 0), pady=16)

        # Output
        self.result_var = tk.StringVar(value="Prediction will appear here")
        ttk.Label(frame, textvariable=self.result_var, font=(None, 12, "bold")).grid(column=0, row=row+1, columnspan=2, pady=8)

    def predict_price(self):
        try:
            # Build feature vector
            x = []
            for f in ["year", "km_driven", "mileage", "engine", "max_power", "torque", "seats"]:
                if f in self.features:
                    val = self.inputs[f].get().strip()
                    if val == "":
                        messagebox.showerror("Input error", f"Please enter a value for {f}")
                        return
                    x.append(float(val))

            # categorical
            if "fuel_encoded" in self.features:
                fuel_val = self.inputs["fuel_encoded"].get()
                x.append(FUEL_MAP.get(fuel_val, -1))
            if "seller_type_encoded" in self.features:
                seller_val = self.inputs["seller_type_encoded"].get()
                x.append(SELLER_MAP.get(seller_val, -1))
            if "transmission_encoded" in self.features:
                trans_val = self.inputs["transmission_encoded"].get()
                x.append(TRANS_MAP.get(trans_val, -1))
            if "owner_encoded" in self.features:
                owner_val = self.inputs["owner_encoded"].get()
                x.append(OWNER_MAP.get(owner_val, -1))

            X_input = np.array(x).reshape(1, -1)
            pred = self.model.predict(X_input)[0]
            self.result_var.set(f"Predicted Selling Price: ₹{pred:,.0f}")
        except ValueError as e:
            messagebox.showerror("Input error", f"Invalid input: {e}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_example(self):
        """Populate the form with example values for quick testing."""
        example = {
            "year": "2016",
            "km_driven": "50000",
            "mileage": "18.0",
            "engine": "1197",
            "max_power": "77",
            "torque": "190",
            "seats": "5",
            # categorical
            "fuel_encoded": "Petrol",
            "seller_type_encoded": "Individual",
            "transmission_encoded": "Manual",
            "owner_encoded": "First Owner",
        }

        for k, v in example.items():
            widget = self.inputs.get(k)
            if widget is None:
                continue
            # Entry widgets
            try:
                # entries have .delete and .insert
                widget.delete(0, tk.END)
                widget.insert(0, v)
            except Exception:
                # Comboboxes use set
                try:
                    widget.set(v)
                except Exception:
                    pass


def main():
    if not os.path.exists(CSV_PATH):
        tk.messagebox.showerror("File missing", f"Dataset not found at {CSV_PATH}. Place 'Car details v3.csv' in this folder.")
        return

    df = load_and_prepare_data(CSV_PATH)
    model, features = train_or_load_model(df)

    app = CarPriceGUI(model, features)
    app.mainloop()


if __name__ == "__main__":
    main()
