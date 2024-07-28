import tkinter as tk
from tkinter import messagebox
import numpy as np
import pickle

# Load the model and scaler
with open('parkinsons_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

def predict_parkinsons():
    try:
        features = [float(entry.get()) for entry in entries]
        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)
        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0][prediction]
        
        result = f'Prediction: {"Parkinsons" if prediction == 1 else "Healthy"}\nProbability: {prob:.2f}'
        messagebox.showinfo("Prediction Result", result)
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create the main window
window = tk.Tk()
window.title("Parkinson's Disease Detection")

# Create and place labels and entry fields for the features
entries = []
feature_names = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']

for i, feature in enumerate(feature_names):
    label = tk.Label(window, text=feature)
    label.grid(row=i, column=0)
    entry = tk.Entry(window)
    entry.grid(row=i, column=1)
    entries.append(entry)

# Create and place the predict button
predict_button = tk.Button(window, text="Predict", command=predict_parkinsons)
predict_button.grid(row=len(feature_names), column=0, columnspan=2)

# Start the GUI event loop
window.mainloop()
