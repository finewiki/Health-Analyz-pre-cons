import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def submit():
    # Retrieving the values entered by the user
    hemoglobin1 = float(hemoglobin_entry.get())
    whiteblood1 = float(whiteblood_entry.get())
    bloodpressure1 = float(bloodpressure_entry.get())
    age1 = int(age_entry.get())
    gender1 = int(gender_entry.get())  # 0 = Female, 1 = Male (Example)
    
    # Load the dataset and split it into training and testing sets
    health_data = pd.read_csv('blood_data.csv')  # Add your dataset here
    X_train, X_test, y_train, y_test = train_test_split(health_data.loc[:, health_data.columns != 'HealthStatus'],
                                                        health_data['HealthStatus'],
                                                        stratify=health_data['HealthStatus'],
                                                        random_state=66)

    # Initialize and train the KNN model
    knn = KNeighborsClassifier(n_neighbors=9)
    knn.fit(X_train, y_train)

    # Get the user input data and make a prediction
    Xn = [[hemoglobin1, whiteblood1, bloodpressure1, age1, gender1]]
    result = knn.predict(Xn)

    # Display result in a message box
    if result == 0:
        messagebox.showinfo(title="Result", message="Your blood values are normal, you are healthy.")
    else:
        messagebox.showinfo(title="Result", message="Some of your blood values may be risky, consult your doctor.")

# Create the main window
root = tk.Tk()
root.title("Blood Value Analyzer")

# Labels and entry fields for user input
hemoglobin_label = tk.Label(root, text="Hemoglobin Value (g/dL):")
hemoglobin_label.grid(row=0, column=0)
hemoglobin_entry = tk.Entry(root)
hemoglobin_entry.grid(row=0, column=1)

whiteblood_label = tk.Label(root, text="White Blood Cell Count (10^9/L):")
whiteblood_label.grid(row=1, column=0)
whiteblood_entry = tk.Entry(root)
whiteblood_entry.grid(row=1, column=1)

bloodpressure_label = tk.Label(root, text="Blood Pressure (mmHg):")
bloodpressure_label.grid(row=2, column=0)
bloodpressure_entry = tk.Entry(root)
bloodpressure_entry.grid(row=2, column=1)

age_label = tk.Label(root, text="Age:")
age_label.grid(row=3, column=0)
age_entry = tk.Entry(root)
age_entry.grid(row=3, column=1)

gender_label = tk.Label(root, text="Gender (0=Female, 1=Male):")
gender_label.grid(row=4, column=0)
gender_entry = tk.Entry(root)
gender_entry.grid(row=4, column=1)

# Submit button
submit_button = tk.Button(root, text="Show Result", command=submit)
submit_button.grid(row=5, columnspan=2)

# Start the GUI event loop
root.mainloop()

