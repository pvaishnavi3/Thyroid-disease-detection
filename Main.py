from tkinter import messagebox, simpledialog, filedialog
from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import seaborn as sns

global filename, X, Y, dataset, text, classifier
global X_train, X_test, y_train, y_test, predict_cls
accuracy, precision, recall, fscore = [], [], [], []

# Initialize GUI
main = Tk()
main.title("Detection of Thyroid Disorders Using Machine Learning")
main.geometry("1300x800")

# Function to upload dataset
def uploadDataset():
    global filename, dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    
    if not filename:
        text.insert(END, "No file selected!\n")
        return
    
    dataset = pd.read_csv(filename)
    text.insert(END, "Dataset loaded successfully!\n\n")
    text.insert(END, "Preview of dataset:\n")
    text.insert(END, str(dataset.head()) + "\n\n")
    
    # Plot class distribution
    plt.figure(figsize=(6,4))
    dataset['FLAG'].value_counts().plot(kind="bar", color=['blue', 'red'])
    plt.title("Thyroid Disorder Class Distribution (0 = Normal, 1 = Disorder)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

# Function to preprocess data
def trainTest():
    global X, Y, dataset, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    
    if dataset is None:
        text.insert(END, "Please upload dataset first!\n")
        return
    
    dataset.fillna(0, inplace=True)  # Replace missing values with 0
    Y = dataset['FLAG'].values       # Extract target variable
    X = dataset.iloc[:, 4:-2].values # Extract features (check indexing based on dataset)

    # Normalize features
    X = normalize(X)

    # Shuffle dataset
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X, Y = X[indices], Y[indices]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    text.insert(END, "Dataset processed successfully!\n")
    text.insert(END, f"Total records: {X.shape[0]}\n")
    text.insert(END, f"Total features: {X.shape[1]}\n")
    text.insert(END, f"Training data: {X_train.shape[0]}\n")
    text.insert(END, f"Testing data: {X_test.shape[0]}\n")

# Function to calculate evaluation metrics
def calculateMetrics(algorithm, y_pred):
    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, average='macro') * 100
    rec = recall_score(y_test, y_pred, average='macro') * 100
    f1 = f1_score(y_test, y_pred, average='macro') * 100

    accuracy.append(acc)
    precision.append(prec)
    recall.append(rec)
    fscore.append(f1)

    text.insert(END, f"{algorithm} Accuracy  : {acc:.2f}%\n")
    text.insert(END, f"{algorithm} Precision : {prec:.2f}%\n")
    text.insert(END, f"{algorithm} Recall    : {rec:.2f}%\n")
    text.insert(END, f"{algorithm} F1 Score  : {f1:.2f}%\n\n")

# Function to run Logistic Regression
def runLogisticRegression():
    text.delete('1.0', END)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    calculateMetrics("Logistic Regression", y_pred)

# Function to run Multi-Layer Perceptron (MLP)
def runMLP():
    text.delete('1.0', END)
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    calculateMetrics("MLP Classifier", y_pred)

# Function to run Random Forest
def runRF():
    global predict_cls
    text.delete('1.0', END)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    predict_cls = rf  # Save model for prediction
    calculateMetrics("Random Forest", y_pred)

# Function to predict on new data
def predict():
    global predict_cls
    text.delete('1.0', END)
    
    if predict_cls is None:
        text.insert(END, "Please train Random Forest before predicting!\n")
        return
    
    filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    
    if not filename:
        text.insert(END, "No file selected!\n")
        return
    
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace=True)
    X_new = dataset.iloc[:, 4:-2].values
    X_new = normalize(X_new)
    
    predictions = predict_cls.predict(X_new)

    for i, pred in enumerate(predictions):
        result = "NORMAL" if pred == 0 else "THYROID DISORDER"
        text.insert(END, f"Test Data {i+1}: Predicted as {result}\n")

# GUI Elements
font = ('times', 14, 'bold')
title = Label(main, text='Detection of Thyroid Disorders Using Machine Learning', bg='greenyellow', fg='dodger blue', font=font, height=2, width=100)
title.pack()

text = Text(main, height=20, width=120, font=('times', 12))
text.pack()

btn_font = ('times', 12, 'bold')

btn1 = Button(main, text="Upload & Preprocess Dataset", command=uploadDataset, font=btn_font)
btn1.pack(pady=5)

btn2 = Button(main, text="Train/Test Split", command=trainTest, font=btn_font)
btn2.pack(pady=5)

btn3 = Button(main, text="Run Logistic Regression", command=runLogisticRegression, font=btn_font)
btn3.pack(pady=5)

btn4 = Button(main, text="Run MLP Classifier", command=runMLP, font=btn_font)
btn4.pack(pady=5)

btn5 = Button(main, text="Run Random Forest", command=runRF, font=btn_font)
btn5.pack(pady=5)

btn6 = Button(main, text="Predict Disease", command=predict, font=btn_font)
btn6.pack(pady=5)

main.mainloop()
