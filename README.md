# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and create/load the dataset containing student details such as CGPA, aptitude score, and placement status.
2. Split the dataset into training and testing sets and standardize the input features.
3. Train the Logistic Regression model using the training data and predict the placement status for the test data.
4. Display the predicted results, calculate the model accuracy, and predict the placement status for a new student.


## Program:
```

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix

# Sample Dataset
# Features: [CGPA, Aptitude Score]
X = np.array([
    [6.5, 60],
    [7.0, 65],
    [7.5, 70],
    [8.0, 75],
    [8.2, 80],
    [8.5, 85],
    [9.0, 90],
    [6.8, 62],
    [7.8, 74],
    [8.8, 88]
])

# Target: 0 = Not Placed, 1 = Placed
y = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 1])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# Logistic Regression Model
model = LogisticRegression()

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Actual Placement Status:")

print(y_test)

print("\nPredicted Placement Status:")

print(y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, y_pred))

# Predict for a new student
new_student = np.array([[8.3, 82]])

new_student = scaler.transform(new_student)

prediction = model.predict(new_student)

if prediction[0] == 1:

    print("\nPrediction: Student is Placed")
else:

    print("\nPrediction: Student is Not Placed")

```

## Output:
<img width="293" height="226" alt="image" src="https://github.com/user-attachments/assets/3fc3a90c-1f91-433e-8c92-a0ae664133cf" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
