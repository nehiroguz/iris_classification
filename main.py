import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = datasets.load_iris()

data = pd.DataFrame(data= iris.data, columns= iris.feature_names)
data['target'] = iris.target

X = data.iloc[:, :-1] 
Y = data['target']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.3, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test) 

lr = LogisticRegression() 
lr.fit(x_train, y_train) 

predictions = lr.predict(x_test)

accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)
report = classification_report(y_test, predictions)

print(f'Accuracy score: {accuracy}')
print(f'Confusion matrix:\n {cm}')
print(f'Clssification report:\n {report}')