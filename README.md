# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to preditct the values.
4. Find the accuracy and display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Surjith D
RegisterNumber: 212223043006
*/
```
```
import chardet
file = 'spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
print("Detected Encoding:", result)

import pandas as pd
data = pd.read_csv("spam.csv", encoding='windows-1252')

print(data.head())
print(data.info())
print("Missing values:\n", data.isnull().sum())

x = data["v2"].values
y = data["v1"].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
## Output:
![SVM For Spam Mail Detection](sam.png)
```
## Output
![Screenshot 2025-05-21 105958](https://github.com/user-attachments/assets/c6b51da6-db4b-4ef8-896f-2bc1ff5144f0)
![Screenshot 2025-05-21 110123](https://github.com/user-attachments/assets/c2091a2a-de84-4009-b2c9-5bb70ec5c438)
![Screenshot 2025-05-21 110314](https://github.com/user-attachments/assets/d2467366-a933-4f02-8432-bc5c7d63829a)
![Screenshot 2025-05-21 110314](https://github.com/user-attachments/assets/e4d13fa2-06b4-4aaa-997c-f60cee2daf5e)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
