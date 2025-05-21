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
```

## Output:
![Screenshot 2025-05-21 105958](https://github.com/user-attachments/assets/834e1a89-fd6b-4f3f-842c-bd9438024b01)
![Screenshot 2025-05-21 110123](https://github.com/user-attachments/assets/4e05738e-2444-4584-b2ee-8f46e247d516)
![Screenshot 2025-05-21 110223](https://github.com/user-attachments/assets/d34c6180-60f1-4c1b-994c-86604686d125)
![Screenshot 2025-05-21 110314](https://github.com/user-attachments/assets/bfe5e5d0-8187-476b-9250-24ea69798364)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
