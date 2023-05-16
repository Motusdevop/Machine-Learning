# Здесь должен быть твой код

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv("titanic.csv")

df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis = 1, inplace = True)

df["Embarked"].fillna("S", inplace = True)

def fill_age(row):
    if pd.isnull(row["Age"]):
        if row["Pclass"] == 1:
            return age_1
        if row["Pclass"] == 2:
            return age_2
        if row["Pclass"] == 3:
            return age_3
    return row["Age"]

def fill_sex(sex):
    if sex == "male":
        return 1
    return 0

age_1 = df[df["Pclass"] == 1]["Age"].median()
age_2 = df[df["Pclass"] == 2]["Age"].median()
age_3 = df[df["Pclass"] == 3]["Age"].median()


df["Age"] = df.apply(fill_age, axis = 1)

df[list(pd.get_dummies(df["Embarked"]).columns)] = pd.get_dummies(df["Embarked"])

df.drop("Embarked", axis = 1, inplace = True)

df["Sex"] = df["Sex"].apply(fill_sex)

x = df.drop("Survived", axis = 1)
y = df["Survived"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

classifier = KNeighborsClassifier(n_neighbors= 12)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

precent = accuracy_score(y_test, y_pred) * 100

print(precent)