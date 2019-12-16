from pandas import read_csv, DataFrame, Series
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RandomForestClassifier
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import mlab

from sklearn.model_selection import train_test_split
train_dataset = read_csv('train.csv')
test_dataset = read_csv('test.csv')
train_dataset.info()
test_dataset.info()
dep_var = train_dataset["Survived"] #зависимая переменная
train_dataset.drop(['Survived', 'Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
#test_dataset.drop(['Survived', 'Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
#нормализация данных
def harmonize_data(titanic):
    #отсутствующим полям возраста присваивается медианное значение
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
    titanic["Age"].median()
    #пол преобразуется в числовой формат
    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
    #пустое место отплытия заполняется наиболее популярным S
    titanic["Embarked"] = titanic["Embarked"].fillna("S")
    # место отплытия преобразуется в числовой формат
    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
    # отсутствующим полям суммы отплаты за плавание присваивается медианное значение
    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())
    return titanic

train_harm = harmonize_data(train_dataset)
test_harm  = harmonize_data(test_dataset)
train_harm.info()
test_harm.info()
y = test_harm['Survived']
clf = RandomForestClassifier()
param_grid = {}

gs = GridSearchCV(clf,return_train_score=True, param_grid=param_gridBN, scoring=scoringBN, cv=10, refit='Accuracy')
gs.fit(test_harm, y)
results = gs.cv_results_

print('=' * 100)
print("best params: " + str(gs.best_estimator_))
print("best params: " + str(gs.best_params_))
print('best score:', gs.best_score_)
print('=' * 100)