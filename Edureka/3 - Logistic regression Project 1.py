# logistic regression : will predict survived or not - yes or no? 
# what factors made people more likely to survive of the titanic


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import math



# Step 1: collecting data

titanic_data = pd.read_csv("Titanic.csv")
titanic_data.head(10) # 1st 10 values

print("number of passengeres in original data: ", len(titanic_data.index))




# Step 2: analyzing data

sns.countplot(x = "Survived", data = titanic_data) # a graph of the number of both survived and died
sns.countplot(x = "Survived", hue = "Sex", data = titanic_data) # male female in both survived and died
sns.countplot(x = "Survived", hue = "Pclass", data = titanic_data) # classes

titanic_data["Age"].plot.hist() # graph of ages  # using pandas
titanic_data["Fare"].plot.hist(bin = 20, figsize = (10, 5))

titanic_data.info()

sns.countplot(x = "SibSp", data = titanic_data) # 




# Step 3: data wrangling: clean the data by removing the null and unnecessary columns


titanic_data.isnull().sum() # can see the numbers of nulls
# can also use the heatmap to see ..
sns.heatmap(titanic_data.isnull(), yticketlabels==False)
sns.boxplot(x = "Pclass", y = "Age", data=titanic_data)


titanic_data.head(5)

titanic_data.drop("Cabin", axis=1, inplace=True) # deleting cabin as max null
titanic_data.dropna(inplace=True) # non number values

# to check null values removed or not
sns.heatmap(titanic_data.isnull(), yticketlabels==False)
titanic_data.isnull().sum()


# further removing
# we dont need both male and female

sex = pd.get_dummies(titanic_data['Sex'], drop_first=True)
sex.head(5) # female removed

embarked = pd.get_dummies(titanic_data["Embarked"], drop_first=True)
pcl = pd.get_dummies(titanic_data["Pclass"], drop_first=True)


# concatenation
titanic_data = pd.concat([titanic_data, sex, embark, Pcl],axis=1)
titanic_data.head(5)


# again dropping
titanic_data.drop(['Sex', 'Embarked', 'PassengerId', 'Name', 'Ticket', 'Pclass'], axis = 1, inplace = True)
titanic_data.head(5) # final cleaned data




# Step 4: Train data


X = titanic_data.drop("Survived", axis = 1) # independent variables
y = titanic_data["Survived"] # this is to determine - dependent variable


# splitting data in training and testing subsets
# for that using sklearn

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70 30 ratio

from sklearn.linear_model import LogisticRegression

logmodel =  LogisticRegression()
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)




# step 5: Accuracy check

# now performance

from sklearn.metrics import classification_report
classification_report(y_test, predictions)


# confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)

'''
array([[105, 21]        PN  PY   PY/N - predicted yes/no
       [25, 63]])    AN  .  .    AY/N - actual yes/no
                     AY  .  .
'''

from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions) # 0.7850.....
