# a car company has released a new SUV in the market. using the previous data about the sales of their SUV's, 
# they want to predict the category of people who might be interested in buying this...

# what factors made people more interested in buying SUV? 
# will a particular person buy that car or not?

# purchased is the discrete column / categorical column / y


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

dataset = pd.read_csv("SUV_Prediction.csv")
dataset.head(10) # first 10 row


# skipping step 2 and 3


# step 4: Train data

# iloc - pandas
X = dataset.iloc[:,[2, 3]].values   # independent variable  # taking only 2nd and 3rd column (age and salary)
y = dataset.iloc[:,4]  # dependent variable # the 4th column (purchased)


from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0) # 75 25 ratio  # random state - using same sample in each run


from sklearn.preprocessing import StandardScaler 

# scale down for test
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


# accuracy

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred) * 100 # 89%

#   :)
