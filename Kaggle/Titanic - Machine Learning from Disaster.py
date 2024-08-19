import pandas as pd

from google.colab import files
uploaded = files.upload()   // train

from google.colab import files
uploaded = files.upload()  // test


data = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
test_ids = test["PassengerId"]
data.head()


def clean(data):
  data = data.drop(["Ticket", "Cabin", "Name", "PassengerId"], axis=1)
  cols = ["SibSp", "Parch", "Fare", "Age"]
  for col in cols:
    data[col].fillna(data[col].median(), inplace=True)
  data.Embarked.fillna("U", inplace=True)
  return data

data = clean(data)
test = clean(test)
data.head(5)


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
columns = ["Sex", "Embarked"]

for col in columns:
  data[col] = le.fit_transform(data[col])
  test[col] = le.transform(test[col])
  print(le.classes_)
data.head(5)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = data.drop("Survived", axis=1)
Y = data["Survived"]

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, Y_train)


predictions = clf.predict(X_val)
from sklearn.metrics import accuracy_score
accuracy_score(Y_val, predictions)


submission_preds = clf.predict(test)


df = pd.DataFrame({"PassengerId": test_ids.values,
                  "Survived": submission_preds,
                  })


df.to_csv("submission.csv", index=False)

from google.colab import files
files.download('submission.csv')








