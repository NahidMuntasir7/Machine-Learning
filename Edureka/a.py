
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

print("Hello")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names = names) # dataset loaded


print(dataset.shape) # (150, 5) row and column
print(dataset.head(10)) # printing first 10 results
print(dataset.describe()) # summary of each attribute
print(dataset.groupby('class').size()) # grouping by classes   # class Iris-setosa 50 | Iris-versicolor 50 | Iris-virginica 50



# visualization
dataset.plot(kind = 'box', subplots = True, layout = (2, 2), sharex = False, sharey = False)
plt.show()
dataset.hist()
plt.show()
scatter_matrix(dataset)
plt.show()



# validation dataset = training dataset

# 80% data will be used to train the model and the rest 20% will be used to verify the model

array = dataset.values
X = array[:,0:4] # all the column starting from 0 to 4
Y = array[:,4] # just need the 4th column which is the class column
validation_size = 0.20
seed = 6

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = validation_size, random_state = seed)

seed = 6
scoring = 'accuracy'


models = []

models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('LDA', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []

for name, model in models:
  kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)
  cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv = kfold, scoring = scoring)
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  print(msg)

'''
LR: 0.958333 (0.041667)
LDA: 0.975000 (0.038188)
KNN: 0.958333 (0.041667)
LDA: 0.941667 (0.053359)
NB: 0.966667 (0.040825)
SVM: 0.941667 (0.053359)
'''

