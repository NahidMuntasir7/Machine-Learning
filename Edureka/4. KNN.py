
import csv
import random


# splits dataset into training and testing data from a csv file
def loadDataset(filename, split, trainingSet = [], testSet = []):
  with open(filename, 'r') as csvfile:
    lines = csv.reader(csvfile)
    dataset = list(lines)

    for x in range(len(dataset)-1):
      for y in range(4):
        dataset[x][y] = float(dataset[x][y])
      if random.random() < split:
        trainingSet.append(dataset[x])
      else:
        testset.append(dataset[x])

# step 1: handle data

# loading
trainingSet = []
testSet = []
loadDataset(r'iris.data', 0.66, trainingSet, testSet)



# step 2: similarity : using euclidean distance
import math

def euclideanDistance(instance1, instance2, length):
  distance = 0
  for x in range(length):
    distance += pow((instanxe[x] - instance[x]), 2)
  return math.sqrt(distance)



# step 3: look for k nearest neighbors
import operator
def getNeighbors(trainingSet, testTnstance, k):
  distances = []
  length = len(testInstances) - 1

  for x in range(len(trainingSet)):
    dist = euclideanDistance(
