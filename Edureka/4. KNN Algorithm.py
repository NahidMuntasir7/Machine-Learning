# KNN: used in  for simple recommendation systems, pattern recognition, data mining, financial market predictions, intrusion detection, and more.
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
    dist = euclideanDistance(testInstance, trainingSet[x], length)
    distances.append(trainingSet[x], dist)

  distances.sort(key=operator.itemgetter(1))
  neighbors = []
  for x in range(k):
    neighbors.append(distances[x][0])
  
  return neighbors

# step 4: get response: generate a response from a set of data instance

import operator
def getResponse(neighbors):
  classVotes = {} 
  for x in range(len(neighbors)):
    response = neighbors[x][-1]
    if response in classVotes:
      classVotes[response] += 1
    else:
      classVotes[response] = 1
  sortedVotes = sorted(classvotes.iteritems(), key=operator.itemgetter(1), reverse = True)
  return sortedVotes[0][0]


# step 5: accuracy
def getAccuracy(testSet, predictions):
  correct = 0
  for x in range(len(testSet)):
    if testSet[x][-1] is predictions[x]:
      correct += 1
  return (correct / float(len(testSet))) * 100.0


# main

def main():
  trainingSet = []
  testSet = []
  split = 0.67
  loadDataset('iris.data', split, trainingSet, testSet)
  prediction = []
  k = 3
  for x in range(len(testSet)):
    neighbors = getNeighbors(trainingSet, testSet[x], k)
    result = getResponse(neighbors)
    predictions.append(result)
    print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
  accuracy = getAccuracy(testSet, predictons)
  print('Accuracy: ' + repr(accuracy) + '%')

main()
