#****************************************************************************************
# Homework Seminar 02: Dec 07 2019
# Implement DecisionTree
# 
# Nguyen Dang Cuong  -  1811640
#
#
#
#****************************************************************************************

import numpy as np
import pandas as pd
import random as rd

DATA_PATH = 'data.csv'
SIZE_OF_TRAINING_SET = 320
DEFAULT_LABEL = 0
EPSILON = 3
MIN_SIZE = 10


def binSet(listValue):
    listValue = list(listValue)
    s = set(listValue)
    if len(s)==2: return s
    for item in s:
        try:
            int(item)
        except:
            #if type of item is non-numeric -> return s
            return s
    
    return (0,1,2)

def getBin(value,listValue):

    listValue = list(listValue)
    s = set(listValue)
    if len(s) == 2: return value
    
    try:
        L = list(s)
        L.sort()

        a0 = int(L[0])
        a3 = int(L[-1])
        a1 = a0 + (a3 - a0) // 3
        a2 = a0 + (a3 - a0) // 3 * 2
        if value<a1: return 0
        elif value<a2: return 1
        else: return 2
    except:
        #if value is non-numeric: do nothing
        return value

def modify(data):
    for column in list(data)[1:]:
        for i in range(len(data[column])):
            data[column][i] = getBin(data[column][i],data[column])
    return data

def entropy(attribute):
    if type(attribute) != list: attribute = list(attribute)
    N = len(attribute)
    s = set(attribute)
    E = 0
    for i in s:
        Ni = 0
        for j in attribute:
            if j == i: Ni = Ni + 1
        P = Ni/N
        #assume that 0log(0) = 0
        if P != 0: E = E + P*np.log(P)
    return -E

def isLeafNode(listLabel):
    """return label if afford"""

    if len(listLabel) < MIN_SIZE: return DEFAULT_LABEL
    listLabel = list(listLabel)
    s = set(listLabel)
    count = []
    for i in s:
        _count = 0
        for j in listLabel: 
            if j == i: _count = _count + 1
        count.append((_count,i))
    count.sort()
    count.reverse()
    if count[0][0] >= len(listLabel) - EPSILON: return count[0][1]
    return -1

def buildTree(train):
    if len(train.columns) == 2: return DEFAULT_LABEL  #include column 'index'

    tree = {}
    #find root Node (Entropy Min)
    list_entropy = []
    for i in list(train)[1:-1]:
        list_entropy.append((entropy(train[i]),i))
    list_entropy.sort()
    rootNode = list_entropy[0][1]
    
    #insert new node to decision tree
    tree.setdefault(rootNode,{})
    s = binSet(train[rootNode])

    for item in s:
        label = isLeafNode(train[train[rootNode] == item]['Purchased'])        
        if label!=-1: 
            #leaf node
            tree[rootNode][item] = label
        else:
            #recurse: build tree from smaller training set 
            newTrain = train[train[rootNode] == item].drop(columns = rootNode)
            tree[rootNode][item] = buildTree(newTrain)

    return tree

#================================================================================================
#main function

#load data
data = pd.read_csv(DATA_PATH)
data = modify(data)

#split to training set & testing set
train = pd.DataFrame(columns = ['User ID','Gender','Age','EstimatedSalary','Purchased'])
isExist = []
while len(isExist) < SIZE_OF_TRAINING_SET:
    idx = rd.randint(0,len(data) - 1)
    newID = data.loc[idx]['User ID']
    if (newID not in isExist):
        isExist.append(newID)
        newItem = data.loc[idx]
        train = train.append(newItem)
for i in range(len(data) - 1):
    id = data.loc[i]['User ID']
    if id in isExist:
        data = data.drop(index = i)

data = data.reset_index()
data = data.drop(columns = 'index')

test = data


#build decision tree from training set

train = train.reset_index()
train = train.drop(columns = 'index')
_tree = buildTree(train)
print(_tree)


#testing
test = test.reset_index()
test = test.drop(columns = 'index')
count = 0
for i in range(len(test)):
    item = test.loc[i]
    result = int(item['Purchased'])
    tree = _tree 
    while type(tree) == dict :
        attribute = list(tree)[0]
        tree = tree[attribute][item[attribute]]
    if tree == result: count = count + 1
out = round(count/len(test) * 100,2)
print(str(round(count/len(test) * 100,2)) + '%')