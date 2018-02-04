'''
Yang
Modified Feb 2, 2018
Decision Tree 
'''
import math
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']                       #change to discrete values
    return dataSet, labels

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:                                 
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): 
            labelCounts[currentLabel] = 1
        else:
            labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries          #get the probability value of all the labels
        shannonEnt -= prob * math.log2(prob)                   #log base 2
    return shannonEnt
    
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]                #chop out axis used for splitting, so we don't care about the value used for splitting later.
            reducedFeatVec.extend(featVec[axis+1:])        #extend is to cancatenate the two lists
            retDataSet.append(reducedFeatVec)
    return retDataSet
    
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1                      #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):                           #iterate over all the features
        featList = [example[i] for example in dataSet]     #create a list of all the examples of this feature i
        uniqueVals = set(featList)                         #get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)   #based on this feature to split the dataset
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy                #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):                      #compare this to the best gain so far
            bestInfoGain = infoGain                        #if better than current best, set to best
            bestFeature = i
    return bestFeature                                     #returns an integer

def majorityCnt(classList):                                #when we have iterated all the features but still cannot get the unique decision in one class, we use major count to determine the classfication
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): 
            classCount[vote] = 0
        else:
            classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=lambda x:x[1], reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): 
        return classList[0]                               #stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1:                              #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])                                 #not use this feature any more.
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]                             #copy all of labels, so trees don't mess up existing labels, due to the way python calls lists.
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                            
    
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):                     # whether valueofFrat is a dict or not 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: 
        classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    
fopen=open("C:\\Users\\kevin\\Desktop\\machinelearninginaction-master\\Ch03\\lenses.txt")
lenses =[inst.strip().split('\t') for inst in fopen.readlines()]
lenseLabels = ['age','prescript','astigmatic','tearRate']
tree = createTree(lenses,lenseLabels)
print (tree)    #you should get this result 
                #{'tearRate': {'reduced': 'no lenses', 'normal': {'astigmatic': {'yes': {
                # 'prescript': {'hyper': {'age': {'young': 'hard', 'presbyopic': 'no lenses', 'pre': 'no lenses'
                # }}, 'myope': 'hard'}}, 'no': {'age': {'young': 'soft', 'presbyopic': {
                # 'prescript': {'hyper': 'soft', 'myope': 'no lenses'}}, 'pre': 'soft'}}}}}}