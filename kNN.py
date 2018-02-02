'''
Yang
Modified on Feb 1st, 2018
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            data: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

Note: Load data in windows please better describe the path using '\\' rather than'\'

'''
from numpy import *
from os import listdir                          #to load a list of training file

def classify0(inX, data, labels, k):
    datasize = data.shape[0]
    diffMat = tile(inX, (datasize,1)) - data     #use tile to form the matrix for KNN algorithm
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)          #axis = 1 means row-wise
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()     
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        if voteIlabel in classCount:
            classCount[voteIlabel] += 1
        else:
            classCount[voteIlabel] = 0
    sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)  #here we use lambda to define a function rather than import operator
    return sortedClassCount[0][0]               #return the one with the largest number

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def file2matrix(filename):
    love_dictionary={'largeDoses':3, 'smallDoses':2, 'didntLike':1}
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)            #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return, use np.zeros to create a numberOfLines*3 matrix
    classLabelVector = []                       #prepare labels return   
    index = 0
    for line in arrayOLines:
        line = line.strip()                    #remove the front and back ' ' in line
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        if(listFromLine[-1].isdigit()):
            classLabelVector.append(int(listFromLine[-1]))
        else:
            classLabelVector.append(love_dictionary[listFromLine[-1]])
        index += 1
    return returnMat,classLabelVector

    
def autoNorm(dataSet):
    minVals = dataSet.min(0)                       # 0 means get minimum for each column
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide in numpy
    return normDataSet, ranges, minVals
   
def datingClassTest():
    hoRatio = 0.10                                                       #hold out 10%, so this 10% are used for testing
    datingDataMat,datingLabels = file2matrix('C:\\Users\\kevin\\Desktop\\machinelearninginaction-master\\Ch02_KNN\\datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print ("the classifier came back with: %d, the real answer is: %d" %(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]: 
            errorCount += 1
    print ("the total error rate is: %f" % errorCount/numTestVecs)
    print (errorCount)
    
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input(\
                                  "percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('C:\\Users\\kevin\\Desktop\\machinelearninginaction-master\\Ch02_KNN\datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream, ])
    classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)
    print ("You will probably like this person: %s" % resultList[classifierResult - 1])  #because resultList starts from list index 0

classifyPerson()
    
def img2vector(filename):
    returnVect = zeros((1,1024))                 #a 1*1024 matrix for use, since the image is a 32*32 vector
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('C:\\Users\\kevin\\Desktop\\machinelearninginaction-master\\Ch02_KNN\\trainingDigits')           #load the training set, the path need to modify based on your own folder path
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('C:\\Users\\kevin\\Desktop\\machinelearninginaction-master\\Ch02_KNN\\trainingDigits/%s' % fileNameStr)
    testFileList = listdir('C:\\Users\\kevin\\Desktop\\machinelearninginaction-master\\Ch02_KNN\\testDigits')        #iterate through the test set
    errorCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('C:\\Users\\kevin\\Desktop\\machinelearninginaction-master\\Ch02_KNN\\testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if classifierResult != classNumStr: 
            errorCount += 1
    print ("\nthe total number of errors is: %d" % errorCount)
    print ("\nthe total error rate is: %f" % (errorCount/mTest))

handwritingClassTest()               #I get a test error rate 0.015071