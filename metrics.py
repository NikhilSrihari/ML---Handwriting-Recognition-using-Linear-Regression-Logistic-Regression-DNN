import numpy as np
import math


def root_mean_square_error(idealValues0, predictedValues):
	idealValues = np.array(idealValues0.copy())
	sum = np.sum(np.square(np.subtract(idealValues, predictedValues)))
	return math.sqrt( sum/len(predictedValues) )


def accuracy_score(idealValues0, predictedValues):
	idealValues = np.array(idealValues0.copy())
	numOfEntries = len(idealValues)
	correctPredictionsCnt = 0
	i=0
	while(i<numOfEntries):
		if (idealValues[i]==predictedValues[i]):
			correctPredictionsCnt=correctPredictionsCnt+1
		i=i+1
	return ((correctPredictionsCnt/numOfEntries)*100)


'''
	tp / (tp + fn)
'''
def recall_score(idealValues0, predictedValues):
	idealValues = np.array(idealValues0.copy())
	numOfEntries = len(idealValues)
	tp = 0; tn = 0
	fn = 0; fp = 0
	i=0
	while(i<numOfEntries):
		if (idealValues[i]==predictedValues[i]):
			if (idealValues[i]==1):
				tp=tp+1
			else:
				tn=tn+1
		else:
			if (idealValues[i]==1):
				fn=fn+1
			else:
				fp=fp+1
		i=i+1
	if ((tp + fn)==0):
		print("Couldnt calculate precision score as the denominator, tp+fn, was 0")
		return -1
	else:
		return ((tp / (tp + fn))*100)


'''
	tp / (tp + fp)
'''
def precision_score(idealValues0, predictedValues):
	idealValues = np.array(idealValues0.copy())
	numOfEntries = len(idealValues)
	tp = 0; tn = 0
	fn = 0; fp = 0
	i=0
	while(i<numOfEntries):
		if (idealValues[i]==predictedValues[i]):
			if (idealValues[i]==1):
				tp=tp+1
			else:
				tn=tn+1
		else:
			if (idealValues[i]==1):
				fn=fn+1
			else:
				fp=fp+1
		i=i+1
	if ((tp + fp)==0):
		print("Couldnt calculate precision score as the denominator, tp+fp, was 0")
		return -1
	else:
		return ((tp / (tp + fp))*100)