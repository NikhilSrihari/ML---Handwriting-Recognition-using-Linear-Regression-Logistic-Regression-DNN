import pandas as pd
import numpy as np
import math
from metrics import root_mean_square_error


class LogisticRegression:

	def __init__(self):
		self.numberOfFeatures = None
		self.numberOfSamplesInTrainingSet = None
		self.finalWeights = None
		self.learningRate = None
		self.numberOfIterations = None

	
	def __loss(self, idealValues, predictedValues):
		return (-idealValues * np.log(predictedValues) - (1 - idealValues) * np.log(1 - predictedValues)).mean()
	

	def __predict(self, testingData_features, currentWeights=None):
		if (str(type(currentWeights))=="<class 'numpy.ndarray'>"):
			weightsToUse = currentWeights
		else:
			weightsToUse = self.finalWeights
		sum = (np.dot(np.transpose(weightsToUse), testingData_features))
		return (1 / (1 + np.exp(-1*sum)))


	def __fit_SGD(self, trainingData_features, trainingData_target):
		# Adding the Bias feature, always set to 1
		trainingData_features = np.concatenate(( (np.ones((1, self.numberOfSamplesInTrainingSet))), trainingData_features), axis=0)
		self.numberOfFeatures = self.numberOfFeatures + 1
		SGDIterationDetails = []
		W_current = np.zeros(self.numberOfFeatures)
		for i in range(0, self.numberOfIterations ):
			trainingData_predictedTarget = self.__predict(trainingData_features, W_current)
			loss = self.__loss(trainingData_target, trainingData_predictedTarget)
			training_rms_error = root_mean_square_error(trainingData_target, trainingData_predictedTarget)
			SGDIterationDetails.append({"W_current": W_current, "loss": loss, "training_rms_error": training_rms_error})
			if(self.ERMS_threshold!=None and self.ERMS_threshold>training_rms_error):
				break
			elif(self.loss_threshold!=None and self.loss_threshold>loss):
				break
			delta = (-1/self.numberOfSamplesInTrainingSet) * (np.dot(trainingData_features, np.transpose(np.subtract(trainingData_predictedTarget, trainingData_target))))
			W_current = W_current + np.dot(self.learningRate, delta)
		bestIterationDetails = min(SGDIterationDetails, key = lambda x: x["training_rms_error"])
		self.finalWeights = bestIterationDetails["W_current"]


	def fit(self, trainingData_features0, trainingData_target0, learningRate, numberOfIterations, ERMS_threshold=None, loss_threshold=None):
		trainingData_features = np.array(trainingData_features0.copy())
		trainingData_target = np.array(trainingData_target0.copy())
		self.numberOfFeatures = len(trainingData_features)
		self.numberOfSamplesInTrainingSet = len(trainingData_features[0])
		print(trainingData_features.shape)
		print(trainingData_target.shape)
		print("ss")
		self.numberOfIterations = numberOfIterations
		self.learningRate = learningRate
		self.ERMS_threshold = ERMS_threshold
		self.loss_threshold = loss_threshold
		self.__fit_SGD(trainingData_features, trainingData_target)


	def predict(self, testingData_features0):
		testingData_features = np.array(testingData_features0.copy())
		numberOfSamplesInTestingSet = len(testingData_features[0])
		testingData_features = np.concatenate((np.ones((1, numberOfSamplesInTestingSet)), testingData_features), axis=0)
		numberOfSamplesInTestingSet=numberOfSamplesInTestingSet+1
		return self.__predict(testingData_features)


	def mapRegressionOutputToBinaryValues(self, testingData_predictedTarget, sigmoidThreshold):
		testingData_predictedTarget1 = np.array(testingData_predictedTarget.copy())
		i=0
		while(i<len(testingData_predictedTarget1)):
			if (testingData_predictedTarget1[i] <= sigmoidThreshold):
				testingData_predictedTarget1[i] = 0.0
			else:
				testingData_predictedTarget1[i] = 1.0
			i=i+1
		return testingData_predictedTarget1