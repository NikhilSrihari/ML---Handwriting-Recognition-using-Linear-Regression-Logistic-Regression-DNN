import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import math
from metrics import root_mean_square_error


class LinearRegression:

	learningType = None
	M = None
	lambdaa = None
	BigSigma_scalingFactor = None

	BigSigma_OffsetFactor = None
	numberOfFeatures = None
	numberOfSamplesInTrainingSet = None
	SmallMuList = None
	BigSigma = None
	BigSigma_Inverse = None
	finalWeights = None


	def __init__(self, learningType, M, lambdaa, BigSigma_scalingFactor):
		self.learningType = learningType
		self.M = M
		self.lambdaa = lambdaa
		self.BigSigma_scalingFactor = BigSigma_scalingFactor
		self.BigSigma_OffsetFactor = 0.2
		self.numberOfFeatures = None
		self.numberOfSamplesInTrainingSet = None
		self.SmallMuList = None
		self.BigSigma = None
		self.BigSigma_Inverse = None
		self.finalWeights = None


	def __calculateSmallMu(self, trainingData_features):
		kMeansCluster = KMeans(n_clusters=self.M, random_state=0).fit(np.transpose(trainingData_features))
		return kMeansCluster.cluster_centers_


	def __calculateBigSigma(self, trainingData_features):
		BigSigma = np.zeros((self.numberOfFeatures, self.numberOfFeatures))
		i=0
		while(i<self.numberOfFeatures):
			BigSigma[i][i]=np.var(trainingData_features[i])+self.BigSigma_OffsetFactor
			i=i+1
		if (self.BigSigma_scalingFactor==None or self.BigSigma_scalingFactor==1):
			return BigSigma
		else:
			return np.dot(self.BigSigma_scalingFactor, BigSigma)
	    	

	def __calculateSmallPhi(self, sample, j):
		SmallMu = self.SmallMuList[j]
		diff = np.subtract(sample, SmallMu)
		return math.exp( -0.5 * np.dot( (np.dot( np.transpose(diff), self.BigSigma_Inverse )), diff ) )


	def __calculateBigPhi(self, trainingData_features):
		BigPhi = np.ones([self.numberOfSamplesInTrainingSet, self.M])
		i=0
		while(i<self.numberOfSamplesInTrainingSet):
			sample = trainingData_features[:,i]
			j=1
			while(j<self.M):
				BigPhi[i][j] = self.__calculateSmallPhi(sample, j)
				j=j+1
			i=i+1
		return BigPhi


	def __calculateBigPhiStar(self, BigPhi):
		BigPhi_transpose = np.transpose(BigPhi)
		return np.dot( np.linalg.inv( np.add( (np.dot(self.lambdaa,np.identity(self.M))), (np.dot(BigPhi_transpose,BigPhi)) ) ), BigPhi_transpose )


	def __calculateFinalWeights(self, BigPhiStar, trainingData_target):
		return np.dot( BigPhiStar, trainingData_target )


	def __fit_SGD(self, trainingData_features, trainingData_target, SGDParameters):
		if (SGDParameters==None):
			print("SGD Parameters missing")
			raise Exception("SGD Parameters missing")
		SGDIterationDetails = []
		W_current = self.finalWeights
		for i in range(0, SGDParameters["numberOfIterationsInSGDLoop"] ):
			sample = trainingData_features[:,i]
			#Calculating the smallPhiArray
			smallPhiArray = np.ones(self.M)
			j=1
			while(j<self.M):
				smallPhiArray[j] = self.__calculateSmallPhi(sample, j)
				j=j+1
			#smallPhiArray calculated. Starting with deltaED now....
			deltaED = np.dot( -1 * ( (trainingData_target[i]) - (self.__calculateOutput(sample, W_current)) ) , smallPhiArray)
			deltaEW = W_current
			deltaE = np.add(deltaED, np.dot(SGDParameters["lambda"], deltaEW))
			deltaW = np.dot( (-1 * SGDParameters["learningRate"]), deltaE )
			W_current = np.add(W_current, deltaW)
			#Calculating the ERMS for this W value
			training_rms_error = root_mean_square_error(trainingData_target, self.predict(trainingData_features, W_current))
			if (i%20==0):
				print(i)
				print(deltaE)
				print(training_rms_error)
				print()
			SGDIterationDetails.append({"training_rms_error": training_rms_error, "W_current": W_current, "i": i})
			if "ERMS_threshold" in SGDParameters:
				if(SGDParameters["ERMS_threshold"]!=None and SGDParameters["ERMS_threshold"]>training_rms_error):
					break
		bestIterationDetails = min(SGDIterationDetails, key = lambda x: x["training_rms_error"])
		print(str(bestIterationDetails["i"])+" "+str(bestIterationDetails["training_rms_error"]))
		self.finalWeights = bestIterationDetails["W_current"]

	
	def __fit_ClosedForm(self, trainingData_features, trainingData_target):
		self.SmallMuList = self.__calculateSmallMu(trainingData_features)
		self.BigSigma = self.__calculateBigSigma(trainingData_features)
		self.BigSigma_Inverse = np.linalg.inv(self.BigSigma)
		BigPhi = self.__calculateBigPhi(trainingData_features) 
		BigPhiStar = self.__calculateBigPhiStar(BigPhi)
		self.finalWeights = self.__calculateFinalWeights(BigPhiStar, trainingData_target)
		

	def fit(self, trainingData_features0, trainingData_target0, SGDParameters=None):
		trainingData_features = np.array(trainingData_features0.copy())
		trainingData_target = np.array(trainingData_target0.copy())
		self.numberOfFeatures = len(trainingData_features)
		self.numberOfSamplesInTrainingSet = len(trainingData_features[0])
		if(self.learningType=="SGD"):
			self.__fit_ClosedForm(trainingData_features, trainingData_target)
			self.__fit_SGD(trainingData_features, trainingData_target, SGDParameters)
		elif (self.learningType=="CLOSED_FORM"):
			self.__fit_ClosedForm(trainingData_features, trainingData_target)


	def __calculateOutput(self, features, currentWeights=None):
		smallPhiArray = np.transpose(np.ones(self.M))
		j=1
		while(j<self.M):
			smallPhiArray[j] = self.__calculateSmallPhi(features, j)
			j=j+1
		if (str(type(currentWeights))=="<class 'numpy.ndarray'>"):
			weightsToUse = currentWeights
		else:
			weightsToUse = self.finalWeights
		return np.dot(np.transpose(weightsToUse),smallPhiArray)


	def predict(self, testingData_features0, currentWeights=None):
		testingData_features = np.array(testingData_features0.copy())
		outputs=[]
		i=0
		while(i<len(testingData_features[0])):
			sample = testingData_features[:,i]
			if (str(type(currentWeights))=="<class 'numpy.ndarray'>"):
				outputs.append(self.__calculateOutput(sample, currentWeights))
			else:
				outputs.append(self.__calculateOutput(sample))
			i=i+1
		return np.array(outputs)


	def print(self):
		print(self.M)
		print(self.lambdaa)
		print(self.BigSigma_scalingFactor)
		print(self.finalWeights)


	def printFinalWeights(self):

		print(self.finalWeights)