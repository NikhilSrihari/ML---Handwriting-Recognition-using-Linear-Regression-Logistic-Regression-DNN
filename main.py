import csv
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.externals import joblib
from matplotlib import pyplot as plt
from metrics import precision_score, recall_score, accuracy_score, root_mean_square_error
from linearRegression import LinearRegression
from logisticRegression import LogisticRegression
from neuralNetwork import NeuralNetwork


fileBasePath = './'
maxHODataSetSize = maxGSCDataSetSize = 10000
samePairsToDiffPairsDatasetPercentage = 40
trainingDatasetPercentage = 75
validationDatasetPercentage = 10
testingDatasetPercentage = 100 - trainingDatasetPercentage - validationDatasetPercentage
totalHODataSetSize = None; totalGSCDataSetSize = None;
totalsamePairsHODataSetSize = None; totalsamePairsGSCDataSetSize = None;
totaldiffPairsHODataSetSize = None; totaldiffPairsGSCDataSetSize = None;


def readCSVFile(filePath):
	table = []
	with open(fileBasePath+filePath, 'rU') as file:
		csvReader = csv.reader(file)
		for row in csvReader:
			table.append(row)
	columnHeaders = table[0]
	tableContent = table[1:]
	j=0
	while(j<len(columnHeaders)):
		if(columnHeaders[j]==''):
			columnHeaders[j]='XXX'
		j=j+1
	return pd.DataFrame(tableContent, columns=columnHeaders, dtype=float)


def writeDFIntoCSVFile(df, fileName):
	df.to_csv(fileName+'.csv', encoding='utf-8', index=False)
	return 1


def reduceSizeAndMergeSameAndDiffPairs(Dataset_samePairs, Dataset_diffPairs, datasetType):
	global totalHODataSetSize, totalGSCDataSetSize, totalsamePairsHODataSetSize, totaldiffPairsHODataSetSize, totalsamePairsGSCDataSetSize,totaldiffPairsGSCDataSetSize
	maxDataSetSize = maxHODataSetSize if (datasetType=="HO") else maxGSCDataSetSize
	maxSamePairsSize =  (int) (maxDataSetSize * (samePairsToDiffPairsDatasetPercentage/100))
	maxDiffPairsSize = maxDataSetSize - maxSamePairsSize
	if ( (len(Dataset_samePairs.index) > maxSamePairsSize) and (len(Dataset_diffPairs.index) > maxDiffPairsSize) ):
		totSamePairsSize = maxSamePairsSize
		totDiffPairsSize = maxDiffPairsSize
	elif ( (len(Dataset_samePairs.index) > maxSamePairsSize) ):
		totDatasetSize =  (int) (100 * ( (len(Dataset_diffPairs.index)) / (100-samePairsToDiffPairsDatasetPercentage) ))
		totSamePairsSize =  totDatasetSize - (len(Dataset_diffPairs.index))
		totDiffPairsSize = (len(Dataset_diffPairs.index))
	elif ( (len(Dataset_diffPairs.index) > maxDiffPairsSize) ):
		totDatasetSize =  (int) (100 * ( (len(Dataset_samePairs.index)) / (samePairsToDiffPairsDatasetPercentage) ))
		totDiffPairsSize =  totDatasetSize - (len(Dataset_samePairs.index))
		totSamePairsSize = (len(Dataset_samePairs.index))
	else:
		totDatasetSize1 =  (int) (100 * ( (len(Dataset_diffPairs.index)) / (100-samePairsToDiffPairsDatasetPercentage) ))
		totSamePairsSize1 =  totDatasetSize1 - (len(Dataset_diffPairs.index))
		totDatasetSize2 =  (int) (100 * ( (len(Dataset_samePairs.index)) / (samePairsToDiffPairsDatasetPercentage) ))
		totDiffPairsSize2 =  totDatasetSize2 - (len(Dataset_samePairs.index))
		if (totSamePairsSize1<=(len(Dataset_diffPairs.index))):
			totSamePairsSize = totSamePairsSize1
			totDiffPairsSize = (len(Dataset_diffPairs.index))
		else:
			totSamePairsSize = (len(Dataset_samePairs.index))
			totDiffPairsSize = totDiffPairsSize2
	print("		"+datasetType+" Dataset: Total Size = "+str(totSamePairsSize+totDiffPairsSize)+" ; SamePairs Set Size = "+str(totSamePairsSize)+" ; DifferentPairs Set Size = "+str(totDiffPairsSize))
	if (datasetType=="HO"):
		totalsamePairsHODataSetSize = totSamePairsSize
		totaldiffPairsHODataSetSize = totDiffPairsSize
		totalHODataSetSize = totalsamePairsHODataSetSize + totaldiffPairsHODataSetSize
	else:
		totalsamePairsGSCDataSetSize = totSamePairsSize
		totaldiffPairsGSCDataSetSize = totDiffPairsSize
		totalGSCDataSetSize = totalsamePairsGSCDataSetSize + totaldiffPairsGSCDataSetSize
	Dataset_samePairs = Dataset_samePairs.sample(n=totSamePairsSize)
	Dataset_diffPairs = Dataset_diffPairs.sample(n=totDiffPairsSize)
	Dataset_pairs = pd.concat([Dataset_samePairs, Dataset_diffPairs])
	Dataset_pairs = Dataset_pairs.sample(n = len(Dataset_pairs.index))
	return Dataset_pairs


def mapRegressionOutputToBinaryValues(predictedOutputs):
	i=0
	while(i<len(predictedOutputs)):
		if (predictedOutputs[i] <= 0.5):
			predictedOutputs[i] = 0.0
		else:
			predictedOutputs[i] = 1.0
		i=i+1


def featureConcantenationAndSubtraction(Dataset_features, Dataset_pairs, datasetType):
	if (datasetType=='HO'):
		Dataset_features = Dataset_features.drop(['XXX'], axis=1)
	#Performing all inner joins
	dataset = pd.merge(Dataset_pairs, Dataset_features, left_on=['img_id_A'], right_on=['img_id'], how='inner')
	dataset = pd.merge(dataset, Dataset_features, left_on=['img_id_B'], right_on=['img_id'], how='inner')
	dataset = dataset.drop(['img_id_A', 'img_id_B', 'img_id_x', 'img_id_y'], axis=1)
	#Concantenation Features
	concantenatedFeatures_input = pd.DataFrame(dataset)
	#Subtracted Features
	subtractedFeatures_input = pd.DataFrame()
	dataset_temp = dataset.drop(['target'], axis=1)
	dataset_columns = dataset_temp.columns
	dataset_columnsX = dataset_columns[0:(int)(len(dataset_columns)/2)]
	dataset_columnsY = dataset_columns[(int)(len(dataset_columns)/2):]
	i=0
	while(i<len(dataset_columnsX) and i<len(dataset_columnsY)):
		subtractedFeatures_input[dataset_columnsX[i]] = dataset_temp[dataset_columnsX[i]]-dataset_temp[dataset_columnsY[i]]
		subtractedFeatures_input[dataset_columnsX[i]] = subtractedFeatures_input[dataset_columnsX[i]].abs()
		i=i+1
	subtractedFeatures_input['target']=dataset['target']
	return concantenatedFeatures_input, subtractedFeatures_input


def processCSVFiles():
	#Reading Human Observed Data Set
	print("Starting humanObservedDataset CSV read at "+(str)(time.asctime( time.localtime(time.time()) )))
	humanObservedDataset_features = readCSVFile('HumanObserved-Dataset/HumanObserved-Features-Data/HumanObserved-Features-Data.csv')
	humanObservedDataset_diffPairs = readCSVFile('HumanObserved-Dataset/HumanObserved-Features-Data/diffn_pairs.csv')
	humanObservedDataset_samePairs = readCSVFile('HumanObserved-Dataset/HumanObserved-Features-Data/same_pairs.csv')
	humanObservedDataset_pairs = reduceSizeAndMergeSameAndDiffPairs(humanObservedDataset_samePairs, humanObservedDataset_diffPairs, "HO")
	humanObservedDataset_featureConcantenation , humanObservedDataset_featureSubtraction = featureConcantenationAndSubtraction(humanObservedDataset_features, humanObservedDataset_pairs, 'HO')
	print("Finished humanObservedDataset CSV read at "+(str)(time.asctime( time.localtime(time.time()) )))
	#Reading GSC Data Set
	print("Starting gscDataset CSV read at "+(str)(time.asctime( time.localtime(time.time()) )))
	gscDataset_features = readCSVFile('GSC-Dataset/GSC-Features-Data/GSC-Features.csv')
	gscDataset_diffPairs = readCSVFile('GSC-Dataset/GSC-Features-Data/diffn_pairs.csv')
	gscDataset_samePairs = readCSVFile('GSC-Dataset/GSC-Features-Data/same_pairs.csv')
	gscDataset_pairs = reduceSizeAndMergeSameAndDiffPairs(gscDataset_samePairs, gscDataset_diffPairs, "GSC")
	gscDataset_featureConcantenation , gscDataset_featureSubtraction = featureConcantenationAndSubtraction(gscDataset_features, gscDataset_pairs, 'GSC')
	print("Finished gscDataset CSV read at "+(str)(time.asctime( time.localtime(time.time()) )))
	return humanObservedDataset_featureConcantenation, humanObservedDataset_featureSubtraction, gscDataset_featureConcantenation , gscDataset_featureSubtraction


def createTrainingAndValidationAndTestingData():
	humanObservedDataset_featureConcantenation, humanObservedDataset_featureSubtraction, gscDataset_featureConcantenation , gscDataset_featureSubtraction = processCSVFiles()
	writeDFIntoCSVFile(humanObservedDataset_featureConcantenation, "humanObservedDataset_featureConcantenation")
	writeDFIntoCSVFile(humanObservedDataset_featureSubtraction, "humanObservedDataset_featureSubtraction")
	writeDFIntoCSVFile(gscDataset_featureConcantenation, "gscDataset_featureConcantenation")
	writeDFIntoCSVFile(gscDataset_featureSubtraction, "gscDataset_featureSubtraction")
	print("Starting humanObservedDataset processing at "+(str)(time.asctime( time.localtime(time.time()) )))
	Dataset = { "humanObservedDataset": { "concantenated": None, "subtracted": None },
				"gscDataset": { "concantenated": None, "subtracted": None } }
	humanObservedDataset_featureConcantenation_target = humanObservedDataset_featureConcantenation['target']
	humanObservedDataset_featureConcantenation_inputs = humanObservedDataset_featureConcantenation.drop(['target'], axis=1)
	inputs_trainAndValidation, inputs_test, target_trainAndValidation, target_test = train_test_split(humanObservedDataset_featureConcantenation_inputs, humanObservedDataset_featureConcantenation_target, test_size=(testingDatasetPercentage/100), shuffle=False)
	inputs_train, inputs_validation, target_train, target_validation = train_test_split(inputs_trainAndValidation, target_trainAndValidation, test_size=((validationDatasetPercentage/100)/((trainingDatasetPercentage/100)+(validationDatasetPercentage/100))), shuffle=False)
	del inputs_trainAndValidation, target_trainAndValidation
	Dataset["humanObservedDataset"]["concantenated"] = { 
											"training" : { "features": inputs_train, "target": target_train}, 
											"validation" : { "features": inputs_validation, "target": target_validation}, 
											"testing" : { "features": inputs_test, "target": target_test} 
										}
	humanObservedDataset_featureSubtraction_target = humanObservedDataset_featureSubtraction['target']
	humanObservedDataset_featureSubtraction_inputs = humanObservedDataset_featureSubtraction.drop(['target'], axis=1)
	inputs_trainAndValidation, inputs_test, target_trainAndValidation, target_test = train_test_split(humanObservedDataset_featureSubtraction_inputs, humanObservedDataset_featureSubtraction_target, test_size=(testingDatasetPercentage/100), shuffle=False)
	inputs_train, inputs_validation, target_train, target_validation = train_test_split(inputs_trainAndValidation, target_trainAndValidation, test_size=((validationDatasetPercentage/100)/((trainingDatasetPercentage/100)+(validationDatasetPercentage/100))), shuffle=False)
	del inputs_trainAndValidation, target_trainAndValidation
	Dataset["humanObservedDataset"]["subtracted"] = { 
											"training" : { "features": inputs_train, "target": target_train}, 
											"validation" : { "features": inputs_validation, "target": target_validation}, 
											"testing" : { "features": inputs_test, "target": target_test} 
										}
	print("Ending humanObservedDataset processing at "+(str)(time.asctime( time.localtime(time.time()) )))
	print("Starting gscDataset processing at "+(str)(time.asctime( time.localtime(time.time()) )))
	gscDataset_featureConcantenation_target = gscDataset_featureConcantenation['target']
	gscDataset_featureConcantenation_inputs = gscDataset_featureConcantenation.drop(['target'], axis=1)
	inputs_trainAndValidation, inputs_test, target_trainAndValidation, target_test = train_test_split(gscDataset_featureConcantenation_inputs, gscDataset_featureConcantenation_target, test_size=(testingDatasetPercentage/100), shuffle=False)
	inputs_train, inputs_validation, target_train, target_validation = train_test_split(inputs_trainAndValidation, target_trainAndValidation, test_size=((validationDatasetPercentage/100)/((trainingDatasetPercentage/100)+(validationDatasetPercentage/100))), shuffle=False)
	del inputs_trainAndValidation, target_trainAndValidation
	Dataset["gscDataset"]["concantenated"] = { 
											"training" : { "features": inputs_train, "target": target_train}, 
											"validation" : { "features": inputs_validation, "target": target_validation}, 
											"testing" : { "features": inputs_test, "target": target_test} 
										}
	gscDataset_featureSubtraction_target = gscDataset_featureSubtraction['target']
	gscDataset_featureSubtraction_inputs = gscDataset_featureSubtraction.drop(['target'], axis=1)
	inputs_trainAndValidation, inputs_test, target_trainAndValidation, target_test = train_test_split(gscDataset_featureSubtraction_inputs, gscDataset_featureSubtraction_target, test_size=(testingDatasetPercentage/100), shuffle=False)
	inputs_train, inputs_validation, target_train, target_validation = train_test_split(inputs_trainAndValidation, target_trainAndValidation, test_size=((validationDatasetPercentage/100)/((trainingDatasetPercentage/100)+(validationDatasetPercentage/100))), shuffle=False)
	del inputs_trainAndValidation, target_trainAndValidation
	Dataset["gscDataset"]["subtracted"] = { 
											"training" : { "features": inputs_train, "target": target_train}, 
											"validation" : { "features": inputs_validation, "target": target_validation}, 
											"testing" : { "features": inputs_test, "target": target_test} 
										}
	print("Ending gscDataset processing at "+(str)(time.asctime( time.localtime(time.time()) )))
	return Dataset


def performLinearRegression(dataSet):
	print("Starting Linear Regression at "+(str)(time.asctime( time.localtime(time.time()) )))
	finalLinearRegressors = {   "humanObservedDataset": { "concantenated": None, "subtracted": None },
						 		"gscDataset": { "concantenated": None, "subtracted": None } }
	for attr1, value1 in dataSet.items():
		for attr2, value2 in value1.items():
			linearRegressors = []
			hyperParam_learningType_Values = ["SGD"] #["CLOSED_FORM", "SGD"]
			hyperParam_M_Values = [15] #[5, 10, 15, 20, 30, 50]
			hyperParam_lambdaa_Values = [0.0] #[0.0, 0.01, 0.05, 0.1, 0.2]
			hyperParam_BigSigma_scalingFactor_Values = [100] #[1, 5, 100, 200, 300, 500]
			for i in range(0, len(hyperParam_learningType_Values)):
				for j in range(0, len(hyperParam_M_Values)):
					for k in range(0, len(hyperParam_lambdaa_Values)):
						for l in range(0, len(hyperParam_BigSigma_scalingFactor_Values)):
							try:
								print("		Now in : "+attr1+" "+attr2+" "+str(i)+" "+str(j)+" "+str(k)+" "+str(l))
								if (hyperParam_learningType_Values[i]=="CLOSED_FORM"):
									linearRegressor = LinearRegression(hyperParam_learningType_Values[i], hyperParam_M_Values[j], hyperParam_lambdaa_Values[k], hyperParam_BigSigma_scalingFactor_Values[l])
									linearRegressor.fit( (value2["training"]["features"]).transpose(), (value2["training"]["target"]).transpose() )
									training_error = root_mean_square_error( value2["training"]["target"], (linearRegressor.predict( (value2["training"]["features"]).transpose() )) )
									validation_error = root_mean_square_error( value2["validation"]["target"], (linearRegressor.predict( (value2["validation"]["features"]).transpose() )) )
									linearRegressors.append({"linearRegressor": linearRegressor, "validation_error": validation_error, "training_error": training_error, 
															"learningType": hyperParam_learningType_Values[i], "M": hyperParam_M_Values[j], "lambda": hyperParam_lambdaa_Values[k], "BigSigma_scalingFactor": hyperParam_BigSigma_scalingFactor_Values[l], 
															"SGDParameters": None, "testing_error": None, "testing_accuracyScore": None, "testing_precisionScore": None, "testing_recallScore": None})
								else:
									SGDParameters = {"numberOfIterationsInSGDLoop":200, "learningRate":0.01, "lambda": 0, "ERMS_threshold": None}
									linearRegressor = LinearRegression(hyperParam_learningType_Values[i], hyperParam_M_Values[j], hyperParam_lambdaa_Values[k], hyperParam_BigSigma_scalingFactor_Values[l])
									linearRegressor.fit( (value2["training"]["features"]).transpose(), (value2["training"]["target"]).transpose(), SGDParameters)
									training_error = root_mean_square_error( value2["training"]["target"], (linearRegressor.predict( (value2["training"]["features"]).transpose() )) )
									validation_error = root_mean_square_error( value2["validation"]["target"], (linearRegressor.predict( (value2["validation"]["features"]).transpose() )) )
									linearRegressors.append({"linearRegressor": linearRegressor, "validation_error": validation_error, "training_error": training_error, 
															"learningType": hyperParam_learningType_Values[i], "M": hyperParam_M_Values[j], "lambda": hyperParam_lambdaa_Values[k], "BigSigma_scalingFactor": hyperParam_BigSigma_scalingFactor_Values[l], 
															"SGDParameters": SGDParameters, "testing_error": None, "testing_accuracyScore": None, "testing_precisionScore": None, "testing_recallScore": None})
									print(training_error)
									print(validation_error)
									print()
							except Exception as e:
								print("Exception -"+str(e)+"- occured for : "+attr1+" "+attr2+" "+str(i)+" "+str(j)+" "+str(k)+" "+str(l))
			if (len(linearRegressors)!=0):
				bestLinearRegressorObject = min(linearRegressors, key = lambda x: x["validation_error"])
				bestLinearRegressor_TestingResult = ((bestLinearRegressorObject["linearRegressor"]).predict( (value2["testing"]["features"]).transpose() ))
				bestLinearRegressorObject["testing_error"] = root_mean_square_error(value2["testing"]["target"], bestLinearRegressor_TestingResult)
				mapRegressionOutputToBinaryValues(bestLinearRegressor_TestingResult)
				bestLinearRegressorObject["testing_accuracyScore"] = accuracy_score(value2["testing"]["target"], bestLinearRegressor_TestingResult)
				bestLinearRegressorObject["testing_precisionScore"] = precision_score(value2["testing"]["target"], bestLinearRegressor_TestingResult)
				bestLinearRegressorObject["testing_recallScore"] = recall_score(value2["testing"]["target"], bestLinearRegressor_TestingResult)
				finalLinearRegressors[attr1][attr2] = bestLinearRegressorObject
			else:
				finalLinearRegressors[attr1][attr2] = { "linearRegressor": None, "validation_error": None, "training_error": None, "testing_precisionScore": None,
														"learningType": None, "M": None, "lambda": None, "BigSigma_scalingFactor": None, "testing_recallScore": None,
														"SGDParameters": None, "testing_error": None, "testing_accuracyScore": None}
	print("		Linear Regressor Stats:")
	print("			Human Observed DataSet - Concantenated:")
	print("				HyperParameters: LearningType = "+(str)(finalLinearRegressors["humanObservedDataset"]["concantenated"]["learningType"])+" ; M = "+(str)(finalLinearRegressors["humanObservedDataset"]["concantenated"]["M"])+" ; lambda = "+(str)(finalLinearRegressors["humanObservedDataset"]["concantenated"]["lambda"])+" ; BigSigma_scalingFactor = "+(str)(finalLinearRegressors["humanObservedDataset"]["concantenated"]["BigSigma_scalingFactor"])+" ; SGDParameters = "+(str)(finalLinearRegressors["humanObservedDataset"]["concantenated"]["SGDParameters"]))
	print("				Training RMS Error = "+(str)(finalLinearRegressors["humanObservedDataset"]["concantenated"]["training_error"]))
	print("				Testing RMS Error = "+(str)(finalLinearRegressors["humanObservedDataset"]["concantenated"]["testing_error"]))
	print("				Testing Accuracy Score = "+(str)(finalLinearRegressors["humanObservedDataset"]["concantenated"]["testing_accuracyScore"]))
	print("				Testing Precision Score = "+(str)(finalLinearRegressors["humanObservedDataset"]["concantenated"]["testing_precisionScore"]))
	print("				Testing Recall Score = "+(str)(finalLinearRegressors["humanObservedDataset"]["concantenated"]["testing_recallScore"]))
	print("			Human Observed DataSet - Subtracted:")
	print("				HyperParameters: LearningType = "+(str)(finalLinearRegressors["humanObservedDataset"]["subtracted"]["learningType"])+" ; M = "+(str)(finalLinearRegressors["humanObservedDataset"]["subtracted"]["M"])+" ; lambda = "+(str)(finalLinearRegressors["humanObservedDataset"]["subtracted"]["lambda"])+" ; BigSigma_scalingFactor = "+(str)(finalLinearRegressors["humanObservedDataset"]["subtracted"]["BigSigma_scalingFactor"])+" ; SGDParameters = "+(str)(finalLinearRegressors["humanObservedDataset"]["subtracted"]["SGDParameters"]))
	print("				Training RMS Error = "+(str)(finalLinearRegressors["humanObservedDataset"]["subtracted"]["training_error"]))
	print("				Testing RMS Error = "+(str)(finalLinearRegressors["humanObservedDataset"]["subtracted"]["testing_error"]))
	print("				Testing Accuracy Score = "+(str)(finalLinearRegressors["humanObservedDataset"]["subtracted"]["testing_accuracyScore"]))
	print("				Testing Precision Score = "+(str)(finalLinearRegressors["humanObservedDataset"]["subtracted"]["testing_precisionScore"]))
	print("				Testing Recall Score = "+(str)(finalLinearRegressors["humanObservedDataset"]["subtracted"]["testing_recallScore"]))
	print("			GSC DataSet - Concantenated:")
	print("				HyperParameters: LearningType = "+(str)(finalLinearRegressors["gscDataset"]["concantenated"]["learningType"])+" ; M = "+(str)(finalLinearRegressors["gscDataset"]["concantenated"]["M"])+" ; lambda = "+(str)(finalLinearRegressors["gscDataset"]["concantenated"]["lambda"])+" ; BigSigma_scalingFactor = "+(str)(finalLinearRegressors["gscDataset"]["concantenated"]["BigSigma_scalingFactor"])+" ; SGDParameters = "+(str)(finalLinearRegressors["gscDataset"]["concantenated"]["SGDParameters"]))
	print("				Training RMS Error = "+(str)(finalLinearRegressors["gscDataset"]["concantenated"]["training_error"]))
	print("				Testing RMS Error = "+(str)(finalLinearRegressors["gscDataset"]["concantenated"]["testing_error"]))
	print("				Testing Accuracy Score = "+(str)(finalLinearRegressors["gscDataset"]["concantenated"]["testing_accuracyScore"]))
	print("				Testing Precision Score = "+(str)(finalLinearRegressors["gscDataset"]["concantenated"]["testing_precisionScore"]))
	print("				Testing Recall Score = "+(str)(finalLinearRegressors["gscDataset"]["concantenated"]["testing_recallScore"]))
	print("			GSC DataSet - Subtracted:")
	print("				HyperParameters: LearningType = "+(str)(finalLinearRegressors["gscDataset"]["subtracted"]["learningType"])+" ; M = "+(str)(finalLinearRegressors["gscDataset"]["subtracted"]["M"])+" ; lambda = "+(str)(finalLinearRegressors["gscDataset"]["subtracted"]["lambda"])+" ; BigSigma_scalingFactor = "+(str)(finalLinearRegressors["gscDataset"]["subtracted"]["BigSigma_scalingFactor"])+" ; SGDParameters = "+(str)(finalLinearRegressors["gscDataset"]["subtracted"]["SGDParameters"]))
	print("				Training RMS Error = "+(str)(finalLinearRegressors["gscDataset"]["subtracted"]["training_error"]))
	print("				Testing RMS Error = "+(str)(finalLinearRegressors["gscDataset"]["subtracted"]["testing_error"]))
	print("				Testing Accuracy Score = "+(str)(finalLinearRegressors["gscDataset"]["subtracted"]["testing_accuracyScore"]))
	print("				Testing Precision Score = "+(str)(finalLinearRegressors["gscDataset"]["subtracted"]["testing_precisionScore"]))
	print("				Testing Recall Score = "+(str)(finalLinearRegressors["gscDataset"]["subtracted"]["testing_recallScore"]))
	print("Ending Linear Regression at "+(str)(time.asctime( time.localtime(time.time()) )))


def performLogisticRegression(dataSet):
	print("Starting Logistic Regression at "+(str)(time.asctime( time.localtime(time.time()) )))
	finalLogisticRegressors = {   "humanObservedDataset": { "concantenated": None, "subtracted": None },
						 			"gscDataset": { "concantenated": None, "subtracted": None } }
	for attr1, value1 in dataSet.items():
		for attr2, value2 in value1.items():
			logisticRegressors = []
			hyperParam_learningRate_Values = [0.05] #[0.05, 0.1, 0.2] 
			hyperParam_NumberOfIterations_Values = [2000, 5000]  #[5000, 10000, 20000, 30000] 
			hyperParam_SigmoidThreshold_Values = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6]
			for i in range(0, len(hyperParam_learningRate_Values)):
				for j in range(0, len(hyperParam_NumberOfIterations_Values)):
					try:
						logisticRegressor = LogisticRegression()
						logisticRegressor.fit( (value2["training"]["features"]).transpose(), (value2["training"]["target"]).transpose(), hyperParam_learningRate_Values[i], hyperParam_NumberOfIterations_Values[j] )
						training_error = root_mean_square_error( value2["training"]["target"], (logisticRegressor.predict( (value2["training"]["features"]).transpose() )) )
						logisticRegressor_ValidationResult = (logisticRegressor.predict( (value2["validation"]["features"]).transpose() ))
						validation_error = root_mean_square_error( value2["validation"]["target"], logisticRegressor_ValidationResult )
						for k in range(0, len(hyperParam_SigmoidThreshold_Values)):
							print("		Now in : "+attr1+" "+attr2+" "+str(i)+" "+str(j)+" "+str(k))
							logisticRegressor_ValidationResult_mapped = logisticRegressor.mapRegressionOutputToBinaryValues(logisticRegressor_ValidationResult, hyperParam_SigmoidThreshold_Values[k])
							validation_precisionScore = precision_score(value2["validation"]["target"], logisticRegressor_ValidationResult_mapped)
							logisticRegressors.append({"logisticRegressor": logisticRegressor, "validation_error": validation_error, "validation_precisionScore": validation_precisionScore, "training_error": training_error, 
														"learningRate": hyperParam_learningRate_Values[i], "NumberOfIterations": hyperParam_NumberOfIterations_Values[j], "SigmoidThreshold": hyperParam_SigmoidThreshold_Values[k] })
					except Exception as e:
						print("Exception -"+str(e)+"- occured for : "+attr1+" "+attr2+" "+str(i)+" "+str(j))
			if (len(logisticRegressors)!=0):
				bestLogisticRegressorObject_minValError = min(logisticRegressors, key = lambda x: x["validation_error"])
				bestLogisticRegressorObject_temp = list(filter(lambda x: x["validation_error"]==bestLogisticRegressorObject_minValError["validation_error"], logisticRegressors))
				bestLogisticRegressorObject = max(bestLogisticRegressorObject_temp, key = lambda x: x["validation_precisionScore"])
				bestLogisticRegressor_TestingResult = ((bestLogisticRegressorObject["logisticRegressor"]).predict( (value2["testing"]["features"]).transpose() ))
				bestLogisticRegressorObject["testing_error"] = root_mean_square_error(value2["testing"]["target"], bestLogisticRegressor_TestingResult)
				bestLogisticRegressor_TestingResult = logisticRegressor.mapRegressionOutputToBinaryValues(bestLogisticRegressor_TestingResult, bestLogisticRegressorObject["SigmoidThreshold"])
				bestLogisticRegressorObject["testing_accuracyScore"] = accuracy_score(value2["testing"]["target"], bestLogisticRegressor_TestingResult)
				bestLogisticRegressorObject["testing_precisionScore"] = precision_score(value2["testing"]["target"], bestLogisticRegressor_TestingResult)
				bestLogisticRegressorObject["testing_recallScore"] = recall_score(value2["testing"]["target"], bestLogisticRegressor_TestingResult)
				finalLogisticRegressors[attr1][attr2] = bestLogisticRegressorObject
			else:
				finalLogisticRegressors[attr1][attr2] = { "logisticRegressor": None, "validation_error": None, "training_error": None, "testing_precisionScore": None,
														"testing_recallScore": None, "testing_error": None, "testing_accuracyScore": None,
														"learningRate": None, "NumberOfIterations": None, "SigmoidThreshold": None }
	print("		Logistic Regressor Stats:")
	print("			Human Observed DataSet - Concantenated:")
	print("				HyperParameters: learningRate = "+(str)(finalLogisticRegressors["humanObservedDataset"]["concantenated"]["learningRate"])+" ; NumberOfIterations = "+(str)(finalLogisticRegressors["humanObservedDataset"]["concantenated"]["NumberOfIterations"])+" ; SigmoidThreshold = "+(str)(finalLogisticRegressors["humanObservedDataset"]["concantenated"]["SigmoidThreshold"]) )
	print("				Training RMS Error = "+(str)(finalLogisticRegressors["humanObservedDataset"]["concantenated"]["training_error"]))
	print("				Testing RMS Error = "+(str)(finalLogisticRegressors["humanObservedDataset"]["concantenated"]["testing_error"]))
	print("				Testing Accuracy Score = "+(str)(finalLogisticRegressors["humanObservedDataset"]["concantenated"]["testing_accuracyScore"]))
	print("				Testing Precision Score = "+(str)(finalLogisticRegressors["humanObservedDataset"]["concantenated"]["testing_precisionScore"]))
	print("				Testing Recall Score = "+(str)(finalLogisticRegressors["humanObservedDataset"]["concantenated"]["testing_recallScore"]))
	print("			Human Observed DataSet - Subtracted:")
	print("				HyperParameters: learningRate = "+(str)(finalLogisticRegressors["humanObservedDataset"]["subtracted"]["learningRate"])+" ; NumberOfIterations = "+(str)(finalLogisticRegressors["humanObservedDataset"]["subtracted"]["NumberOfIterations"])+" ; SigmoidThreshold = "+(str)(finalLogisticRegressors["humanObservedDataset"]["subtracted"]["SigmoidThreshold"]) )
	print("				Training RMS Error = "+(str)(finalLogisticRegressors["humanObservedDataset"]["subtracted"]["training_error"]))
	print("				Testing RMS Error = "+(str)(finalLogisticRegressors["humanObservedDataset"]["subtracted"]["testing_error"]))
	print("				Testing Accuracy Score = "+(str)(finalLogisticRegressors["humanObservedDataset"]["subtracted"]["testing_accuracyScore"]))
	print("				Testing Precision Score = "+(str)(finalLogisticRegressors["humanObservedDataset"]["subtracted"]["testing_precisionScore"]))
	print("				Testing Recall Score = "+(str)(finalLogisticRegressors["humanObservedDataset"]["subtracted"]["testing_recallScore"]))
	print("			GSC DataSet - Concantenated:")
	print("				HyperParameters: learningRate = "+(str)(finalLogisticRegressors["gscDataset"]["concantenated"]["learningRate"])+" ; NumberOfIterations = "+(str)(finalLogisticRegressors["gscDataset"]["concantenated"]["NumberOfIterations"])+" ; SigmoidThreshold = "+(str)(finalLogisticRegressors["gscDataset"]["concantenated"]["SigmoidThreshold"]) )
	print("				Training RMS Error = "+(str)(finalLogisticRegressors["gscDataset"]["concantenated"]["training_error"]))
	print("				Testing RMS Error = "+(str)(finalLogisticRegressors["gscDataset"]["concantenated"]["testing_error"]))
	print("				Testing Accuracy Score = "+(str)(finalLogisticRegressors["gscDataset"]["concantenated"]["testing_accuracyScore"]))
	print("				Testing Precision Score = "+(str)(finalLogisticRegressors["gscDataset"]["concantenated"]["testing_precisionScore"]))
	print("				Testing Recall Score = "+(str)(finalLogisticRegressors["gscDataset"]["concantenated"]["testing_recallScore"]))
	print("			GSC DataSet - Subtracted:")
	print("				HyperParameters: learningRate = "+(str)(finalLogisticRegressors["gscDataset"]["subtracted"]["learningRate"])+" ; NumberOfIterations = "+(str)(finalLogisticRegressors["gscDataset"]["subtracted"]["NumberOfIterations"])+" ; SigmoidThreshold = "+(str)(finalLogisticRegressors["gscDataset"]["subtracted"]["SigmoidThreshold"]) )
	print("				Training RMS Error = "+(str)(finalLogisticRegressors["gscDataset"]["subtracted"]["training_error"]))
	print("				Testing RMS Error = "+(str)(finalLogisticRegressors["gscDataset"]["subtracted"]["testing_error"]))
	print("				Testing Accuracy Score = "+(str)(finalLogisticRegressors["gscDataset"]["subtracted"]["testing_accuracyScore"]))
	print("				Testing Precision Score = "+(str)(finalLogisticRegressors["gscDataset"]["subtracted"]["testing_precisionScore"]))
	print("				Testing Recall Score = "+(str)(finalLogisticRegressors["gscDataset"]["subtracted"]["testing_recallScore"]))
	print("Ending Logistic Regression at "+(str)(time.asctime( time.localtime(time.time()) )))


def performSVC(dataSet):
	print("Starting SVC at "+(str)(time.asctime( time.localtime(time.time()) )))
	finalSVCRegressors = {   "humanObservedDataset": { "concantenated": None, "subtracted": None },
							"gscDataset": { "concantenated": None, "subtracted": None } }
	for attr1, value1 in dataSet.items():
		for attr2, value2 in value1.items():
			SVC_classifier = SVC()
			SVC_classifier.fit( (value2["training"]["features"]), (value2["training"]["target"]) )
			training_error = root_mean_square_error( value2["training"]["target"], (SVC_classifier.predict( value2["training"]["features"] )) )
			validation_error = root_mean_square_error( value2["validation"]["target"], (SVC_classifier.predict( value2["validation"]["features"] )) )
			SVC_classifier_TestingResult = (SVC_classifier).predict(value2["testing"]["features"])
			bestSVCRegressorObject = {"SVC_classifier": SVC_classifier, "validation_error": validation_error, "training_error": training_error, 
										"testing_error": None, "testing_accuracyScore": None, "testing_precisionScore": None, "testing_recallScore": None}
			bestSVCRegressorObject["testing_error"] = root_mean_square_error(value2["testing"]["target"], SVC_classifier_TestingResult)
			bestSVCRegressorObject["testing_accuracyScore"] = accuracy_score(value2["testing"]["target"], SVC_classifier_TestingResult)
			bestSVCRegressorObject["testing_precisionScore"] = precision_score(value2["testing"]["target"], SVC_classifier_TestingResult)
			bestSVCRegressorObject["testing_recallScore"] = recall_score(value2["testing"]["target"], SVC_classifier_TestingResult)
			finalSVCRegressors[attr1][attr2] = bestSVCRegressorObject
	print("		SVC Stats:")
	print("			Human Observed DataSet - Concantenated:")
	print("				Training RMS Error = "+(str)(finalSVCRegressors["humanObservedDataset"]["concantenated"]["training_error"]))
	print("				Testing RMS Error = "+(str)(finalSVCRegressors["humanObservedDataset"]["concantenated"]["testing_error"]))
	print("				Testing Accuracy Score = "+(str)(finalSVCRegressors["humanObservedDataset"]["concantenated"]["testing_accuracyScore"]))
	print("				Testing Precision Score = "+(str)(finalSVCRegressors["humanObservedDataset"]["concantenated"]["testing_precisionScore"]))
	print("				Testing Recall Score = "+(str)(finalSVCRegressors["humanObservedDataset"]["concantenated"]["testing_recallScore"]))
	print("			Human Observed DataSet - Subtracted:")
	print("				Training RMS Error = "+(str)(finalSVCRegressors["humanObservedDataset"]["subtracted"]["training_error"]))
	print("				Testing RMS Error = "+(str)(finalSVCRegressors["humanObservedDataset"]["subtracted"]["testing_error"]))
	print("				Testing Accuracy Score = "+(str)(finalSVCRegressors["humanObservedDataset"]["subtracted"]["testing_accuracyScore"]))
	print("				Testing Precision Score = "+(str)(finalSVCRegressors["humanObservedDataset"]["subtracted"]["testing_precisionScore"]))
	print("				Testing Recall Score = "+(str)(finalSVCRegressors["humanObservedDataset"]["subtracted"]["testing_recallScore"]))
	print("			GSC DataSet - Concantenated:")
	print("				Training RMS Error = "+(str)(finalSVCRegressors["gscDataset"]["concantenated"]["training_error"]))
	print("				Testing RMS Error = "+(str)(finalSVCRegressors["gscDataset"]["concantenated"]["testing_error"]))
	print("				Testing Accuracy Score = "+(str)(finalSVCRegressors["gscDataset"]["concantenated"]["testing_accuracyScore"]))
	print("				Testing Precision Score = "+(str)(finalSVCRegressors["gscDataset"]["concantenated"]["testing_precisionScore"]))
	print("				Testing Recall Score = "+(str)(finalSVCRegressors["gscDataset"]["concantenated"]["testing_recallScore"]))
	print("			GSC DataSet - Subtracted:")
	print("				Training RMS Error = "+(str)(finalSVCRegressors["gscDataset"]["subtracted"]["training_error"]))
	print("				Testing RMS Error = "+(str)(finalSVCRegressors["gscDataset"]["subtracted"]["testing_error"]))
	print("				Testing Accuracy Score = "+(str)(finalSVCRegressors["gscDataset"]["subtracted"]["testing_accuracyScore"]))
	print("				Testing Precision Score = "+(str)(finalSVCRegressors["gscDataset"]["subtracted"]["testing_precisionScore"]))
	print("				Testing Recall Score = "+(str)(finalSVCRegressors["gscDataset"]["subtracted"]["testing_recallScore"]))
	print("Ending SVC at "+(str)(time.asctime( time.localtime(time.time()) )))


def performNNModelling(dataSet):
	print("Starting MM Modelling at "+(str)(time.asctime( time.localtime(time.time()) )))
	finalNNs = {   "humanObservedDataset": { "concantenated": None, "subtracted": None },
					"gscDataset": { "concantenated": None, "subtracted": None } }
	learningRate = 0.05
	momentum = 0.4
	decay = 1e-6
	numOfEpochs = 10000
	modelBatchSize = 128
	tbBatchSize = 32
	earlyPatience = 100
	for attr1, value1 in dataSet.items():
		for attr2, value2 in value1.items():
			NN = NeuralNetwork(learningRate, momentum , decay)
			NN.fit(value2["training"]["features"], value2["training"]["target"], validationDatasetPercentage, numOfEpochs, modelBatchSize, tbBatchSize, earlyPatience)
			training_error = root_mean_square_error( value2["training"]["target"], NN.predict( value2["training"]["features"] ) )
			validation_error = root_mean_square_error( value2["validation"]["target"], NN.predict( value2["validation"]["features"] ) )
			result = NN.predict( value2["testing"]["features"] )
			testing_error = root_mean_square_error(value2["testing"]["target"], result)
			mapRegressionOutputToBinaryValues(result)
			testing_accuracyScore = accuracy_score(value2["testing"]["target"], result)
			testing_precisionScore = precision_score(value2["testing"]["target"], result)
			testing_recallScore = recall_score(value2["testing"]["target"], result)
			NNObj = {"NN":NN, "training_error":training_error, "validation_error":validation_error, "testing_error":testing_error, 
			"testing_accuracyScore":testing_accuracyScore, "testing_precisionScore":testing_precisionScore, "testing_recallScore":testing_recallScore} 
			finalNNs[attr1][attr2] = NNObj
	print("		NN Stats:")
	print("			Human Observed DataSet - Concantenated:")
	print("				Training RMS Error = "+(str)(finalNNs["humanObservedDataset"]["concantenated"]["training_error"]))
	print("				Testing RMS Error = "+(str)(finalNNs["humanObservedDataset"]["concantenated"]["testing_error"]))
	print("				Testing Accuracy Score = "+(str)(finalNNs["humanObservedDataset"]["concantenated"]["testing_accuracyScore"]))
	print("				Testing Precision Score = "+(str)(finalNNs["humanObservedDataset"]["concantenated"]["testing_precisionScore"]))
	print("				Testing Recall Score = "+(str)(finalNNs["humanObservedDataset"]["concantenated"]["testing_recallScore"]))
	print("			Human Observed DataSet - Subtracted:")
	print("				Training RMS Error = "+(str)(finalNNs["humanObservedDataset"]["subtracted"]["training_error"]))
	print("				Testing RMS Error = "+(str)(finalNNs["humanObservedDataset"]["subtracted"]["testing_error"]))
	print("				Testing Accuracy Score = "+(str)(finalNNs["humanObservedDataset"]["subtracted"]["testing_accuracyScore"]))
	print("				Testing Precision Score = "+(str)(finalNNs["humanObservedDataset"]["subtracted"]["testing_precisionScore"]))
	print("				Testing Recall Score = "+(str)(finalNNs["humanObservedDataset"]["subtracted"]["testing_recallScore"]))
	print("			GSC DataSet - Concantenated:")
	print("				Training RMS Error = "+(str)(finalNNs["gscDataset"]["concantenated"]["training_error"]))
	print("				Testing RMS Error = "+(str)(finalNNs["gscDataset"]["concantenated"]["testing_error"]))
	print("				Testing Accuracy Score = "+(str)(finalNNs["gscDataset"]["concantenated"]["testing_accuracyScore"]))
	print("				Testing Precision Score = "+(str)(finalNNs["gscDataset"]["concantenated"]["testing_precisionScore"]))
	print("				Testing Recall Score = "+(str)(finalNNs["gscDataset"]["concantenated"]["testing_recallScore"]))
	print("			GSC DataSet - Subtracted:")
	print("				Training RMS Error = "+(str)(finalNNs["gscDataset"]["subtracted"]["training_error"]))
	print("				Testing RMS Error = "+(str)(finalNNs["gscDataset"]["subtracted"]["testing_error"]))
	print("				Testing Accuracy Score = "+(str)(finalNNs["gscDataset"]["subtracted"]["testing_accuracyScore"]))
	print("				Testing Precision Score = "+(str)(finalNNs["gscDataset"]["subtracted"]["testing_precisionScore"]))
	print("				Testing Recall Score = "+(str)(finalNNs["gscDataset"]["subtracted"]["testing_recallScore"]))
	print("Ending NN Modelling at "+(str)(time.asctime( time.localtime(time.time()) )))


def performScikitLearnLinearRegression(dataSet):
	print("Starting Linear Regression at "+(str)(time.asctime( time.localtime(time.time()) )))
	finalLinearRegressors = {   "humanObservedDataset": { "concantenated": None, "subtracted": None },
						 		"gscDataset": { "concantenated": None, "subtracted": None } }
	for attr1, value1 in dataSet.items():
		for attr2, value2 in value1.items():
			linearRegressors = []
			fit_intercept = False 
			normalize = True
			i=1
			while(i<=3):
				linearRegressor = LinearRegression(fit_intercept=fit_intercept, normalize=normalize).fit(value2["training"]["features"], value2["training"]["target"])
				training_error = mean_squared_error(value2["training"]["target"], (linearRegressor.predict(value2["training"]["features"])) )
				validation_error = mean_squared_error(value2["validation"]["target"], (linearRegressor.predict(value2["validation"]["features"])) )
				linearRegressors.append({"linearRegressor": linearRegressor, "validation_error": validation_error, "training_error": training_error, 
										"testing_error": None, "testing_accuracyScore": None, "testing_precisionScore": None, "testing_recallScore": None})
				if (fit_intercept==False):
					fit_intercept=True
				else:
					normalize = False
				i=i+1
			bestLinearRegressorObject = min(linearRegressors, key = lambda x: x["validation_error"])
			bestLinearRegressor_TestingResult = ((bestLinearRegressorObject["linearRegressor"]).predict(value2["testing"]["features"]))
			bestLinearRegressorObject["testing_error"] = mean_squared_error(value2["testing"]["target"], bestLinearRegressor_TestingResult)
			mapRegressionOutputToBinaryValues(bestLinearRegressor_TestingResult)
			bestLinearRegressorObject["testing_accuracyScore"] = accuracy_score(value2["testing"]["target"], bestLinearRegressor_TestingResult)
			bestLinearRegressorObject["testing_precisionScore"] = precision_score(value2["testing"]["target"], bestLinearRegressor_TestingResult)
			bestLinearRegressorObject["testing_recallScore"] = recall_score(value2["testing"]["target"], bestLinearRegressor_TestingResult)
			finalLinearRegressors[attr1][attr2] = bestLinearRegressorObject
	print("		Linear Regressor Stats:")
	print("			Human Observed DataSet - Concantenated:")
	print("				Training RMS Error = "+(str)(finalLinearRegressors["humanObservedDataset"]["concantenated"]["training_error"]))
	print("				Testing RMS Error = "+(str)(finalLinearRegressors["humanObservedDataset"]["concantenated"]["testing_error"]))
	print("				Testing Accuracy Score = "+(str)(finalLinearRegressors["humanObservedDataset"]["concantenated"]["testing_accuracyScore"]))
	print("				Testing Precision Score = "+(str)(finalLinearRegressors["humanObservedDataset"]["concantenated"]["testing_precisionScore"]))
	print("				Testing Recall Score = "+(str)(finalLinearRegressors["humanObservedDataset"]["concantenated"]["testing_recallScore"]))
	print("			Human Observed DataSet - Subtracted:")
	print("				Training RMS Error = "+(str)(finalLinearRegressors["humanObservedDataset"]["subtracted"]["training_error"]))
	print("				Testing RMS Error = "+(str)(finalLinearRegressors["humanObservedDataset"]["subtracted"]["testing_error"]))
	print("				Testing Accuracy Score = "+(str)(finalLinearRegressors["humanObservedDataset"]["subtracted"]["testing_accuracyScore"]))
	print("				Testing Precision Score = "+(str)(finalLinearRegressors["humanObservedDataset"]["subtracted"]["testing_precisionScore"]))
	print("				Testing Recall Score = "+(str)(finalLinearRegressors["humanObservedDataset"]["subtracted"]["testing_recallScore"]))
	print("			GSC DataSet - Concantenated:")
	print("				Training RMS Error = "+(str)(finalLinearRegressors["gscDataset"]["concantenated"]["training_error"]))
	print("				Testing RMS Error = "+(str)(finalLinearRegressors["gscDataset"]["concantenated"]["testing_error"]))
	print("				Testing Accuracy Score = "+(str)(finalLinearRegressors["gscDataset"]["concantenated"]["testing_accuracyScore"]))
	print("				Testing Precision Score = "+(str)(finalLinearRegressors["gscDataset"]["concantenated"]["testing_precisionScore"]))
	print("				Testing Recall Score = "+(str)(finalLinearRegressors["gscDataset"]["concantenated"]["testing_recallScore"]))
	print("			GSC DataSet - Subtracted:")
	print("				Training RMS Error = "+(str)(finalLinearRegressors["gscDataset"]["subtracted"]["training_error"]))
	print("				Testing RMS Error = "+(str)(finalLinearRegressors["gscDataset"]["subtracted"]["testing_error"]))
	print("				Testing Accuracy Score = "+(str)(finalLinearRegressors["gscDataset"]["subtracted"]["testing_accuracyScore"]))
	print("				Testing Precision Score = "+(str)(finalLinearRegressors["gscDataset"]["subtracted"]["testing_precisionScore"]))
	print("				Testing Recall Score = "+(str)(finalLinearRegressors["gscDataset"]["subtracted"]["testing_recallScore"]))
	print("Ending Linear Regression at "+(str)(time.asctime( time.localtime(time.time()) )))


def main():
	print("Starting Main at "+(str)(time.asctime( time.localtime(time.time()) )))
	dataSet = createTrainingAndValidationAndTestingData()
	#performLinearRegression(dataSet)
	#performLogisticRegression(dataSet)
	#performSVC(dataSet)
	performNNModelling(dataSet)
	print("Main Ended at "+(str)(time.asctime( time.localtime(time.time()) )))


main()

