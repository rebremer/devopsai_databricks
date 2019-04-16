# Databricks notebook source
# MAGIC %md Azure ML & Azure Databricks notebooks by Rene Bremer (original taken from Parashar Shah)
# MAGIC 
# MAGIC Copyright (c) Microsoft Corporation. All rights reserved.
# MAGIC 
# MAGIC Licensed under the MIT License.

# COMMAND ----------

# MAGIC %md In this notebook, the level of income of a person is predicted (higher of lower than 50k per year). 
# MAGIC In this notebook, the following steps are executed:
# MAGIC 
# MAGIC 1. Ingest data
# MAGIC 2. Explore and Prepare data
# MAGIC 3. Logistic Regression - 1 feature using hours of week worked
# MAGIC 4. Logistic Regression - all features (hours of work, age, state, etc)
# MAGIC 5. Logistic regression - prevent overfitting (regularization)
# MAGIC 6. Decision tree - different algorithm
# MAGIC 

# COMMAND ----------

# MAGIC %md #1. Ingest Data

# COMMAND ----------

import os
import urllib
import pprint
import numpy as np
import time

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# COMMAND ----------

# Download AdultCensusIncome.csv from Azure CDN. This file has 32,561 rows.
basedataurl = "https://amldockerdatasets.azureedge.net"
datafile = "AdultCensusIncome.csv"
datafile_dbfs = os.path.join("/dbfs", datafile)

if os.path.isfile(datafile_dbfs):
    print("found {} at {}".format(datafile, datafile_dbfs))
else:
    print("downloading {} to {}".format(datafile, datafile_dbfs))
    urllib.request.urlretrieve(os.path.join(basedataurl, datafile), datafile_dbfs)
    time.sleep(30)
    

# COMMAND ----------

# MAGIC %md #2. Explore Data

# COMMAND ----------

# Create a Spark dataframe out of the csv file.
data_all = sqlContext.read.format('csv').options(header='true', inferSchema='true', ignoreLeadingWhiteSpace='true', ignoreTrailingWhiteSpace='true').load(datafile)
print("({}, {})".format(data_all.count(), len(data_all.columns)))

#renaming columns, all columns that contain a - will be replaced with an "_"
columns_new = [col.replace("-", "_") for col in data_all.columns]
data_all = data_all.toDF(*columns_new)

data_all.printSchema()

# COMMAND ----------

display(data_all.limit(5000))

# COMMAND ----------

# MAGIC %md #3. Logistic Regression - 1 feature

# COMMAND ----------

stages=[]
numericCols = ["hours_per_week"]
assembler = VectorAssembler(inputCols=numericCols, outputCol="features")
stages = [assembler]

# Convert label into label indices using the StringIndexer
label_stringIdx = StringIndexer(inputCol="income", outputCol="label")
stages += [label_stringIdx]

partialPipeline = Pipeline().setStages(stages)
pipelineModel = partialPipeline.fit(data_all)
preppedDataDF = pipelineModel.transform(data_all)

selectedcols = ["label", "features"] + ["hours_per_week"] + ["income"]
dataset = preppedDataDF.select(selectedcols)

# COMMAND ----------

(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=122423)

lrModel = LogisticRegression().fit(trainingData)

print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))

# COMMAND ----------

# Determine intersect point of "hours_per_week" where model goes from predicting income "<= 50k" to income ">50k"
predictions = lrModel.transform(testData)

selected = predictions.select("income", "label", "prediction", "probability", "hours_per_week").filter("hours_per_week > 65 and hours_per_week < 69")
display(selected)

# COMMAND ----------

# evaluate. note only 2 metrics are supported out of the box by Spark ML.
bce = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction')
au_roc = bce.setMetricName('areaUnderROC').evaluate(predictions)
au_prc = bce.setMetricName('areaUnderPR').evaluate(predictions)

truePositive = predictions.select("label").filter("label = 1 and prediction = 1").count()
falsePositive = predictions.select("label").filter("label = 0 and prediction = 1").count()
trueNegative = predictions.select("label").filter("label = 0 and prediction = 0").count()
falseNegative = predictions.select("label").filter("label = 1 and prediction = 0").count()

precision = truePositive/(truePositive + falsePositive)
recall = truePositive/(truePositive + falseNegative)

print("truePositive: " + str(truePositive))
print("falsePositive: " + str(falsePositive))
print("trueNegative: " + str(trueNegative))
print("falseNegative: " + str(falseNegative))
print("precision: " + str(precision))
print("recall: " + str(recall))

# "Official" statical measurement (the closer to 1, the better)
print("Area under ROC: {}".format(au_roc))
print("Area Under PR: {}".format(au_prc))

# COMMAND ----------

# MAGIC %md #4. Logistic regression - all features

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
categoricalColumns = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]
stages = [] # stages in our Pipeline
for categoricalCol in categoricalColumns:
    # Category Indexing with StringIndexer
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
    # Use OneHotEncoder to convert categorical variables into binary SparseVectors
    # encoder = OneHotEncoderEstimator(inputCol=categoricalCol + "Index", outputCol=categoricalCol + "classVec")
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    # Add stages.  These are not run here, but will run all at once later on.
    stages += [stringIndexer, encoder]
    
    
# Convert label into label indices using the StringIndexer
label_stringIdx = StringIndexer(inputCol="income", outputCol="label")
stages += [label_stringIdx]

# Transform all features into a vector using VectorAssembler
numericCols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

partialPipeline = Pipeline().setStages(stages)
pipelineModel = partialPipeline.fit(data_all)
preppedDataDF = pipelineModel.transform(data_all)

selectedcols = ["label", "features"] + ["income"] + categoricalColumns + numericCols
dataset = preppedDataDF.select(selectedcols)

# COMMAND ----------

(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=122423)

lrModel = LogisticRegression().fit(trainingData)

print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))

# COMMAND ----------

# Determine intersect point of "hours_per_week" where model goes from predicting income "<= 50k" to income ">50k"
predictions = lrModel.transform(testData)

selected = predictions.select("income", "label", "prediction", "probability", "hours_per_week").filter("hours_per_week > 65 and hours_per_week < 69")
display(selected)

# COMMAND ----------

# evaluate. note only 2 metrics are supported out of the box by Spark ML.
bce = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction')
au_roc = bce.setMetricName('areaUnderROC').evaluate(predictions)
au_prc = bce.setMetricName('areaUnderPR').evaluate(predictions)

truePositive = predictions.select("label").filter("label = 1 and prediction = 1").count()
falsePositive = predictions.select("label").filter("label = 0 and prediction = 1").count()
trueNegative = predictions.select("label").filter("label = 0 and prediction = 0").count()
falseNegative = predictions.select("label").filter("label = 1 and prediction = 0").count()

precision = truePositive/(truePositive + falsePositive)
recall = truePositive/(truePositive + falseNegative)

print("truePositive: " + str(truePositive))
print("falsePositive: " + str(falsePositive))
print("trueNegative: " + str(trueNegative))
print("falseNegative: " + str(falseNegative))
print("precision: " + str(precision))
print("recall: " + str(recall))

# "Official" statical measurement (the closer to 1, the better)
print("Area under ROC: {}".format(au_roc))
print("Area Under PR: {}".format(au_prc))

# COMMAND ----------

# MAGIC %md #5. Logistic regression - prevent overfitting (regularization)

# COMMAND ----------

regParam = [0, 0.01, 0.5, 2.0]
for reg in regParam:
    print("Regularization: " + str(reg))  
    lrModel = LogisticRegression(regParam=reg).fit(trainingData)

    predictions = lrModel.transform(testData)

    truePositive = predictions.select("label").filter("label = 1 and prediction = 1").count()
    falsePositive = predictions.select("label").filter("label = 0 and prediction = 1").count()
    trueNegative = predictions.select("label").filter("label = 0 and prediction = 0").count()
    falseNegative = predictions.select("label").filter("label = 1 and prediction = 0").count()

    print("truePositive: " + str(truePositive))
    print("falsePositive: " + str(falsePositive))
    print("trueNegative: " + str(trueNegative))
    print("falseNegative: " + str(falseNegative))
    print("-----")

# COMMAND ----------

# MAGIC %md #6. Decision tree - different algorithm

# COMMAND ----------

dtModel = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=3).fit(trainingData)

predictions = dtModel.transform(testData)

truePositive = predictions.select("label").filter("label = 1 and prediction = 1").count()
falsePositive = predictions.select("label").filter("label = 0 and prediction = 1").count()
trueNegative = predictions.select("label").filter("label = 0 and prediction = 0").count()
falseNegative = predictions.select("label").filter("label = 1 and prediction = 0").count()

print("truePositive: " + str(truePositive))
print("falsePositive: " + str(falsePositive))
print("trueNegative: " + str(trueNegative))
print("falseNegative: " + str(falseNegative))

# COMMAND ----------

maxDepth = [1, 3, 5, 10]
for maxd in maxDepth:
    print("MaxDepth tree: " + str(maxd))  
    dtModel = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=maxd).fit(trainingData)

    predictions = dtModel.transform(testData)

    truePositive = predictions.select("label").filter("label = 1 and prediction = 1").count()
    falsePositive = predictions.select("label").filter("label = 0 and prediction = 1").count()
    trueNegative = predictions.select("label").filter("label = 0 and prediction = 0").count()
    falseNegative = predictions.select("label").filter("label = 1 and prediction = 0").count()
    
    print("truePositive: " + str(truePositive))
    print("falsePositive: " + str(falsePositive))
    print("trueNegative: " + str(trueNegative))
    print("falseNegative: " + str(falseNegative))
    print("-----")
