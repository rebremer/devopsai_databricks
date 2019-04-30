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
# MAGIC 1. Initialize Azure Machine Learning Service
# MAGIC 2. Add model to Azure Machine Learning Service
# MAGIC 

# COMMAND ----------

# MAGIC %md #1. Initialize Azure Machine Learning Service

# COMMAND ----------

workspace="<Name of your workspace>"
subscription_id="<Subscription id>"
resource_grp="<Name of your resource group where aml service is created>"

experiment_name = "experiment_model_int"
model_name = "databricksmodel.mml" # in case you want to change the name, keep the .mml extension

# COMMAND ----------

import azureml.core
from azureml.core import Workspace
from azureml.core.run import Run
from azureml.core.experiment import Experiment

# Check core SDK version number
print("SDK version:", azureml.core.VERSION)

# COMMAND ----------

ws = Workspace(workspace_name = workspace,
               subscription_id = subscription_id,
               resource_group = resource_grp)

ws.get_details()

print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')

# COMMAND ----------

# MAGIC %md #2. Add model to Azure Machine Learning Service

# COMMAND ----------

import os
import urllib
import pprint
import numpy as np
import shutil
import time

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import OneHotEncoder, OneHotEncoderEstimator, StringIndexer, VectorAssembler
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

# Create a Spark dataframe out of the csv file.
data_all = sqlContext.read.format('csv').options(header='true', inferSchema='true', ignoreLeadingWhiteSpace='true', ignoreTrailingWhiteSpace='true').load(datafile)
print("({}, {})".format(data_all.count(), len(data_all.columns)))

#renaming columns, all columns that contain a - will be replaced with an "_"
columns_new = [col.replace("-", "_") for col in data_all.columns]
data_all = data_all.toDF(*columns_new)

data_all.printSchema()

# COMMAND ----------

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

(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=122423)

# COMMAND ----------

model_dbfs = os.path.join("/dbfs", model_name)

# start a training run by defining an experiment
myexperiment = Experiment(ws, experiment_name)
root_run = myexperiment.start_logging()

# COMMAND ----------

# Regularization Rates
from pyspark.ml.classification import LogisticRegression
regParam = [0,0.01, 0.5, 2.0]
# try a bunch of alpha values in a Linear Regression (Ridge) model
for reg in regParam:
    print("Regularization rate: {}".format(reg))
    # create a bunch of child runs
    with root_run.child_run("reg-" + str(reg)) as run:
        # create a new Logistic Regression model.
        
        lr = LogisticRegression(regParam=reg)
        
        # put together the pipeline
        pipe = Pipeline(stages=[lr])

        # train the model
        model_pipeline = pipe.fit(trainingData)
        predictions = model_pipeline.transform(testData)

        # evaluate. note only 2 metrics are supported out of the box by Spark ML.
        bce = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction')
        au_roc = bce.setMetricName('areaUnderROC').evaluate(predictions)
        au_prc = bce.setMetricName('areaUnderPR').evaluate(predictions)
        truePositive = predictions.select("label").filter("label = 1 and prediction = 1").count()
        falsePositive = predictions.select("label").filter("label = 0 and prediction = 1").count()
        trueNegative = predictions.select("label").filter("label = 0 and prediction = 0").count()
        falseNegative = predictions.select("label").filter("label = 1 and prediction = 0").count()

        # log reg, au_roc, au_prc and feature names in run history
        run.log("reg", reg)
        run.log("au_roc", au_roc)
        run.log("au_prc", au_prc)
        
        print("Area under ROC: {}".format(au_roc))
        print("Area Under PR: {}".format(au_prc))
       
        run.log("truePositive", truePositive)
        run.log("falsePositive", falsePositive)
        run.log("trueNegative", trueNegative)
        run.log("falseNegative", falseNegative)
                                                                                                                                                                  
        print("TP: " + str(truePositive) + ", FP: " + str(falsePositive) + ", TN: " + str(trueNegative) + ", FN: " + str(falseNegative))                                                                         
        
        run.log_list("columns", trainingData.columns)

        # save model
        model_pipeline.write().overwrite().save(model_name)
        
        # upload the serialized model into run history record
        mdl, ext = model_name.split(".")
        model_zip = mdl + ".zip"
        shutil.make_archive(mdl, 'zip', model_dbfs)
        run.upload_file("outputs/" + model_name, model_zip)        
        #run.upload_file("outputs/" + model_name, path_or_stream = model_dbfs) #cannot deal with folders

        # now delete the serialized model from local folder since it is already uploaded to run history 
        shutil.rmtree(model_dbfs)
        os.remove(model_zip)


# COMMAND ----------

# Regularization Rates
maxDepth = [1, 3, 5, 10]
for maxd in maxDepth:
    print("Max depth: {}".format(maxd))
    # create a bunch of child runs
    with root_run.child_run("maxd-" + str(maxd)) as run:
        # create a new Logistic Regression model.
        
        dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=maxd)        
        # put together the pipeline
        pipe = Pipeline(stages=[dt])

        # train the model
        model_pipeline = pipe.fit(trainingData)
        predictions = model_pipeline.transform(testData)

        # evaluate. note only 2 metrics are supported out of the box by Spark ML.
        bce = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction')
        au_roc = bce.setMetricName('areaUnderROC').evaluate(predictions)
        au_prc = bce.setMetricName('areaUnderPR').evaluate(predictions)
        
        print("Area under ROC: {}".format(au_roc))
        print("Area Under PR: {}".format(au_prc))
        
        run.log("truePositive", truePositive)
        run.log("falsePositive", falsePositive)
        run.log("trueNegative", trueNegative)
        run.log("falseNegative", falseNegative)
                                                                                                                                                                  
        print("TP: " + str(truePositive) + ", FP: " + str(falsePositive) + ", TN: " + str(trueNegative) + ", FN: " + str(falseNegative)) 

        # log reg, au_roc, au_prc and feature names in run history
        run.log("reg", reg)
        run.log("au_roc", au_roc)
        run.log("au_prc", au_prc)
        run.log("truePositive", truePositive)
        run.log("falsePositive", falsePositive)
        run.log("trueNegative", trueNegative)
        run.log("falseNegative", falseNegative)
        
        run.log_list("columns", trainingData.columns)

        # save model
        model_pipeline.write().overwrite().save(model_name)
        
        # upload the serialized model into run history record
        mdl, ext = model_name.split(".")
        model_zip = mdl + ".zip"
        shutil.make_archive(mdl, 'zip', model_dbfs)
        run.upload_file("outputs/" + model_name, model_zip)        
        #run.upload_file("outputs/" + model_name, path_or_stream = model_dbfs) #cannot deal with folders

        # now delete the serialized model from local folder since it is already uploaded to run history 
        shutil.rmtree(model_dbfs)
        os.remove(model_zip)

# COMMAND ----------

# Declare run completed
root_run.complete()
root_run_id = root_run.id
print ("run id:", root_run.id)
