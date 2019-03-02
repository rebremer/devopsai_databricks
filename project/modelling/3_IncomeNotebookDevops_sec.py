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
# MAGIC The compact version of this notebook can be found in the same folder as Extensive_predictionIncomeLevel, in which only step 1 and step 7 are executed.

# COMMAND ----------

par_model_name= dbutils.widgets.get("model_name")
par_stor2_name = dbutils.widgets.get("stor2_name")
par_stor2_container = dbutils.widgets.get("stor2_container")
par_stor2_datafile = dbutils.widgets.get("stor2_data_file")
par_secret_scope = dbutils.widgets.get("secret_scope")

par_stor2_key = dbutils.secrets.get(scope = par_secret_scope, key = "stor-key")

# COMMAND ----------

import os
import urllib
import pprint
import numpy as np
import shutil

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import OneHotEncoder, OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# COMMAND ----------

#Authenticate to storage account
spark.conf.set("fs.azure.account.key." + par_stor2_name + ".dfs.core.windows.net", par_stor2_key)
spark.conf.set("fs.azure.createRemoteFileSystemDuringInitialization", "true")
dbutils.fs.ls("abfss://" + par_stor2_container + "@" + par_stor2_name + ".dfs.core.windows.net/")
spark.conf.set("fs.azure.createRemoteFileSystemDuringInitialization", "false")

# Create a Spark dataframe out of the csv file.
data_all = sqlContext.read.format('csv').options(header='true', inferSchema='true', ignoreLeadingWhiteSpace='true', ignoreTrailingWhiteSpace='true').load("abfss://" + par_stor2_container + "@" + par_stor2_name + ".dfs.core.windows.net/" + par_stor2_datafile)


print("({}, {})".format(data_all.count(), len(data_all.columns)))

#renaming columns, all columns that contain a - will be replaced with an "_"
columns_new = [col.replace("-", "_") for col in data_all.columns]
data_all = data_all.toDF(*columns_new)

data_all.printSchema()

# COMMAND ----------

(trainingData, testData) = data_all.randomSplit([0.7, 0.3], seed=122423)

# COMMAND ----------

label = "income"
dtypes = dict(trainingData.dtypes)
dtypes.pop(label)

si_xvars = []
ohe_xvars = []
featureCols = []
for idx,key in enumerate(dtypes):
    if dtypes[key] == "string":
        featureCol = "-".join([key, "encoded"])
        featureCols.append(featureCol)
        
        tmpCol = "-".join([key, "tmp"])
        # string-index and one-hot encode the string column
        #https://spark.apache.org/docs/2.3.0/api/java/org/apache/spark/ml/feature/StringIndexer.html
        #handleInvalid: Param for how to handle invalid data (unseen labels or NULL values). 
        #Options are 'skip' (filter out rows with invalid data), 'error' (throw an error), 
        #or 'keep' (put invalid data in a special additional bucket, at index numLabels). Default: "error"
        si_xvars.append(StringIndexer(inputCol=key, outputCol=tmpCol, handleInvalid="skip"))
        ohe_xvars.append(OneHotEncoder(inputCol=tmpCol, outputCol=featureCol))
    else:
        featureCols.append(key)

# string-index the label column into a column named "label"
si_label = StringIndexer(inputCol=label, outputCol='label')

# assemble the encoded feature columns in to a column named "features"
assembler = VectorAssembler(inputCols=featureCols, outputCol="features")

# COMMAND ----------

model_dbfs = os.path.join("/dbfs", par_model_name)

# COMMAND ----------

# Regularization Rates
from pyspark.ml.classification import LogisticRegression

# try a bunch of alpha values in a Linear Regression (Ridge) model
reg=0
print("Regularization rate: {}".format(reg))
# create a bunch of child runs
#with root_run.child_run("reg-" + str(reg)) as run:
# create a new Logistic Regression model.
        
lr = LogisticRegression(regParam=reg)
        
# put together the pipeline
pipe = Pipeline(stages=[*si_xvars, *ohe_xvars, si_label, assembler, lr])

# train the model
model_pipeline = pipe.fit(trainingData)
        
# make prediction
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
#run.log("reg", reg)
#run.log("au_roc", au_roc)
#run.log("au_prc", au_prc)
        
print("Area under ROC: {}".format(au_roc))
print("Area Under PR: {}".format(au_prc))
       
#    run.log("truePositive", truePositive)
#    run.log("falsePositive", falsePositive)
#    run.log("trueNegative", trueNegative)
#    run.log("falseNegative", falseNegative)
                                                                                                                                                                  
print("TP: " + str(truePositive) + ", FP: " + str(falsePositive) + ", TN: " + str(trueNegative) + ", FN: " + str(falseNegative))                                                                         
        
#    run.log_list("columns", trainingData.columns)

# save model
model_pipeline.write().overwrite().save(par_model_name)
        
# upload the serialized model into run history record
mdl, ext = par_model_name.split(".")
model_zip = mdl + ".zip"
shutil.make_archive('/dbfs/'+ mdl, 'zip', model_dbfs)
##    run.upload_file("outputs/" + par_model_name, model_zip)        
    #run.upload_file("outputs/" + model_name, path_or_stream = model_dbfs) #cannot deal with folders

    # now delete the serialized model from local folder since it is already uploaded to run history 
    #shutil.rmtree(model_dbfs)
    #os.remove(model_zip)


# COMMAND ----------

# Declare run completed
#root_run.complete()
#root_run_id = root_run.id
#print ("run id:", root_run.id)

# COMMAND ----------

#Register the model already in the Model Managment of azure ml service workspace
#from azureml.core.model import Model
#mymodel = Model.register(model_path = "/dbfs/" + par_model_name, # this points to a local file
#                       model_name = par_model_name, # this is the name
#                       description = "testrbdbr",
#                       workspace = ws)
#print(mymodel.name, mymodel.id, mymodel.version, sep = '\t')

# COMMAND ----------


