# Databricks notebook source
# MAGIC %md Azure ML & Azure Databricks notebooks by René Bremer (original taken from Parashar Shah)
# MAGIC 
# MAGIC Copyright (c) Microsoft Corporation. All rights reserved.
# MAGIC 
# MAGIC Licensed under the MIT License.

# COMMAND ----------

# MAGIC %md ##### In this notebook the following steps will be excuted:
# MAGIC 
# MAGIC 1. Log metrics of models that was trained on 2000 pictures and all 60000 pictures
# MAGIC 2. Register best model (trained with 60000 pictures)
# MAGIC 
# MAGIC Make sure you added libraries to azureml-sdk[databricks], Keras and TensorFlow to your cluster.

# COMMAND ----------

# MAGIC %md #0. Set parameters

# COMMAND ----------

workspace="<<your_name_of_azure_ml_service_workspace>>"
resource_grp="<<your_resource_group_amlservice>>"
subscription_id="<<your_subscriptionid_amlservice>>"


path= '/dbfs/CIFAR10/models/'
par_model2000_name = 'cifar_2000pictures.h5'
par_modelall_name = 'cifar_allpictures.h5' 

par_experiment_name = 'cifar10'

# In case cell gets status "cancelled" after execution, uninstall libraries, restart cluster and reinstall libraries

# COMMAND ----------

# MAGIC %md #1.  Log metrics of models

# COMMAND ----------

# MAGIC %md ##### 1a. Authenticate to Azure ML workspace (interactive, using AAD and browser)

# COMMAND ----------

import sys
import requests
import time
import base64
import datetime
import azureml.core
import shutil
import os, json
from azureml.core import Workspace
from azureml.core.run import Run
from azureml.core.experiment import Experiment
from azureml.core.model import Model
import azureml.core
from azureml.core.authentication import ServicePrincipalAuthentication

ws = Workspace(workspace_name = workspace,
               subscription_id = subscription_id,
               resource_group = resource_grp)

ws.get_details()

# COMMAND ----------

# MAGIC %md ##### 1b. Load models from disk where it was stored in previous notebook 1_DeepLearningCifar!0NotebookExploration.py

# COMMAND ----------

import keras
from keras.models import load_model

#path= '/dbfs/CIFAR10/models/'
model2000path = path + par_model2000_name
modelallpath = path + par_modelall_name

model2000 = load_model(model2000path)
modelall = load_model(modelallpath)

# COMMAND ----------

model2000.summary()

# COMMAND ----------

# MAGIC %md ##### 2c. Get testdate to regenerate metrics

# COMMAND ----------

from keras.datasets import cifar10
num_classes = 10
# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
#2
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#3
x_train /= 255
x_test /= 255

# COMMAND ----------

# MAGIC %md ##### 2d. Create new experiment in Azure ML service workspace and add both models

# COMMAND ----------

# upload the serialized model into run history record
#mdl, ext = par_model_name.split(".")
#model_zip = mdl + ".zip"
#shutil.make_archive('/dbfs/'+ mdl, 'zip', path)
# start a training run by defining an experiment
myexperiment = Experiment(ws, par_experiment_name)
run = myexperiment.start_logging()

score2000 = model2000.evaluate(x_test, y_test, verbose=0)
scoreall = modelall.evaluate(x_test, y_test, verbose=0)

run.log_list("Test accuracy 2000 pics, all pics", [score2000[1], scoreall[1]])
run.log_list("Test loss 2000 pics, all pics", [score2000[0], scoreall[0]])

run.upload_file("outputs/" + par_model2000_name, model2000path)
run.upload_file("outputs/" + par_modelall_name, modelallpath)

run.complete()
run_id = run.id
print ("run id:", run_id)

# COMMAND ----------

# MAGIC %md #2. Register best model (trained with 60000 pictures)

# COMMAND ----------

registermodelall = Model.register(
    model_path=modelallpath,  # this points to a local file
    model_name=par_modelall_name,  # this is the name the model is registered as
    tags={"area": "spark", "type": "deeplearning", "run_id": run_id},
    description="Keras deeplearning, all pictures",
    workspace=ws,
)
print("Model registered: {} \nModel Description: {} \nModel Version: {}".format(registermodelall.name, registermodelall.description, registermodelall.version))
