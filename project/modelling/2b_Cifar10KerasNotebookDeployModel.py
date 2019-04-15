# Databricks notebook source
# MAGIC %md Azure ML & Azure Databricks notebooks by René Bremer (original taken from Parashar Shah)
# MAGIC 
# MAGIC Copyright (c) Microsoft Corporation. All rights reserved.
# MAGIC 
# MAGIC Licensed under the MIT License.

# COMMAND ----------

# MAGIC %md ##### In this notebook the following steps will be excuted:
# MAGIC 
# MAGIC 1. Create endpoint of best model (trained with 60000 pictures)
# MAGIC 
# MAGIC Make sure you added libraries to azureml-sdk[databricks], Keras and TensorFlow to your cluster.

# COMMAND ----------

# MAGIC %md #0. Set parameters

# COMMAND ----------

workspace="<<your_name_of_azure_ml_service_workspace>>"
resource_grp="<<your_resource_group_amlservice>>"
subscription_id="<<your_subscriptionid_amlservice>>"

par_model_name = 'cifar_allpictures.h5' 
par_service_name = 'cifar10'

# In case cell gets status "cancelled" after execution, uninstall libraries, restart cluster and reinstall libraries

# COMMAND ----------

# MAGIC %md #1. Create endpoint of best model (trained with 60000 pictures)

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

# MAGIC %md ##### 1b. Retrieve best model from Azure ML Service

# COMMAND ----------

model=Model(ws,par_model_name)
model_list = Model.list(workspace=ws)
print("Model picked: {} \nModel Description: {} \nModel Version: {}".format(model.name, model.description, model.version))

# COMMAND ----------

# MAGIC %md ##### 1c. Create score file (script that will be used in endpoint to consume png) and conda env

# COMMAND ----------

#%%writefile score_deeplearning.py
score_deeplearning = """

import json

from azureml.core.model import Model
from keras.models import load_model
from io import BytesIO
import numpy as np
from PIL import Image
from base64 import b64decode

def init():
    global trainedModel
    # retreive the path to the model file using the model name
    # This needs to be the name of your model you registered in EstimatorTrigger.py
    print("Load model")
    model_name = "{model_name}"  # interpolated
    model_path = Model.get_model_path(model_name)
    trainedModel = load_model(model_path)
    print("model loaded")

def run(raw_data):
    print("base64 picture received")
    imagebase64=json.loads(raw_data)['imagebase64']
    img = Image.open(BytesIO(b64decode(imagebase64)))
    new_img = white_bg_square(img)
    resized_img=new_img.resize((32, 32), Image.ANTIALIAS)
    x_data = np.asarray(resized_img)
    x_data = x_data.astype('float32')
    x_data /= 255    
    print("make prediction")
    input_data = []
    input_data.append(x_data)
    predictions = trainedModel.predict_classes([[input_data[0]]])

    categoriesList = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    print("create label prediction")
    label=categoriesList[predictions[0]]
    print("label: " +  label)
    return json.dumps({{"result":label}})

def white_bg_square(img):
    "return a white-background-color image having the img in exact center"
    size = int(img.size[0]), int(img.size[1]) # (int(max(img.size)),)*2
    layer = Image.new('RGB', size, (255,255,255))
    imgsizeint = int(img.size[0]), int(img.size[1])
    layer.paste(img, tuple(map(lambda x:int((x[0]-x[1])/2), zip(size, imgsizeint))))
    return layer

""".format(model_name=par_model_name)

exec(score_deeplearning)

with open("score_deeplearning.py", "w") as file:
    file.write(score_deeplearning)

# COMMAND ----------

from azureml.core.conda_dependencies import CondaDependencies 

myacienv = CondaDependencies.create(conda_packages=['scikit-learn', 'keras','numpy','Pillow'])

with open("deeplearningenv.yml","w") as f:
    f.write(myacienv.serialize_to_string())

# COMMAND ----------

# MAGIC %md ##### 1d. Deploy model and create endpoint

# COMMAND ----------

try:
    oldservice = Webservice(workspace=ws, name=par_service_name)
    print("delete " + par_service_name + " before creating new one")
    oldservice.delete()
except:
    print(par_service_name + " does not exist, create new one")

# COMMAND ----------

from azureml.core.image import ContainerImage
from azureml.core.webservice import AciWebservice, Webservice

image_config = ContainerImage.image_configuration(execution_script="score_deeplearning.py",
                                                  runtime="python",
                                                  conda_file="deeplearningenv.yml")

aci_config = AciWebservice.deploy_configuration(
    cpu_cores = 2,
    memory_gb = 4,
    tags = {'name':'Databricks ALM ACI'},
    description = 'AML Deployment Production')


# COMMAND ----------

service = Webservice.deploy_from_model(
  workspace=ws,
  name=par_service_name,
  deployment_config = aci_config,
  models = [model],
  image_config = image_config
    )

service.wait_for_deployment(show_output=True)
