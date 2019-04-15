# Databricks notebook source
# MAGIC %md Azure ML & Azure Databricks notebooks by René Bremer (original taken from Databricks website)
# MAGIC 
# MAGIC Copyright (c) Microsoft Corporation. All rights reserved.
# MAGIC 
# MAGIC Licensed under the MIT License.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### In this notebook the following steps will be excuted:
# MAGIC 1. Mount blob storage
# MAGIC 2. Unzip pictures in storage account
# MAGIC 3. List and show pictures

# COMMAND ----------

# MAGIC %md #0. Set parameters

# COMMAND ----------

storageaccount="<<your_name_of_storare_account>>"
account_key="<<your_account_key_of_storare_account>>"

containername="2000picscifar10"
mountname=containername

# COMMAND ----------

# MAGIC %md #1. Mount blob storage

# COMMAND ----------

if any(mount.mountPoint == "/mnt/" + mountname for mount in dbutils.fs.mounts()):
  print ("directory " + "/mnt/" + mountname + " is already mounted")
else:
  print ("In case you have a cluster with 0 workers, you need to cancell statement manually after 30 seconds. This is because a spark job is started, which cannot be executed since there are 0 workers. However, the storage is mounted, which can be verified by rerunning cell")
  dbutils.fs.mount(
  source = "wasbs://" + containername + "@" + storageaccount + ".blob.core.windows.net",
  mount_point = "/mnt/" + mountname,
  extra_configs = {"fs.azure.account.key." + storageaccount +".blob.core.windows.net":account_key})

# COMMAND ----------

# MAGIC %md #2. Unzip pictures

# COMMAND ----------

import zipfile
import os
datafile = "2000pics_cifar10.zip"
datafile_dbfs = os.path.join("/dbfs/mnt/" + mountname, datafile)

print ("unzipping takes approximatelly 2 minutes")
zip_ref = zipfile.ZipFile(datafile_dbfs, 'r')
zip_ref.extractall("/dbfs/mnt/" + mountname)
zip_ref.close()

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /dbfs/mnt/2000picscifar10

# COMMAND ----------

# MAGIC %md #3. Show pictures

# COMMAND ----------

import matplotlib.pyplot as plt
import random
import numpy as np
from PIL import Image
import os

categoriesList=["airplane","automobile","bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def plotImagesMount(n_images=8):
    fig, axes = plt.subplots(n_images, n_images, figsize=(9,9))
    axes = axes.flatten()
    
    for i in range(n_images * n_images):
        rand1 = random.randint(0, 9)
        rand2 = random.randint(1, 200)
        filename=str(categoriesList[rand1]) + "_" + str(rand2)
        filenamepng=filename + ".png"
        img = Image.open(os.path.join("/dbfs/mnt/" + mountname + "/", filenamepng))
        ax = axes[i]
    
        ax.imshow( img, cmap="Greys_r")
        ax.set_xticks(())
        ax.set_yticks(())
        
        ax.set_title(filename, fontsize=18 - n_images)
        
    plot = plt.tight_layout()
    return plot
  
display(plotImagesMount(n_images=10))

# COMMAND ----------


