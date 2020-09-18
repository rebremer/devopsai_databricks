import json
import numpy
import time
import pyspark
from azureml.core.model import Model
from pyspark.ml import PipelineModel
from azureml.monitoring import ModelDataCollector

import shutil
import os

def init():
    try:
        # One-time initialization of PySpark and predictive model

        global trainedModel
        global spark

        global inputs_dc, prediction_dc
        model_name = "{model_name}"  # interpolated
        model_path = Model.get_model_path(model_name)

        # hack to such that model_path can work with zip file. Reason is the image can only be created using zip file. 
        # If no zip file was used, then every file in the unzipped file would create a layer and thus exceeding max depth layers (125) of docker
        mdl, ext = model_path.rsplit(".", 1)
        os.mkdir(mdl + ".tmp")
        shutil.unpack_archive(model_path, mdl + ".tmp")
        os.remove(model_path)
        os.rename(mdl + ".tmp", model_path)

        inputs_dc = ModelDataCollector(model_name, identifier="inputs",
                                       feature_names=["json_input_data"])
        prediction_dc = ModelDataCollector(model_name, identifier="predictions", feature_names=["predictions"])

        spark = pyspark.sql.SparkSession.builder.appName("AML Production Model").getOrCreate()
        trainedModel = PipelineModel.load(model_path)
    except Exception as e:
        trainedModel = e

def run(input_json):
    if isinstance(trainedModel, Exception):
        return json.dumps({"trainedModel": str(trainedModel)})
    try:

        sc = spark.sparkContext
        input_list = json.loads(input_json)
        input_rdd = sc.parallelize(input_list)
        input_df = spark.read.json(input_rdd)

        # Compute prediction
        prediction = trainedModel.transform(input_df)
        # result = prediction.first().prediction
        predictions = prediction.collect()

        # Get each scored result
        preds = [str(x['prediction']) for x in predictions]
        result = ",".join(preds)

        # log input and output data
        data = json.loads(input_json)
        data = numpy.array(data)
        print("saving input data" + time.strftime("%H:%M:%S"))
        inputs_dc.collect(data)  # this call is saving our input data into our blob
        prediction_dc.collect(predictions) #this call is saving our prediction data into our blob
    except Exception as e:
        result = str(e)
    return json.dumps({"result": result})