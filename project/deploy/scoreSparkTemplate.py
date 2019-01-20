

import json

def init():
    try:
        # One-time initialization of PySpark and predictive model
        import pyspark
        from azureml.core.model import Model
        from pyspark.ml import PipelineModel

        global trainedModel
        global spark

        spark = pyspark.sql.SparkSession.builder.appName("AML Production Model").getOrCreate()
        model_name = "{model_name}" #interpolated
        model_path = Model.get_model_path(model_name)
        trainedModel = PipelineModel.load(model_path)
    except Exception as e:
        trainedModel = e

def run(input_json):
    if isinstance(trainedModel, Exception):
        return json.dumps({"trainedModel":str(trainedModel)})
    try:
        sc = spark.sparkContext
        input_list = json.loads(input_json)
        input_rdd = sc.parallelize(input_list)
        input_df = spark.read.json(input_rdd)

        # Compute prediction
        prediction = trainedModel.transform(input_df)
        #result = prediction.first().prediction
        predictions = prediction.collect()

        #Get each scored result
        preds = [str(x['prediction']) for x in predictions]
        result = ",".join(preds)
    except Exception as e:
        result = str(e)
    return json.dumps({"result":result})

