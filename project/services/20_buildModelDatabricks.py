import sys
import requests
import time
import base64
import datetime
import azureml.core
import shutil
import os, json
from azureml.core import Workspace
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.run import Run
from azureml.core.experiment import Experiment
from azureml.core.model import Model

from azureml.core.compute import ComputeTarget, DatabricksCompute
from azureml.exceptions import ComputeTargetException
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import DatabricksStep
from azureml.core.datastore import Datastore
from azureml.data.data_reference import DataReference

def trigger_training_job():

    # Define Vars < Change the vars>. 
    # In a production situation, don't put secrets in source code, but as secret variables, 
    # see https://docs.microsoft.com/en-us/azure/devops/pipelines/process/variables?view=azure-devops&tabs=yaml%2Cbatch#secret-variables
    workspace="<Name of your workspace>"
    subscription_id="<Subscription id>"
    resource_grp="<Name of your resource group where aml service is created>"

    domain = "westeurope.azuredatabricks.net" # change location in case databricks instance is not in westeurope
    databricks_name = "<<Your Databricks Service Name>>"
    dbr_pat_token_raw = "<<your Databricks Personal Access Token>>"

    DBR_PAT_TOKEN = bytes(dbr_pat_token_raw, encoding='utf-8') # adding b'
    notebookRemote = "/3_IncomeNotebookDevops"
    experiment_name = "experiment_model_release"
    model_name_run = datetime.datetime.now().strftime("%Y%m%d%H%M%S")+ "_dbrmod.mml" # in case you want to change the name, keep the .mml extension
    model_name = "databricksmodel.mml" # in case you want to change the name, keep the .mml extension
    db_compute_name="dbr-amls-comp"

    #
    # Step 1: Run notebook using Databricks Compute in AML SDK
    #
    cli_auth = AzureCliAuthentication()

    ws = Workspace(workspace_name = workspace,
               subscription_id = subscription_id,
               resource_group = resource_grp,
               auth=cli_auth)
    ws.get_details()

    try:
        databricks_compute = DatabricksCompute(workspace=ws, name=db_compute_name)
    except:
        print ("Compute not found")

    dbNbStep = DatabricksStep(
        name="DBNotebookInAMLS",
        num_workers=1,
        notebook_path=notebookRemote,
        notebook_params={"model_name": model_name_run},
        run_name='AMLS Notebook production' + model_name_run,
        compute_target=databricks_compute,
        allow_reuse=False
    )
    #
    steps = [dbNbStep]
    pipeline = Pipeline(description="Build model in production", workspace=ws, steps=steps)
    pipeline_run = Experiment(ws, "dbr-pipeline-run").submit(pipeline)
    pipeline_run.wait_for_completion()

    #
    # Step 2: Retrieve model from dbfs
    #
    mdl, ext = model_name_run.split(".")
    model_zip_run = mdl + ".zip"
    
    response = requests.get(
        'https://%s/api/2.0/dbfs/read?path=/%s' % (domain, model_zip_run),
        headers={'Authorization': b"Bearer " + DBR_PAT_TOKEN}
    )
    if response.status_code != 200:
        print("Error copying dbfs results: %s: %s" % (response.json()["error_code"], response.json()["message"]))
        exit(1)

    model_output = base64.b64decode(response.json()['data'])

    # download model in deploy folder
    os.chdir("deploy")
    with open(model_zip_run, "wb") as outfile:
        outfile.write(model_output)
    print("Downloaded model {} to Project root directory".format(model_name))

    #
    # Step 3: Retrieve model metrics from dbfs
    #
    mdl, ext = model_name_run.split(".")
    model_metrics_json_run = mdl + "_metrics.json"
    
    response = requests.get(
        'https://%s/api/2.0/dbfs/read?path=/%s' % (domain, model_metrics_json_run),
        headers={'Authorization': b"Bearer " + DBR_PAT_TOKEN}
    )
    if response.status_code != 200:
        print("Error copying dbfs results: %s: %s" % (response.json()["error_code"], response.json()["message"]))
        exit(2)

    model_metrics_output = json.loads(base64.b64decode(response.json()['data']))

    #
    # Step 4: Put model and metrics to Azure ML Service
    #

    # start a training run by defining an experiment
    myexperiment = Experiment(ws, experiment_name)
    run = myexperiment.start_logging()
    run.upload_file("outputs/" + model_zip_run, model_zip_run)

    run.log("pipeline_run", pipeline_run.id)
    run.log("au_roc", model_metrics_output["Area_Under_ROC"])
    run.log("au_prc", model_metrics_output["Area_Under_PR"])
    run.log("truePostive", model_metrics_output["True_Positives"])
    run.log("falsePostive", model_metrics_output["False_Positives"])
    run.log("trueNegative", model_metrics_output["True_Negatives"])
    run.log("falseNegative", model_metrics_output["False_Negatives"])   

    run.complete()
    run_id = run.id
    print ("run id:", run_id)

    # unzip file to model_name_run
    shutil.unpack_archive(model_zip_run, model_name_run)

    model = Model.register(
        model_path=model_name_run,  # this points to a local file
        model_name=model_name,  # this is the name the model is registered as
        tags={"area": "spar", "type": "regression", "run_id": run_id},
        description="Medium blog test model",
        workspace=ws,
    )
    print("Model registered: {} \nModel Description: {} \nModel Version: {}".format(model.name, model.description, model.version))

    # Step 5. Finally, writing the registered model details to conf/model.json
    model_json = {}
    model_json["model_name"] = model.name
    model_json["model_version"] = model.version
    model_json["run_id"] = run_id
    model_json["model_name_run"] = model_name_run
    with open("../conf/model.json", "w") as outfile:
        json.dump(model_json, outfile)

def main():
    trigger_training_job()

if __name__ == "__main__":
    main()