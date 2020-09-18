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
import sys

def trigger_training_job():

    # Define Vars < Change the vars>. 
    # In a production situation, don't put secrets in source code, but as secret variables, 
    # see https://docs.microsoft.com/en-us/azure/devops/pipelines/process/variables?view=azure-devops&tabs=yaml%2Cbatch#secret-variables
    workspace=sys.argv[1]
    subscription_id=sys.argv[2]
    resource_grp=sys.argv[3]

    domain = sys.argv[4]
    dbr_pat_token_raw = sys.argv[5]

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


    #
    # Step 2: Create job and attach it to cluster
    #
    # In this steps, secret are added as parameters (spn_tenant, spn_clientid, spn_clientsecret)
    # Never do this in a production situation, but use secret scope backed by key vault instead
    # See https://docs.azuredatabricks.net/user-guide/secrets/secret-scopes.html#azure-key-vault-backed-scopes
    response = requests.post(
        'https://%s/api/2.0/jobs/create' % domain,
        headers={'Authorization': b"Bearer " + DBR_PAT_TOKEN},
        json={
                "name": "Run AzureDevopsNotebook Job",
                "new_cluster": {
                    "spark_version": "6.6.x-scala2.11",
                    "node_type_id": "Standard_D3_v2",
                    "spark_env_vars": {
                        'PYSPARK_PYTHON': '/databricks/python3/bin/python3',
                    },
                    "autoscale": {
                        "min_workers": 1,
                        "max_workers": 2
                    }
                },
                "notebook_task": {
                "notebook_path": notebookRemote,
                "base_parameters": [{"key":"subscription_id", "value":subscription_id}, {"key":"resource_group", "value":resource_grp}, {"key":"workspace_name","value":workspace},
                                    {"key":"model_name", "value":model_name_run}
                                   ]
             }
        }
    )

    if response.status_code != 200:
        print("Error launching cluster: %s: %s" % (response.json()["error_code"], response.json()["message"]))
        exit(2)

    #
    # Step 3: Start job
    #
    databricks_job_id = response.json()['job_id']

    response = requests.post(
        'https://%s/api/2.0/jobs/run-now' % domain,
        headers={'Authorization': b"Bearer " + DBR_PAT_TOKEN},
        json={
            "job_id": + databricks_job_id
        }
    )

    if response.status_code != 200:
        print("Error launching cluster: %s: %s" % (response.json()["error_code"], response.json()["message"]))
        exit(3)

    print(response.json()['run_id'])

    #
    # Step 4: Wait until job is finished
    #
    databricks_run_id = response.json()['run_id']
    scriptRun = 1
    count = 0
    while scriptRun == 1:
        response = requests.get(
            'https://%s/api/2.0/jobs/runs/get?run_id=%s' % (domain, databricks_run_id),
            headers={'Authorization': b"Bearer " + DBR_PAT_TOKEN},
        )

        state = response.json()['state']
        life_cycle_state = state['life_cycle_state']
        print(state)

        if life_cycle_state in ["TERMINATED", "SKIPPED", "INTERNAL_ERROR"]:
            result_state = state['result_state']
            if result_state == "SUCCESS":
                print("run ok")
                scriptRun = 0
            #exit(0)
            else:
                exit(4)
        elif count > 180:
            print("time out occurred after 30 minutes")
            exit(5)
        else:
            count += 1
            time.sleep(30) # wait 30 seconds before next status update

    #
    # Step 5: Retrieve model from dbfs
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
    # Step 6: Retrieve model metrics from dbfs
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
    # Step 7: Put model and metrics to Azure ML Service
    #

    # start a training run by defining an experiment
    myexperiment = Experiment(ws, experiment_name)
    run = myexperiment.start_logging()
    run.upload_file("outputs/" + model_zip_run, model_zip_run)

    #run.log("pipeline_run", pipeline_run.id)
    run.log("au_roc", model_metrics_output["Area_Under_ROC"])
    run.log("au_prc", model_metrics_output["Area_Under_PR"])
    run.log("truePostive", model_metrics_output["True_Positives"])
    run.log("falsePostive", model_metrics_output["False_Positives"])
    run.log("trueNegative", model_metrics_output["True_Negatives"])
    run.log("falseNegative", model_metrics_output["False_Negatives"])   

    run.complete()
    run_id = run.id
    print ("run id:", run_id)

    # Register the model as zip file. The model will be unzipped in the init of the score.py. 
    # In case no zip file is used and the entire directory is uploaded, this results that more than 125 layers
    # are created in the docker image creation. This results docker max depth exceeded and the docker build fails
    model = Model.register(
        model_path=model_zip_run,  # this points to a local file
        model_name=model_name,  # this is the name the model is registered as
        tags={"area": "spar", "type": "regression", "run_id": run_id},
        description="Medium blog test model",
        workspace=ws,
    )
    print("Model registered: {} \nModel Description: {} \nModel Version: {}".format(model.name, model.description, model.version))

    # Step 8. Finally, writing the registered model details to conf/model.json
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