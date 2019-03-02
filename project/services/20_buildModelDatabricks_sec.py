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

def trigger_training_job():

    # get the parameter values
    workspace = sys.argv[1]
    subscription_id = sys.argv[2]
    resource_grp = sys.argv[3]

    domain = sys.argv[4] # change location in case databricks instance is not in westeurope
    DBR_PAT_TOKEN = bytes(sys.argv[5], encoding='utf-8')  # adding b'
    
    stor2_name = sys.argv[6]
    stor2_container = sys.argv[7]
    secret_scope = sys.argv[8]

    dataset = "AdultCensusIncome.csv"
    notebookRemote = "/3_IncomeNotebookDevops_sec"
    experiment_name = "experiment_model_release"
    model_name_run = datetime.datetime.now().strftime("%Y%m%d%H%M%S")+ "_dbrmod.mml" # in case you want to change the name, keep the .mml extension
    model_name = "databricksmodel.mml"
    #
    # Step 1: Create job and attach it to cluster
    #
    response = requests.post(
        'https://%s/api/2.0/jobs/create' % domain,
        headers={'Authorization': b"Bearer " + DBR_PAT_TOKEN},
        json={
                "name": "Run AzureDevopsNotebook Job",
                "new_cluster": {
                    "spark_version": "5.2.x-scala2.11",
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
                "base_parameters": [{"key":"model_name", "value":model_name_run},
                                    {"key":"stor2_name", "value":stor2_name}, {"key":"stor2_container", "value":stor2_container},{"key":"stor2_data_file", "value":dataset},
                                    {"key":"secret_scope", "value":secret_scope}
                                   ]
             }
        }
    )

    if response.status_code != 200:
        print("Error launching cluster: %s: %s" % (response.json()["error_code"], response.json()["message"]))
        exit(2)

    #
    # Step 2: Start job
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
    # Step 3: Wait until job is finished
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
    # Step 4: Retrieve model from dbfs
    #
    mdl, ext = model_name_run.split(".")
    model_zip_run = mdl + ".zip"
    
    response = requests.get(
        'https://%s/api/2.0/dbfs/read?path=/%s' % (domain, model_zip_run),
        headers={'Authorization': b"Bearer " + DBR_PAT_TOKEN}
    )
    if response.status_code != 200:
        print("Error copying dbfs results: %s: %s" % (response.json()["error_code"], response.json()["message"]))
        exit(6)

    model_output = base64.b64decode(response.json()['data'])

    # download model in deploy folder
    os.chdir("deploy")
    with open(model_zip_run, "wb") as outfile:
        outfile.write(model_output)
    print("Downloaded model {} to Project root directory".format(model_name))

    #
    # Step 5: Put model to Azure ML Service
    #
    cli_auth = AzureCliAuthentication()

    ws = Workspace(workspace_name = workspace,
               subscription_id = subscription_id,
               resource_group = resource_grp,
               auth=cli_auth)
    ws.get_details()
    # start a training run by defining an experiment
    myexperiment = Experiment(ws, experiment_name)
    run = myexperiment.start_logging()
    run.upload_file("outputs/" + model_zip_run, model_zip_run)
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

    # Step 6. Finally, writing the registered model details to conf/model.json
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