import requests
import time

def trigger_training_job(script_folder, notebook):

    # Define Vars < Change the vars>
    tenant_id="<Enter Your Tenant Id>"
    app_id="<Application Id of the SPN you Create>"
    app_key= "<Key for the SPN>"
    workspace="<Name of your workspace>"
    subscription_id="<Subscription id>"
    resource_grp="<Name of your resource group where aml service is created>"

    domain = "westeurope.azuredatabricks.net" # change location in case databricks instance is not in westeurope
    DBR_PAT_TOKEN = bytes("<<your Databricks Personal Access Token>>", encoding='utf-8')  # adding b'

    notebookRemote = "/3_IncomeNotebookDevops"
    experiment_name = "experiment_model_release"
    model_name = "databricksmodel.mml" # in case you want to change the name, keep the .mml extension

    #
    # Step 1: Create job and attach it to cluster
    #
    response = requests.post(
        'https://%s/api/2.0/jobs/create' % domain,
        headers={'Authorization': b"Bearer " + DBR_PAT_TOKEN},
        json={
                "name": "Run AzureDevopsNotebook Job",
                "new_cluster": {
                    "spark_version": "4.0.x-scala2.11",
                    "node_type_id": "Standard_D3_v2",
                    "spark_env_vars": {
                        'PYSPARK_PYTHON': '/databricks/python3/bin/python3',
                    },
                    "autoscale": {
                        "min_workers": 1,
                        "max_workers": 2
                    }
                },
                "libraries": [
                   {
                     "pypi": {
                        "package": "azureml-sdk[databricks]"
                     }
                  }
                ],
                "notebook_task": {
                "notebook_path": notebookRemote,
                "base_parameters": [{"key":"subscription_id", "value":subscription_id}, {"key":"resource_group", "value":resource_grp}, {"key":"workspace_name","value":workspace},
                                    {"key":"experiment_name", "value":experiment_name}, {"key":"model_name", "value":model_name},
                                    {"key": "spn_tenant", "value": tenant_id},{"key": "spn_clientid", "value": app_id}, {"key": "spn_clientsecret", "value": app_key}
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

def main():
    script_folder = "../modelling"
    notebook = "/trainDBrNotebook"
    trigger_training_job(script_folder, notebook)

if __name__ == "__main__":
    main()