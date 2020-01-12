import azureml.core
from azureml.core import Workspace
from azureml.core import Experiment
import base64
from azureml.core.authentication import AzureCliAuthentication
import requests
import sys

def trigger_env_prep():

    # Define Vars < Change the vars>. 
    # In a production situation, don't put secrets in source code, but as secret variables, 
    # see https://docs.microsoft.com/en-us/azure/devops/pipelines/process/variables?view=azure-devops&tabs=yaml%2Cbatch#secret-variables
    workspace=sys.argv[1]
    subscription_id=sys.argv[2]
    resource_grp=sys.argv[3]

    domain = sys.argv[4]
    dbr_pat_token_raw = sys.argv[5]

    DBR_PAT_TOKEN = bytes(dbr_pat_token_raw, encoding='utf-8') # adding b'
    dataset = "AdultCensusIncome.csv"
    notebook = "3_IncomeNotebookDevops.py"
    experiment_name = "experiment_model_release"

    # Print AML Version
    print("Azure ML SDK Version: ", azureml.core.VERSION)

    # Point file to conf directory containing details for the aml service

    cli_auth = AzureCliAuthentication()
    ws = Workspace(workspace_name = workspace,
                   subscription_id = subscription_id,
                   resource_group = resource_grp,
                   auth=cli_auth)

    print(ws.name, ws._workspace_name, ws.resource_group, ws.location, sep = '\t')

    # Create a new experiment
    print("Starting to create new experiment")
    Experiment(workspace=ws, name=experiment_name)

    # Upload notebook to Databricks

    print("Upload notebook to databricks")
    upload_notebook(domain, DBR_PAT_TOKEN, notebook)

def upload_notebook(domain, DBR_PAT_TOKEN, notebook):
    # Upload notebook to Databricks
    print("Upload notebook to Databricks DBFS")
    with open("modelling/" + notebook) as f:
        notebookContent = f.read()

    # Encode notebook to base64
    string = base64.b64encode(bytes(notebookContent, 'utf-8'))
    notebookContentb64 = string.decode('utf-8')
    print(notebookContentb64)

    notebookName, ext = notebook.split(".")
    print(notebookName)

    # Copy notebook to Azure Databricks using REST API
    response = requests.post(
        'https://%s/api/2.0/workspace/import' % domain,
        headers={'Authorization': b"Bearer " + DBR_PAT_TOKEN},
        json={
            "content": notebookContentb64,
            "path": "/" + notebookName,
            "language": "PYTHON",
            "overwrite": "true",
            "format": "SOURCE"
        }
    )
    # TBD: Expecting synchroneous result. Only result back when data is completely copied
    if response.status_code != 200:
        print("Error copying notebook: %s: %s" % (response.json()["error_code"], response.json()["message"]))
        exit(1)

if __name__ == "__main__":
    trigger_env_prep()