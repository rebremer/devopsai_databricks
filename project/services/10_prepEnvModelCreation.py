import azureml.core
from azureml.core import Workspace
from azureml.core import Experiment
import base64
from azureml.core.authentication import AzureCliAuthentication
import requests
from azureml.core.compute import ComputeTarget, DatabricksCompute
from azureml.exceptions import ComputeTargetException
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import DatabricksStep
from azureml.core.datastore import Datastore
from azureml.data.data_reference import DataReference

def trigger_env_prep():

    # Define Vars < Change the vars>. 
    # In a production situation, don't put secrets in source code, but as secret variables, 
    # see https://docs.microsoft.com/en-us/azure/devops/pipelines/process/variables?view=azure-devops&tabs=yaml%2Cbatch#secret-variables
    workspace="<Name of your workspace>"
    subscription_id="<Subscription id>"
    resource_grp="<Name of your resource group where aml service is created>"

    domain = "westeurope.azuredatabricks.net" # change location in case databricks instance is not in westeurope
    databricks_name = "<<Your Databricks Name>>"
    dbr_pat_token_raw = "<<your Databricks Personal Access Token>>"

    DBR_PAT_TOKEN = bytes(dbr_pat_token_raw, encoding='utf-8') # adding b'
    dataset = "AdultCensusIncome.csv"
    notebook = "3_IncomeNotebookDevops.py"
    experiment_name = "experiment_model_release"
    db_compute_name="dbr-amls-comp"

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

    print("Add databricks env to Azure ML Service Compute")
    # Create databricks workspace in AML SDK
    try:
        databricks_compute = DatabricksCompute(workspace=ws, name=db_compute_name)
        print('Compute target {} already exists'.format(db_compute_name))
    except ComputeTargetException:
        print('Compute not found, will use below parameters to attach new one')
        config = DatabricksCompute.attach_configuration(
            resource_group = resource_grp,
            workspace_name = databricks_name,
            access_token= dbr_pat_token_raw)
        databricks_compute=ComputeTarget.attach(ws, db_compute_name, config)
        databricks_compute.wait_for_completion(True)

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