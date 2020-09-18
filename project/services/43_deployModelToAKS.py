import azureml
import os, json, sys
from azureml.core import Workspace, Run
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.image import ContainerImage, Image
from azureml.core.webservice import Webservice, AksWebservice
from azureml.core.compute import AksCompute
from azureml.core.model import Model
from azureml.core.model import InferenceConfig
import sys

# display the core SDK version number
print("Azure ML SDK Version: ", azureml.core.VERSION)

# Define Vars < Change the vars>. 
# In a production situation, don't put secrets in source code, but as secret variables, 
# see https://docs.microsoft.com/en-us/azure/devops/pipelines/process/variables?view=azure-devops&tabs=yaml%2Cbatch#secret-variables
workspace=sys.argv[1]
subscription_id=sys.argv[2]
resource_grp=sys.argv[3]

model_name = "databricksmodel.mml" # in case you want to change the name, keep the .mml extension
aks_name = sys.argv[4]
service_name = sys.argv[5]

# Get the latest evaluation result
try:
    with open("conf/model.json") as f:
        config = json.load(f)
except:
    print("No new model to register thus no need to create new scoring image")
    # raise Exception('No new model to register as production model perform better')
    sys.exit(0)

model_name = config["model_name"]
model_version = config["model_version"]

# Start creating 
# Point file to conf directory containing details for the aml service
cli_auth = AzureCliAuthentication()
ws = Workspace(workspace_name = workspace,
               subscription_id = subscription_id,
               resource_group = resource_grp,
               auth=cli_auth)

model_list = Model.list(workspace=ws)
model, = (m for m in model_list if m.version == model_version and m.name == model_name)
print("Model picked: {} \nModel Description: {} \nModel Version: {}".format(model.name, model.description, model.version))

try:
    service = AksWebservice(name=service_name, workspace=ws)
    print("delete " + service_name + " before creating new one")
    service.delete()
except:
    print(service_name + " does not exist yet")


os.chdir("deploy")

# Add model name to scorefile
with open("scoreSparkTemplate.py") as fr:
    score = fr.read()

score = score.replace("{model_name}", model_name)
#
with open("scoreSpark.py", "w") as fw:
    fw.write(score)

inference_config = InferenceConfig(entry_script='scoreSpark.py', runtime='spark-py', conda_file='myenv.yml')
deployment_config = AksWebservice.deploy_configuration(auth_enabled=False, collect_model_data=True, enable_app_insights=True, cpu_cores = 2, memory_gb = 2)
aks_target = AksCompute(ws,aks_name)

service = Model.deploy(ws, service_name, [model], inference_config, deployment_config, aks_target)
service.wait_for_deployment(show_output = True)

print(service.state)
print(service.scoring_uri)


aks_webservice = {}
aks_webservice['aks_name'] = service.name
aks_webservice['aks_url'] = service.scoring_uri
with open('../conf/' + service_name + '.json', 'w') as outfile:
  json.dump(aks_webservice,outfile)