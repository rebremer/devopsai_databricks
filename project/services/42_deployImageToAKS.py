import azureml
import os, json, sys
from azureml.core import Workspace, Run
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.image import ContainerImage, Image
from azureml.core.webservice import Webservice, AksWebservice
from azureml.core.compute import AksCompute

# display the core SDK version number
print("Azure ML SDK Version: ", azureml.core.VERSION)

# Define Vars < Change the vars>. 
# In a production situation, don't put secrets in source code, but as secret variables, 
# see https://docs.microsoft.com/en-us/azure/devops/pipelines/process/variables?view=azure-devops&tabs=yaml%2Cbatch#secret-variables
workspace="<Name of your workspace>"
subscription_id="<Subscription id>"
resource_grp="<Name of your resource group where aml service is created>"

model_name = "databricksmodel.mml" # in case you want to change the name, keep the .mml extension
service_name = "databricksmodel"

aks_name = sys.argv[1]

# Get the latest evaluation result
try:
    with open("conf/image.json") as f:
        config = json.load(f)
    if not config["image_name"]:
        raise Exception('No new image created in build')
except:
    print('No image to deploy')
    #raise Exception('No new model to register as production model perform better')
    sys.exit(1)

# Start creating 
# Point file to conf directory containing details for the aml service
cli_auth = AzureCliAuthentication()
ws = Workspace(workspace_name = workspace,
               subscription_id = subscription_id,
               resource_group = resource_grp,
               auth=cli_auth)

image_name = config["image_name"]
image_version = config["image_version"]
images = Image.list(workspace=ws)
image, = (m for m in images if m.version==image_version and m.name == image_name)
print('From image.json, Model used: {}\nModel Version {}'.format(config["model_name"], config["model_version"]))
print('From image.json, Image used to deploy webservice on ACI: {}\nImage Version: {}\nImage Location = {}'.format(image.name, image.version, image.image_location))

try:
    service = AksWebservice(name=service_name, workspace=ws)
    print("delete " + service_name + " before creating new one")
    service.delete()
except:
    print(service_name + " does not exist yet")

aks_target = AksCompute(ws,aks_name)
aks_config = AksWebservice.deploy_configuration(auth_enabled=False, collect_model_data=True, enable_app_insights=True, cpu_cores = 1, memory_gb = 1)
service = Webservice.deploy_from_image(
        workspace=ws,
        name=service_name,
        image=image,
        deployment_config=aks_config,
        deployment_target=aks_target
)
service.wait_for_deployment(show_output=True)

print (service.scoring_uri)

aks_webservice = {}
aks_webservice['aks_name'] = service.name
aks_webservice['aks_url'] = service.scoring_uri
with open('conf/aks_webservice.json', 'w') as outfile:
  json.dump(aks_webservice,outfile)