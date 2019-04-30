import azureml
import os, json, sys
from azureml.core import Workspace, Run
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.image import ContainerImage, Image

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
print('From image.json, Image used to deploy webservice on ACI: {}\nImage Version: {}\nImage Location = {}'.format(image.name, image.version, image.image_location))

#delete old service first before new one
try:
    oldservice = Webservice(workspace=ws, name=service_name)
    print("delete " + service_name + " before creating new one")
    oldservice.delete()
except:
    print(service_name + " does not exist, create new one")

aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                               memory_gb=1, 
                                               tags={"data": "Income",  "method" : "SparkML"}, 
                                               description='Predict Income with sparkML')  

service = Webservice.deploy_from_image(deployment_config=aciconfig,
                                        image=image,
                                        name=service_name,
                                        workspace=ws)

service.wait_for_deployment(show_output=True)

print(service.scoring_uri)

aci_webservice = {}
aci_webservice['aci_name'] = service.name
aci_webservice['aci_url'] = service.scoring_uri
with open('conf/aci_webservice.json', 'w') as outfile:
  json.dump(aci_webservice,outfile)