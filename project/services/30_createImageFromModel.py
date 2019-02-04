import sys
import os, json
import shutil
import azureml
from azureml.core import Workspace, Run
from azureml.core.conda_dependencies import CondaDependencies 
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.image import ContainerImage, Image
from azureml.core.model import Model

# display the core SDK version number
print("Azure ML SDK Version: ", azureml.core.VERSION)

# Define Vars < Change the vars>. 
# In a production situation, don't put secrets in source code, but as secret variables, 
# see https://docs.microsoft.com/en-us/azure/devops/pipelines/process/variables?view=azure-devops&tabs=yaml%2Cbatch#secret-variables
tenant_id="<Enter Your Tenant Id>"
app_id="<Application Id of the SPN you Create>"
app_key= "<Key for the SPN>"
workspace="<Name of your workspace>"
subscription_id="<Subscription id>"
resource_grp="<Name of your resource group where aml service is created>"

model_name = "databricksmodel.mml" # in case you want to change the name, keep the .mml extension
service_name = "databricksmodel"

print("Starting to download the model file")
    
# Start creating 
# Point file to conf directory containing details for the aml service
spn = ServicePrincipalAuthentication(tenant_id,app_id,app_key)
ws = Workspace(auth = spn,
            workspace_name = workspace,
            subscription_id = subscription_id,
            resource_group = resource_grp)
model=Model(ws,model_name)

#print("Writing Conda File")
#myenv = CondaDependencies()
#myenv.add_conda_package("scikit-learn")


#with open("myenv.yml","w") as f:
#    f.write(myenv.serialize_to_string())
#print("Finished Writing Conda File")

os.chdir("deploy")

# Add model name to scorefile
with open("scoreSparkTemplate.py") as fr:
    score = fr.read()

score = score.replace("{model_name}", model_name)

with open("scoreSpark.py", "w") as fw:
    fw.write(score)

# configure the image
image_config = ContainerImage.image_configuration(execution_script="scoreSpark.py",
                                                  runtime="spark-py",
                                                  conda_file="myenv.yml")                                    

image = Image.create(name = service_name,
                     models = [model],
                     image_config = image_config,
                     workspace = ws)

image.wait_for_creation(show_output = True)

# Writing the model and image details to /aml_config/image.json

image_json = {}
image_json['model_name'] = model.name
image_json['model_version'] = model.version
image_json['image_name'] = image.name
image_json['image_version'] = image.version
image_json['image_location'] = image.image_location
with open('../conf/image.json', 'w') as outfile:
  json.dump(image_json,outfile)