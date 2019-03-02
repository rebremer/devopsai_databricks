import sys
import os, json
import shutil
import azureml
from azureml.core import Workspace, Run
from azureml.core.conda_dependencies import CondaDependencies 
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.image import ContainerImage, Image
from azureml.core.model import Model

# display the core SDK version number
print("Azure ML SDK Version: ", azureml.core.VERSION)

# get the parameter values
workspace = sys.argv[1]
subscription_id = sys.argv[2]
resource_grp = sys.argv[3]

model_name = "databricksmodel.mml" # in case you want to change the name, keep the .mml extension
service_name = "databricksmodel"

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
image_json["model_name"] = model.name
image_json["model_version"] = model.version
image_json["image_name"] = image.name
image_json["image_version"] = image.version
image_json["image_location"] = image.image_location
with open("../conf/image.json", "w") as outfile:
  json.dump(image_json,outfile)