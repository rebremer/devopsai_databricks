import requests
import json
import numpy as np
import gzip
import struct
import os
import urllib.request
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core import Workspace, Run
from azureml.core.webservice import Webservice

# Define Vars < Change the vars>
tenant_id="<Enter Your Tenant Id>"
app_id="<Application Id of the SPN you Create>"
app_key= "<Key for the SPN>"
workspace="<Name of your workspace>"
subscription_id="<Subscription id>"
resource_grp="<Name of your resource group where aml service is created>"

service_name = "databricksmodel"

print("Starting trigger engine")
# Start creating 
# Point file to conf directory containing details for the aml service
spn = ServicePrincipalAuthentication(tenant_id,app_id,app_key)
ws = Workspace(auth = spn,
            workspace_name = workspace,
            subscription_id = subscription_id,
            resource_group = resource_grp)
print(ws.name, ws._workspace_name, ws.resource_group, ws.location, sep = '\t')


# Get the hosted web service
service=Webservice(name = service_name, workspace =ws)

# Input for Model with all features
input_j = [{"hours_per_week":1, "income":"<=50K"}, {"hours_per_week":1, "income":"<=50K"}, {"hours_per_week":1, "income":"<=50K"}, {"hours_per_week":1, "income":"<=50K"}, {"hours_per_week":1, "income":"<=50K"}]
test_sample = json.dumps(input_j)
print(test_sample)
try:
    prediction = service.run(input_data = test_sample)
    print(prediction)
except Exception as e:
    result = str(e)
    print(result)
    print('ACI service is not working as expected')
    exit(1)

exit(0)