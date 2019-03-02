#!/bin/bash
#variables
webapp=$1
registry=$2
resource_grp=$3
# $(echo $(az acr list --resource-group $(rg-name) --query [0].name)  | tr -d '"')
image=$(echo $(cat conf/image.json) | jq -r '.image_name'):$(echo $(cat conf/image.json) | jq -r '.image_version')
password=$(echo $(az acr credential show -n ${registry} --query passwords[0].value)  | tr -d '"')
#update webapp with container
az webapp config container set --name ${webapp} --resource-group ${resource_grp} --docker-custom-image-name ${registry}.azurecr.io/$image --docker-registry-server-url https://${registry}.azurecr.io --docker-registry-server-user ${registry} --docker-registry-server-password $password