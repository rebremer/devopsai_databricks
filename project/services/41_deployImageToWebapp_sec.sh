#!/bin/bash
#variables
# $1=webapp
# $2=registry name
# $3=resource group
image=$(echo $(cat conf/image.json) | jq -r '.image_name'):$(echo $(cat conf/image.json) | jq -r '.image_version')
password=$(echo $(az acr credential show -n $2 --query passwords[0].value)  | tr -d '"')
#update webapp with container
az webapp config container set --name $1 --resource-group $3 --docker-custom-image-name ${2}.azurecr.io/$image --docker-registry-server-url https://${2}.azurecr.io --docker-registry-server-user ${2} --docker-registry-server-password $password