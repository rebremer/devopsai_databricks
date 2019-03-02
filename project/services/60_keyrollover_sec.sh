#!/bin/bash
#variables
current_dbr_key=$1
current_dbr_token=$2
domain=$3
vault=$4
number_of_days=$5
number_of_seconds=$((${number_of_days}*60*60*24))
#create new databricks access token
create_token=$(curl -X POST -H "Content-Type: application/json" -H "Authorization: Bearer "${current_dbr_key}"" -d "{\"lifetime_seconds\": "${number_of_seconds}", \"comment\": \"keyrollover\"}" https://${domain}/api/2.0/token/create)
new_dbr_key=$(echo $create_token | jq -r '.token_value')
new_dbr_token=$(echo $create_token | jq -r '.token_info.token_id')
if [ -z "$new_dbr_key" ]
then
   echo "no new dbr token created"
   exit 0
fi
#add new token to key vault
add_key_akv=$(echo $(az keyvault secret set -n dbr-key --vault-name ${vault} --value ${new_dbr_key}) | jq -r '.attributes.created')
if [ -z "$add_key_akv" ]
then
   echo "key not added to key vault"
   exit 0
fi
add_token_akv=$(echo $(az keyvault secret set -n dbr-token --vault-name ${vault} --value ${new_dbr_token}) | jq -r '.attributes.created')
if [ -z "$add_token_akv" ]
then
   echo "token not added to key vault, add current key back"
   add_key_akv=$(echo $(az keyvault secret set -n dbr-key --vault-name ${vault} --value ${current_dbr_key}) | jq -r '.attributes.created')
   exit 0
fi
#invalidate old databricks token
revoke=$(curl -X POST -H "Content-Type: application/json" -H "Authorization: Bearer "${new_dbr_key}"" -d "{\"token_id\": \""${current_dbr_token}"\"}" https://${domain}/api/2.0/token/delete)
if [ "$revoke" != "{}" ]
then
   echo $revoke
   echo "key not revoked in Databricks"
   exit 0
fi