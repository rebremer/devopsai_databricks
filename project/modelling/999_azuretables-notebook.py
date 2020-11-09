# Databricks notebook source
# See https://docs.microsoft.com/en-us/azure/cosmos-db/table-storage-how-to-use-python
# Make sure that azure-cosmosdb-table is installed on the cluster. This can be done by adding azure-cosmosdb-table  as PyPi library on the cluster

AccountName="<<storage account>>"
AccountKey="<<access key>>"
table_name="<<your table name>>"

# COMMAND ----------

from azure.cosmosdb.table.tableservice import TableService
from azure.cosmosdb.table.models import Entity

table_service = TableService(account_name=AccountName, account_key=AccountKey)
table_service.create_table(table_name)

# COMMAND ----------

entity_example = {
   'PartitionKey': 'Car',
   'RowKey': 'Tesla',
   'text': 'Musk',
   'color': 'Purple',
   'price': '5'
}
table_service.insert_entity(table_name, entity_example)
entity_example = {
   'PartitionKey': 'Car',
   'RowKey': 'Audi',
   'text': 'Germany',
   'color': 'Green',
   'price': '15'
}
table_service.insert_entity(table_name, entity_example)
entity_example = {
   'PartitionKey': 'Bike',
   'RowKey': 'Audi',
   'text': 'Germany',
   'color': 'Green',
   'price': '5'
}
table_service.insert_entity(table_name, entity_example)

# COMMAND ----------

task = table_service.get_entity(table_name, 'Car', 'Tesla')
print(task)

# COMMAND ----------

tasks = table_service.query_entities(table_name, filter="price eq '5'")
for task in tasks:
    print(task.PartitionKey)
    print(task.RowKey)
    print(task.text)
    print("---")

# COMMAND ----------


