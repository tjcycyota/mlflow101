# Databricks notebook source
# MAGIC %md
# MAGIC ## MLflow 101 on Databricks
# MAGIC > with Automatic MLflow Logging
# MAGIC 
# MAGIC ### Cluster Requirements:
# MAGIC * ML runtime (tested on `DBR 7.6 ML`) [Docs](https://docs.databricks.com/release-notes/runtime/7.6ml.html)
# MAGIC * Enable Cluster Spark config: `spark.databricks.mlflow.autologging.enabled true` [Docs](https://docs.databricks.com/clusters/configure.html#spark-configuration)
# MAGIC 
# MAGIC ### Setup: Imports and Load Data

# COMMAND ----------

from sklearn import datasets, linear_model, tree
import pandas as pd
iris = datasets.load_iris()
print("Feature Data: \n", iris.data[::50], "\nTarget Classes: \n", iris.target[::50])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model 1: LogisticRegression

# COMMAND ----------

model_1 = linear_model.LogisticRegression(max_iter=200)
model_1.fit(iris.data, iris.target)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model 2: Decision Tree

# COMMAND ----------

model_2 = tree.DecisionTreeClassifier()
model_2.fit(iris.data, iris.target)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Alternative: Enable MLflow Autologging

# COMMAND ----------

import mlflow # Import MLflow 
mlflow.autolog() # Turn on "autologging"

with mlflow.start_run(run_name="Sklearn Decision Tree"): #Pass in run_name using "with" Python syntax
  model_3 = tree.DecisionTreeClassifier(max_depth=5).fit(iris.data, iris.target) #Instantiate and fit model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make Predictions with Model
# MAGIC After registering your model to the model registry and transitioning it to Stage `Production`, load it back to make predictions

# COMMAND ----------

model_name = "mlflow_101_demo" #Or replace with your model name
model_uri = "models:/{}/production".format(model_name)

print("Loading PRODUCTION model stage with name: '{}'".format(model_uri))
model = mlflow.pyfunc.load_model(model_uri)
print("Model object of type:", type(model))

# COMMAND ----------

predictions = model.predict(pd.DataFrame(iris.data[::50]))
pd.DataFrame(predictions).head()
