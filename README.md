# mlflow101
A no-nonsense demo of MLflow on Databricks

## Instructions
1. In Databricks, you can either:
   * Download and import this file as a `.py` file
   * If using Databricks Repos, click "Create Repo" and pass in the public URL for this Github Repo
2. Create a Cluster with (at least) the following settings:
   * ML runtime (tested on `DBR 7.6 ML`) [Docs](https://docs.databricks.com/release-notes/runtime/7.6ml.html)
   * Enable Cluster Spark config: `spark.databricks.mlflow.autologging.enabled true` [Docs](https://docs.databricks.com/clusters/configure.html#spark-configuration)
3. Run the cells!
   * You will see runs for each model appear in the sidebar on the right. 
   * Click one of these to explore the MLflow UI
   * From the Artifacts of a run, click Register Model under a new name (e.g. `mlflow101_model`)
   * In Model Registry, promote this model to Production, then run the last cell in the notebook using the name of that model (e.g. `mlflow101_model`)
