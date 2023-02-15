-- Databricks notebook source
-- MAGIC %md 
-- MAGIC You may find this series of notebooks at https://github.com/PulkitXChadha/segmentation.git. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/customer-segmentation.

-- COMMAND ----------

-- MAGIC %md The purpose of this notebook is to access and prepare the data required for our segmentation work. 

-- COMMAND ----------

-- DBTITLE 1,Create Database
DROP DATABASE IF EXISTS journey CASCADE;
CREATE DATABASE journey;
