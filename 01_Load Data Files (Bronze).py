# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/PulkitXChadha/segmentation.git. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/customer-segmentation.

# COMMAND ----------

# MAGIC %md The purpose of this notebook is to access and prepare the data required for our segmentation work. 

# COMMAND ----------

# MAGIC %md ## Step 1: Access the Data
# MAGIC 
# MAGIC The purpose of this exercise is to demonstrate how a Promotions Management team interested in segmenting customer households based on promotion responsiveness might perform the analytics portion of their work.  The dataset we will use has been made available by Dunnhumby via Kaggle and is referred to as [*The Complete Journey*](https://www.kaggle.com/frtgnn/dunnhumby-the-complete-journey). It consists of numerous files identifying household purchasing activity in combination with various promotional campaigns for about 2,500 households over a nearly 2 year period. The schema of the overall dataset may be represented as follows:
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/segmentation_journey_schema3.png' width=500>
# MAGIC 
# MAGIC To make this data available for our analysis, you can download, extract and load to the permanent location of the *bronze* folder of a [cloud-storage mount point](https://docs.databricks.com/data/databricks-file-system.html#mount-object-storage-to-dbfs) named */mnt/completejourney*.  We have automated this downloading step for you and use a */tmp/completejourney* storage path throughout this accelerator.  

# COMMAND ----------

# MAGIC %md From there, we might prepare the data as follows:

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from pyspark.sql.types import *
from pyspark.sql.functions import min, max

# COMMAND ----------

# DBTITLE 1,Use Database
# MAGIC %sql
# MAGIC USE journey;

# COMMAND ----------

# DBTITLE 1,Transactions
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS transactions')

# expected structure of the file
transactions_schema = StructType([
  StructField('household_id', IntegerType()),
  StructField('basket_id', LongType()),
  StructField('day', IntegerType()),
  StructField('product_id', IntegerType()),
  StructField('quantity', IntegerType()),
  StructField('sales_amount', FloatType()),
  StructField('store_id', IntegerType()),
  StructField('discount_amount', FloatType()),
  StructField('transaction_time', IntegerType()),
  StructField('week_no', IntegerType()),
  StructField('coupon_discount', FloatType()),
  StructField('coupon_discount_match', FloatType())
  ])

# read data to dataframe
( spark
    .read
    .csv(
      '/tmp/completejourney/bronze/transaction_data.csv',
      header=True,
      schema=transactions_schema
      )
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .save('/tmp/completejourney/completejourney/silver/transactions')
  )

# create table object to make delta lake queriable
_ = spark.sql('''
    CREATE TABLE transactions 
    USING DELTA 
    LOCATION '/tmp/completejourney/completejourney/silver/transactions'
    ''')

# show data
display(
  spark.table('transactions')
  )

# COMMAND ----------

# DBTITLE 1,Products
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS products')

# expected structure of the file
products_schema = StructType([
  StructField('product_id', IntegerType()),
  StructField('manufacturer', StringType()),
  StructField('department', StringType()),
  StructField('brand', StringType()),
  StructField('commodity', StringType()),
  StructField('subcommodity', StringType()),
  StructField('size', StringType())
  ])

# read data to dataframe
( spark
    .read
    .csv(
      '/tmp/completejourney/bronze/product.csv',
      header=True,
      schema=products_schema
      )
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .save('/tmp/completejourney/silver/products')
  )

# create table object to make delta lake queriable
_ = spark.sql('''
    CREATE TABLE products
    USING DELTA 
    LOCATION '/tmp/completejourney/silver/products'
    ''')

# show data
display(
  spark.table('products')
  )

# COMMAND ----------

# DBTITLE 1,Households
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS households')

# expected structure of the file
households_schema = StructType([
  StructField('age_bracket', StringType()),
  StructField('marital_status', StringType()),
  StructField('income_bracket', StringType()),
  StructField('homeownership', StringType()),
  StructField('composition', StringType()),
  StructField('size_category', StringType()),
  StructField('child_category', StringType()),
  StructField('household_id', IntegerType())
  ])

# read data to dataframe
households = (
  spark
    .read
    .csv(
      '/tmp/completejourney/bronze/hh_demographic.csv',
      header=True,
      schema=households_schema
      )
  )

# make queriable for later work
households.createOrReplaceTempView('households')

# income bracket sort order
income_bracket_lookup = (
  spark.createDataFrame(
    [(0,'Under 15K'),
     (15,'15-24K'),
     (25,'25-34K'),
     (35,'35-49K'),
     (50,'50-74K'),
     (75,'75-99K'),
     (100,'100-124K'),
     (125,'125-149K'),
     (150,'150-174K'),
     (175,'175-199K'),
     (200,'200-249K'),
     (250,'250K+') ],
    schema=StructType([
            StructField('income_bracket_numeric',IntegerType()),
            StructField('income_bracket', StringType())
            ])
    )
  )

# make queriable for later work
income_bracket_lookup.createOrReplaceTempView('income_bracket_lookup')

# household composition sort order
composition_lookup = (
  spark.createDataFrame(
    [ (0,'Single Female'),
      (1,'Single Male'),
      (2,'1 Adult Kids'),
      (3,'2 Adults Kids'),
      (4,'2 Adults No Kids'),
      (5,'Unknown') ],
    schema=StructType([
            StructField('sort_order',IntegerType()),
            StructField('composition', StringType())
            ])
    )
  )

# make queriable for later work
composition_lookup.createOrReplaceTempView('composition_lookup')

# persist data with sort order data and a priori segments
(
  spark
    .sql('''
      SELECT
        a.household_id,
        a.age_bracket,
        a.marital_status,
        a.income_bracket,
        COALESCE(b.income_bracket_numeric, -1) as income_bracket_alt,
        a.homeownership,
        a.composition,
        COALESCE(c.sort_order, -1) as composition_sort_order,
        a.size_category,
        a.child_category
      FROM households a
      LEFT OUTER JOIN income_bracket_lookup b
        ON a.income_bracket=b.income_bracket
      LEFT OUTER JOIN composition_lookup c
        ON a.composition=c.composition
      ''')
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .save('/tmp/completejourney/silver/households')
  )

# create table object to make delta lake queriable
_ = spark.sql('''
    CREATE TABLE households 
    USING DELTA 
    LOCATION '/tmp/completejourney/silver/households'
    ''')

# show data
display(
  spark.table('households')
  )

# COMMAND ----------

# DBTITLE 1,Coupons
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS coupons')

# expected structure of the file
coupons_schema = StructType([
  StructField('coupon_upc', StringType()),
  StructField('product_id', IntegerType()),
  StructField('campaign_id', IntegerType())
  ])

# read data to dataframe
( spark
    .read
    .csv(
      '/tmp/completejourney/bronze/coupon.csv',
      header=True,
      schema=coupons_schema
      )
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .save('/tmp/completejourney/silver/coupons')
  )

# create table object to make delta lake queriable
_ = spark.sql('''
    CREATE TABLE coupons
    USING DELTA 
    LOCATION '/tmp/completejourney/silver/coupons'
    ''')

# show data
display(
  spark.table('coupons')
  )

# COMMAND ----------

# DBTITLE 1,Campaigns
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS campaigns')

# expected structure of the file
campaigns_schema = StructType([
  StructField('description', StringType()),
  StructField('campaign_id', IntegerType()),
  StructField('start_day', IntegerType()),
  StructField('end_day', IntegerType())
  ])

# read data to dataframe
( spark
    .read
    .csv(
      '/tmp/completejourney/bronze/campaign_desc.csv',
      header=True,
      schema=campaigns_schema
      )
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .save('/tmp/completejourney/silver/campaigns')
  )

# create table object to make delta lake queriable
_ = spark.sql('''
    CREATE TABLE campaigns
    USING DELTA 
    LOCATION '/tmp/completejourney/silver/campaigns'
    ''')

# show data
display(
  spark.table('campaigns')
  )

# COMMAND ----------

# DBTITLE 1,Coupon Redemptions
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS coupon_redemptions')

# expected structure of the file
coupon_redemptions_schema = StructType([
  StructField('household_id', IntegerType()),
  StructField('day', IntegerType()),
  StructField('coupon_upc', StringType()),
  StructField('campaign_id', IntegerType())
  ])

# read data to dataframe
( spark
    .read
    .csv(
      '/tmp/completejourney/bronze/coupon_redempt.csv',
      header=True,
      schema=coupon_redemptions_schema
      )
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .save('/tmp/completejourney/silver/coupon_redemptions')
  )

# create table object to make delta lake queriable
_ = spark.sql('''
    CREATE TABLE coupon_redemptions
    USING DELTA 
    LOCATION '/tmp/completejourney/silver/coupon_redemptions'
    ''')

# show data
display(
  spark.table('coupon_redemptions')
  )

# COMMAND ----------

# DBTITLE 1,Campaign-Household Relationships
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS campaigns_households')

# expected structure of the file
campaigns_households_schema = StructType([
  StructField('description', StringType()),
  StructField('household_id', IntegerType()),
  StructField('campaign_id', IntegerType())
  ])

# read data to dataframe
( spark
    .read
    .csv(
      '/tmp/completejourney/bronze/campaign_table.csv',
      header=True,
      schema=campaigns_households_schema
      )
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .save('/tmp/completejourney/silver/campaigns_households')
  )

# create table object to make delta lake queriable
_ = spark.sql('''
    CREATE TABLE campaigns_households
    USING DELTA 
    LOCATION '/tmp/completejourney/silver/campaigns_households'
    ''')

# show data
display(
  spark.table('campaigns_households')
  )

# COMMAND ----------

# DBTITLE 1,Causal Data
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS causal_data')

# expected structure of the file
causal_data_schema = StructType([
  StructField('product_id', IntegerType()),
  StructField('store_id', IntegerType()),
  StructField('week_no', IntegerType()),
  StructField('display', StringType()),
  StructField('mailer', StringType())
  ])

# read data to dataframe
( spark
    .read
    .csv(
      '/tmp/completejourney/bronze/causal_data.csv',
      header=True,
      schema=causal_data_schema
      )
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .save('/tmp/completejourney/silver/causal_data')
  )

# create table object to make delta lake queriable
_ = spark.sql('''
    CREATE TABLE causal_data
    USING DELTA 
    LOCATION '/tmp/completejourney/silver/causal_data'
    ''')

# show data
display(
  spark.table('causal_data')
  )

# COMMAND ----------

# MAGIC %md ## Step 2: Adjust Transactional Data
# MAGIC 
# MAGIC With the raw data loaded, we need to make some adjustments to the transactional data.  While this dataset is focused on retailer-managed campaigns, the inclusion of coupon discount matching information would indicate the transaction data reflects discounts originating from both retailer- and manufacturer-generated coupons.  Without the ability to link a specific product-transaction to a specific coupon (when a redemption takes place), we will assume that any *coupon_discount* value associated with a non-zero *coupon_discount_match* value originates from a manufacturer's coupon.  All other coupon discounts will be assumed to be from retailer-generated coupons.  
# MAGIC 
# MAGIC In addition to the separation of retailer and manufacturer coupon discounts, we will calculate a list amount for a product as the sales amount minus all discounts applied:

# COMMAND ----------

# DBTITLE 1,Adjusted Transactions
# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS transactions_adj;
# MAGIC 
# MAGIC CREATE TABLE transactions_adj
# MAGIC USING DELTA
# MAGIC AS
# MAGIC   SELECT
# MAGIC     household_id,
# MAGIC     basket_id,
# MAGIC     week_no,
# MAGIC     day,
# MAGIC     transaction_time,
# MAGIC     store_id,
# MAGIC     product_id,
# MAGIC     amount_list,
# MAGIC     campaign_coupon_discount,
# MAGIC     manuf_coupon_discount,
# MAGIC     manuf_coupon_match_discount,
# MAGIC     total_coupon_discount,
# MAGIC     instore_discount,
# MAGIC     amount_paid,
# MAGIC     units
# MAGIC   FROM (
# MAGIC     SELECT 
# MAGIC       household_id,
# MAGIC       basket_id,
# MAGIC       week_no,
# MAGIC       day,
# MAGIC       transaction_time,
# MAGIC       store_id,
# MAGIC       product_id,
# MAGIC       COALESCE(sales_amount - discount_amount - coupon_discount - coupon_discount_match,0.0) as amount_list,
# MAGIC       CASE 
# MAGIC         WHEN COALESCE(coupon_discount_match,0.0) = 0.0 THEN -1 * COALESCE(coupon_discount,0.0) 
# MAGIC         ELSE 0.0 
# MAGIC         END as campaign_coupon_discount,
# MAGIC       CASE 
# MAGIC         WHEN COALESCE(coupon_discount_match,0.0) != 0.0 THEN -1 * COALESCE(coupon_discount,0.0) 
# MAGIC         ELSE 0.0 
# MAGIC         END as manuf_coupon_discount,
# MAGIC       -1 * COALESCE(coupon_discount_match,0.0) as manuf_coupon_match_discount,
# MAGIC       -1 * COALESCE(coupon_discount - coupon_discount_match,0.0) as total_coupon_discount,
# MAGIC       COALESCE(-1 * discount_amount,0.0) as instore_discount,
# MAGIC       COALESCE(sales_amount,0.0) as amount_paid,
# MAGIC       quantity as units
# MAGIC     FROM transactions
# MAGIC     );
# MAGIC     
# MAGIC SELECT * FROM transactions_adj;
