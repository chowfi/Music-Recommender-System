#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Template script to connect to Active Spark Session
Usage:
    $ spark-submit --deploy-mode client train_val_split.py <any arguments you wish to add>
'''

import os
import sys
import getpass
import numpy as np

from random import randint
#from functools import reduce

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import row_number, col, rand, udf, collect_list, from_json, sum, percent_rank
from pyspark.sql.types import StructType,StructField, IntegerType, ArrayType, StringType

def main(spark, userID):
    full_interaction = spark.read.parquet('full_interaction_all_FINAL.parquet', header=True, schema='interactions INT, recording_mbid STRING, user_id INT')
    full_interaction = full_interaction.repartition(50)
    
    ## 1. correcting the schema
    full_interaction = full_interaction[['interactions', 'user_id', 'recording_mbid']]

    ## 2. partitioning the dataset into train and val
    full_interaction = full_interaction.repartition(50)
    full_interaction.createOrReplaceTempView('full_interaction')

    print('Total number of rows: ', full_interaction.count())
    print('Split full_interaction into train_interaction and val_interaction... ')
    
    print('Finding the number of rows each user_id has... ')
    user_row_counts = spark.sql("SELECT user_id, CAST(0.8*COUNT(*) AS int) row_counts FROM full_interaction GROUP BY user_id")
    print("Assigning row number to each record... ")
    full_labeled = spark.sql("SELECT user_id, interactions, recording_mbid, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY (SELECT NULL)) rank from full_interaction")
        
    user_row_counts = user_row_counts.repartition(100)
    full_labeled = full_labeled.repartition(100)
    full_labeled_withCounts = user_row_counts.join(full_labeled, user_row_counts.user_id==full_labeled.user_id, "right").drop(user_row_counts.user_id)

    print("Splitting the dataset into train and val... ")
    full_labeled_withCounts = full_labeled_withCounts.repartition(100)
    full_labeled_withCounts.createOrReplaceTempView("full_labeled_withCounts")
    train_interaction = spark.sql("SELECT user_id, interactions, recording_mbid from full_labeled_withCounts WHERE rank<=row_counts")
    val_interaction = spark.sql("SELECT user_id, interactions, recording_mbid from full_labeled_withCounts WHERE rank>row_counts")
    print(train_interaction.take(2))
    train_interaction = train_interaction.repartition(50)
    val_interaction = val_interaction.repartition(50)
    print('Number of rows in training set: ', train_interaction.count())
    print('Number of rows in testing set: ', val_interaction.count())
    #train_interaction.write.parquet('train_interaction.parquet')
    #val_interaction.write.parquet('val_interaction.parquet')

    print ('Splitting train_interaction and val_interaction further into cold start vs non cold start for each dataset...')

    # Aggregate interactions by user
    user_interactions = full_interaction.groupBy('user_id').agg(sum('interactions').alias('total_interactions'))
    print(f'users and total interactions: {user_interactions.take(5)}')

    # Define window function
    user_window = Window.orderBy('total_interactions')

    # Calculate percent_rank for each user
    user_interactions = user_interactions.withColumn("percent_rank", percent_rank().over(user_window))
    print(f'users and percent rank of interactions: {user_interactions.take(5)}')

    # Split into cold-start and non-cold-start users
    cold_start_users = user_interactions.filter(col('percent_rank') <= 0.1)
    print(f'cold start users: {cold_start_users.take(5)}')
    non_cold_start_users = user_interactions.filter(col('percent_rank') > 0.1)

    # Join with train_interaction and val_interaction to get interactions for these users
    train_interaction_cold_start = train_interaction.join(cold_start_users, on='user_id', how='inner')
    print(f'train interaction cold start: {train_interaction_cold_start.take(5)}')
    val_interaction_cold_start = val_interaction.join(cold_start_users, on='user_id', how='inner')

    train_interaction_non_cold_start = train_interaction.join(non_cold_start_users, on='user_id', how='inner')
    val_interaction_non_cold_start = val_interaction.join(non_cold_start_users, on='user_id', how='inner')

    #train_interaction_cold_start.write.parquet('train_interaction_coldstart_ALL.parquet')
    #train_interaction_non_cold_start.write.parquet('train_interaction_noncoldstart_ALL.parquet')
    #val_interaction_cold_start.write.parquet('val_interaction_coldstart_ALL.parquet')
    #val_interaction_non_cold_start.write.parquet('val_interaction_noncoldstart_ALL.parquet')

    #train_interaction.createOrReplaceTempView('train_interaction')
    #val_interaction.createOrReplaceTempView('val_interaction')
    #full_user = spark.sql("SELECT DISTINCT user_id from full_interaction")
    #train_user = spark.sql("SELECT DISTINCT user_id user_id_train from train_interaction")
    #val_user = spark.sql("SELECT DISTINCT user_id user_id_val from val_interaction")
    #print("CHECK")
    #print(full_user.count())
    #print(train_user.count())
    #print(val_user.count())

    #train_user.repartition(100)
    #val_user.repartition(100)
    #check_missing = train_user.join(val_user, train_user.user_id_train==val_user.user_id_val, "right")
    #check_missing1 = check_missing.where(check_missing.user_id_train.isNull())
    #check_missing1.show(5)
    #print("number of user_id missing:")
    #print(check_missing1.count())

    ## 3. extra: making smaller training and validation dataset for faster results
    #train_smaller = train_interaction.limit(1000)
    #val_smaller = val_interaction.limit(250)
    #train_smaller.write.parquet('train_smaller.parquet')
    #val_smaller.write.parquet('val_smaller.parquet')

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = getpass.getuser()

    # Call our main routine
    main(spark, userID)
