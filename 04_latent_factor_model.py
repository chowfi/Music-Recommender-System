#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Template script to connect to Active Spark Session
Usage:
    $ spark-submit --deploy-mode client latent_factor_model.py <any arguments you wish to add>
'''

import os
import sys
import getpass
import hashlib
import time

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, sum, count, lit, array, desc, row_number, collect_list, udf, countDistinct, from_json
from pyspark.sql.types import IntegerType, DoubleType, ArrayType
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import RankingEvaluator
from sklearn.metrics import ndcg_score

from ast import literal_eval

def main(spark, userID):
    filtered_df_full = spark.read.parquet('hdfs:/user/fc1132_nyu_edu/full_interaction_all_FINAL.parquet', header=True, schema='interactions INT, recording_mbid STRING, user_id INT')
    filtered_df_train = spark.read.parquet('hdfs:/user/fc1132_nyu_edu/train_interaction_noncoldstart_ALL.parquet', header=True, schema='user_id INT, interactions INT, recording_mbid STRING')
    filtered_df_val = spark.read.parquet('hdfs:/user/fc1132_nyu_edu/val_interaction_noncoldstart_ALL.parquet', header=True, schema='user_id INT, interactions INT, recording_mbid STRING')
    df_test = spark.read.parquet('hdfs:/user/xm618_nyu_edu/test.parquet')
    
    #cold = spark.read.parquet('hdfs:/user/xm618_nyu_edu/cold_start_users.parquet')
    #cold_start_user_test = df_test.join(cold, df_test.user_id==cold.user_id, "inner").drop(df_test.user_id)
    #cold_start_user_test = cold_start_user_test.repartition(100)
    #print("SAVING")
    #cold_start_user_test.write.mode("overwrite").parquet("cold_start_test.parquet")
    #print("SAVED")
    
    # Deleting cold-start users from the test set
    filtered_df_train.createOrReplaceTempView('filtered_df_train')
    users_in_train = spark.sql("SELECT DISTINCT user_id from filtered_df_train")
    df_test = df_test.join(users_in_train, df_test.user_id==users_in_train.user_id, "inner").drop(users_in_train.user_id)

    #filtered_df_train.show(3)
    # 1. Hash recording.mbid to intergers
    print("Converting 'recording.mbid' to integers... '")

    indexer = StringIndexer(inputCol="recording_mbid", outputCol="recording_mbid_double", handleInvalid="keep").fit(filtered_df_full)
    full_filtered_indexed_double = indexer.transform(filtered_df_full)
    train_filtered_indexed_double = indexer.transform(filtered_df_train)
    val_filtered_indexed_double = indexer.transform(filtered_df_val)
    test_indexed_double = indexer.transform(df_test)

    full_filtered_indexed_int = full_filtered_indexed_double.withColumn("recording_mbid_int", col("recording_mbid_double").cast("int")).drop(full_filtered_indexed_double.recording_mbid_double)
    train_filtered_indexed_int = train_filtered_indexed_double.withColumn("recording_mbid_int", col("recording_mbid_double").cast("int")).drop(train_filtered_indexed_double.recording_mbid_double).repartition(200)
    val_filtered_indexed_int = val_filtered_indexed_double.withColumn("recording_mbid_int", col("recording_mbid_double").cast("int")).drop(val_filtered_indexed_double.recording_mbid_double)
    test_indexed_int = test_indexed_double.withColumn("recording_mbid_int", col("recording_mbid_double").cast("int")).drop(test_indexed_double.recording_mbid_double)

    # 2. Hyperparameter tuning for ALS
    print("Fitting ALS... ")
    print("\n")
    print("Starting timer")
    
    start_time = time.time()
    als = ALS(userCol="user_id", itemCol="recording_mbid_int", ratingCol="interactions",
              maxIter=15, numUserBlocks=100, numItemBlocks=100, implicitPrefs=True, 
              rank=200, regParam=0.5, alpha=1)
    alsModel = als.fit(train_filtered_indexed_int)
    
    end_time = time.time()
    print("Total ALS model training time: {} seconds".format(end_time - start_time))


    # Making the top_k recommendations for users in val
    filtered_df_val.createOrReplaceTempView('filtered_df_val')
    users_in_val = spark.sql("SELECT DISTINCT user_id from filtered_df_val")
    top_k = 100
    print("Making top_k recommendations... ")
    predictions = alsModel.recommendForUserSubset(users_in_val, 100)
    predictions.createOrReplaceTempView("predictions")
    predictions = predictions.withColumn("song_recs",col("recommendations.recording_mbid_int")).drop("recommendation.recording_mbid_int")
    predictions.createOrReplaceTempView("predictions")

    # Transforming recommendtaion format for evaluation
    print("Transformning recommendation format for evaluation... ")
    df_predicted_labels = predictions.withColumn("predicted_labels", col("song_recs").cast(ArrayType(DoubleType()))).drop(predictions.song_recs)

    # Transforming true label format for evaluation
    window_spec = Window.partitionBy("user_id").orderBy(desc("interactions"))
    df_val_label = val_filtered_indexed_double.withColumn("rank", row_number().over(window_spec))
    top_k_true_labels = df_val_label.filter(col("rank") <= top_k).drop("rank")
    df_true_labels = top_k_true_labels.groupBy("user_id").agg(collect_list("recording_mbid_double").alias("true_labels"))
    
    joined_labels = df_true_labels.join(df_predicted_labels, on="user_id")

    # Create a RankingEvaluator for evaluating the MAP metric
    evaluator_map = RankingEvaluator(k=top_k, metricName="meanAveragePrecisionAtK", labelCol="true_labels",
                             predictionCol="predicted_labels")
      
    # Compute the MAP
    map_score = evaluator_map.evaluate(joined_labels)

    # Print the MAP/NDCG score
    print("Mean Average Precision(val):", map_score)

    # Making the top_k recommendations for users in test
    print("For test")
    df_test.createOrReplaceTempView('df_test')
    #users_in_test = spark.sql("SELECT DISTINCT user_id from df_test")
    top_k = 100
    print("Making top_k recommendations... ")
    predictions_test = alsModel.recommendForAllUsers(top_k)
    predictions_test.createOrReplaceTempView("predictions_test")
    predictions_test = predictions_test.withColumn("song_recs",col("recommendations.recording_mbid_int")).drop("recommendation.recording_mbid_int")
    predictions_test.createOrReplaceTempView("predictions_test")
    
    print("Transformning recommendation format for evaluation... ")
    df_predicted_labels_test = predictions_test.withColumn("predicted_labels", col("song_recs").cast(ArrayType(DoubleType()))).drop(predictions_test.song_recs)

    # Transforming true label format for evaluation
    window_spec = Window.partitionBy("user_id").orderBy(desc("interactions"))
    df_test_label = test_indexed_double.withColumn("rank", row_number().over(window_spec))
    top_k_true_labels_test = df_test_label.filter(col("rank") <= top_k).drop("rank")
    df_true_labels_test = top_k_true_labels_test.groupBy("user_id").agg(collect_list("recording_mbid_double").alias("true_labels"))

    df_true_labels_test = df_true_labels_test.repartition(100)
    df_predicted_labels_test = df_predicted_labels_test.repartition(100)

    joined_labels_test = df_true_labels_test.join(df_predicted_labels_test, df_true_labels_test.user_id==df_predicted_labels_test.user_id, "inner").drop(df_predicted_labels_test.user_id)

    # Create a RankingEvaluator for evaluating the MAP metric
    evaluator_map = RankingEvaluator(k=top_k, metricName="meanAveragePrecisionAtK", labelCol="true_labels",
                             predictionCol="predicted_labels")

    # Compute the MAP
    map_score_test = evaluator_map.evaluate(joined_labels_test)
    print("Mean Average Precision (test):", map_score_test)

if __name__ == "__main__":

    # Create the spark session object
    #spark = SparkSession.builder.getOrCreate()
    spark = SparkSession.builder \
        .config("spark.executor.memory", "49g") \
        .config("spark.executor.cores", "15") \
        .config("spark.executor.instances", "15") \
        .getOrCreate()
    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = getpass.getuser()

    # Call our main routine
    main(spark, userID)
