#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Template script to connect to Active Spark Session
Usage:
    $ spark-submit --deploy-mode client final_project_preprocessing.py <any arguments you wish to add>
'''

import os
import sys
import getpass

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import coalesce, collect_list, struct, col, when, percent_rank
from pyspark.sql.window import Window

def main(spark, userID):
    users_small = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/users_train.parquet')
    interactions_small = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train.parquet')
    tracks_small = spark.read.parquet('hdfs:/user/bm106_nyu_edu/1004-project-2023/tracks_train.parquet')
    
    print('users_small schema')
    # users_small.printSchema()
    # interactions_small.printSchema()
    # tracks_small.printSchema()

    users_small.createOrReplaceTempView('users_small')
    interactions_small.createOrReplaceTempView('interactions_small')
    tracks_small.createOrReplaceTempView('tracks_small')
   
    # query1 = spark.sql('SELECT * FROM interactions_small LIMIT 10')
    # query1.show()

    # query2 = spark.sql('SELECT * FROM tracks_small LIMIT 10')
    # query2.show()
    # initial EDA

    ## pre-processing
    # 1. handling missing values - null mbid replace with msid (imputation)
    tracks_small = tracks_small.drop('__index_level_0__')

    print('Handling missing values... ')
    tracks_small = tracks_small.withColumn('recording_mbid', coalesce('recording_mbid', 'recording_msid'))
    
    ## 2. table join
    print('Full table joins... ')
    interactions_small = interactions_small.repartition(100)
    tracks_small = tracks_small.repartition(100)
    print('Finished repartition')
    tracks_interactions = interactions_small.join(tracks_small, interactions_small.recording_msid == tracks_small.recording_msid, 'left').drop(tracks_small.recording_msid)
    users_small = users_small.repartition(100)
    tracks_interactions = tracks_interactions.repartition(100)
    print("Finished second repartition")
    full_df = users_small.join(tracks_interactions, users_small.user_id == tracks_interactions.user_id, 'left').drop(tracks_interactions.user_id)
    print('Finished join')
    
    ## 3. count interactions
    print('Count interactions... ')
    full_df.createOrReplaceTempView('full_df')
    interactions = spark.sql('SELECT user_id, recording_mbid, COUNT(*) interactions from full_df GROUP BY user_id, recording_mbid')

    ## 4. looking into distributions of number of listeners per tracks and number of track listened per user
    print('Count number of unique listeners per track... ')
    full_df1 = spark.sql('SELECT recording_mbid, count(distinct user_id) listeners from full_df GROUP BY recording_mbid ORDER BY listeners DESC')
    #full_df1.show(10)
    
    #full_df2 = spark.sql('SELECT user_id, count(distinct recording_mbid) tracks from full_df GROUP BY user_id')

    print('Distribution of number of unique listeners per track:')
    full_df1.createOrReplaceTempView('full_df1')

    #full_df1.repartition(50)
    #distribution_inter = full_df1.select(col('*'), 
            # when(col('listeners') < 5, '[0, 5)')
            # .when(col('listeners') < 10, '[5, 10)')
            # .when(col('listeners') < 20, '[10, 20)')
            # .when(col('listeners') < 30, '[20, 30)')
            # .when(col('listeners') < 40, '[30, 40)')
            # .when(col('listeners') < 50, '[40, 50)')
            # .when(col('listeners') < 100, '[50, 100)')
            # .when(col('listeners') < 200, '[100, 200)')
            # .when(col('listeners') < 500, '[200, 500)')
            # .otherwise('>=500').alias('listeners_counts'))

    #distribution_inter.createOrReplaceTempView('distribution_inter')
    #listeners_dist = spark.sql('SELECT listeners_counts, COUNT(*) track_counts FROM distribution_inter GROUP BY listeners_counts ORDER BY track_counts')
    #print("HERE")
    #listeners_dist.show()

    #print('Count number of tracks that each user has interacted with...' )
    #full_df2.createOrReplaceTempView("full_df2")
    #full_df2 = full_df2.repartition(50)
    #distribution_inter1 = full_df2.select(col('*'), 
             #when(col('tracks') < 10, '[0, 10)')
             #.when(col('tracks') < 25, '[10, 25)')
             #.when(col('tracks') < 50, '[25, 50)')
             #.when(col('tracks') < 100, '[50, 100)')
             #.when(col('tracks') < 200, '[100, 200)')
             #.when(col('tracks') < 300, '[200, 300)')
             #.when(col('tracks') < 400, '[300, 400)')
             #.when(col('tracks') < 500, '[400, 500)')
             #.otherwise('>=500').alias('track_counts'))
    
    #print('Distribution of number of tracks listened per user:')
    #distribution_inter1.createOrReplaceTempView('distribution_inter1')
    #tracks_dist = spark.sql('SELECT track_counts, COUNT(*) user_counts FROM distribution_inter1 GROUP BY track_counts ORDER BY user_counts')
    #tracks_dist.show()


    # 5. filtering out tracks with too few listeners and users who interact with few tracks
    print('Filtering out tracks with too few listeners... ')
    df_popular_track = spark.sql("SELECT recording_mbid from full_df1 LIMIT 716735")
    # First try
    #df_popular_track = spark.sql('SELECT TOP(15) PERCENT recording_mbid from full_df1')
    
    # Second try
    #window = Window.partitionBy().orderBy(full_df1['listeners'].desc())

    #df_popular_track = full_df1.select('recording_mbid', percent_rank().over(window).alias('rank')).filter(col('rank') <= 0.15)

    # Third try
    # Calculate the 85th percentile value for listeners
    #percentile_value = full_df1.selectExpr("percentile(listeners, 0.85)").collect()[0][0]

    # Select the top 15% of records based on listeners
    #df_popular_track = full_df1.filter(full_df1.listeners >= percentile_value).select("recording_mbid")

    print("HERE")
    #print('Number of tracks with number of unique listeners count above 5: ', df_popular_track.count())
    df_popular_track.createOrReplaceTempView('df_popular_track')
    interactions.createOrReplaceTempView('interactions')
    interactions = interactions.repartition(100)
    df_popular_track = df_popular_track.repartition(100)
    interaction_filtered_tracks = interactions.join(df_popular_track, interactions.recording_mbid==df_popular_track.recording_mbid, 'inner').drop(interactions.recording_mbid)

    print("Filtering out the users who interacted with less than 5 unique songs... ")
    interaction_filtered_tracks.createOrReplaceTempView("interaction_filtered_tracks")
    df_frequent_users = spark.sql("SELECT user_id from interaction_filtered_tracks GROUP BY user_id HAVING COUNT(*) >= 10")
    #cold_start_users = spark.sql("SELECT DISTINCT user_id from interaction_filtered_tracks GROUP BY user_id HAVING COUNT(*) < 10")
    #cold_start_users = cold_start_users.repartition(100)
    #print("SAVING parquet")
    #cold_start_users.write.mode("overwrite").parquet("cold_start_users.parquet")
    #print("USER SAVED")
    #print('Number of users who interacted with more than 5 tracks: ', df_frequent_users.count())
    df_frequent_users.createOrReplaceTempView('df_frequent_users')
    full_interaction = interaction_filtered_tracks.join(df_frequent_users, interaction_filtered_tracks.user_id==df_frequent_users.user_id, 'inner').drop(interaction_filtered_tracks.user_id)
    
    #print('Filtering out users with too few interactions... ')
    #df_frequent_users = spark.sql('SELECT user_id from full_df2 WHERE tracks > 5')
    #print('Number of users who interacted with more than 5 tracks: ', df_frequent_users.count())
    #df_frequent_users.createOrReplaceTempView('df_frequent_users')
    #temp.createOrReplaceTempView('temp')
    #full_interaction = temp.join(df_frequent_users, ['user_id'], 'right').drop(temp.user_id)
    
#    full_interaction.show(10)
    
    ## 6, saving the full_interaction dataframe as a parquet file
    print('Saving the dataframe as a parquet file... ')
    # full_interaction.createOrReplaceTempView('full_interaction')
    #full_interaction.write.mode("overwrite").parquet('full_interaction_all_FINAL.parquet')
    print("SAVED")
    
if __name__ == "__main__":

    # Create the spark session object
    #spark = SparkSession.builder.getOrCreate()

    spark = SparkSession.builder \
        .config("spark.executor.memory", "32g") \
        .config("spark.executor.cores", "8") \
        .config("spark.executor.instances", "15") \
        .getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = getpass.getuser()

    # Call our main routine
    main(spark, userID)
