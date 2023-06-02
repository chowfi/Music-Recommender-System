# Music_Recommender_System

This is a three-person group project of a collaborative-filter based music recommender machine utilizing implicit feedback from crowdsourced ListenBrainz dataset. We used distributed computed systems (NYU High Performance Computing, Spark) to build and evaluate this recommender machine. 

## The repo consists of:

01_preprocessing.py <br>
02_train_val_split.py <br>
03_baseline_model.py <br>
04_latent_factor_model.py <br>
05_LightFM.ipynb <br>
06_Final_Report.pdf <br>

## The data consists of:

Implicit feedback from music listening behavior, spanning several thousand users and tens of millions of songs(~45GB, ~8000 users, ~180 million interactions, ~28 million tracks). Each observation consists of a single interaction between a user and a song. 

## 1. DATA PRE-PROCESSING

The steps we took to pre-process the full data include:

1. Missing value imputation - replacing missing 'recording_mbid' with the associated 'recording_msid'.
2. Omitting metadata or irrelevant columns.
3. Keeping the top 3% of tracks that have the most unique listeners.
4. Dropping users where the number of unique songs listened to is less than 10.

One choice that we made to improve the efficiency and stability of our recommendation system was to drop tracks with few unique listeners and users who interact with few unique tracks. To set the thresholds, we looked into the distribution of those quantities demonstrated in Table 1 and Table 2.

We removed approximately 97% of tracks with less than 10 unique listeners since they had limited engagement. Additionally, we filtered out around 3% of users who had interacted with fewer than 10 songs to ensure an adequate level of user activity. These filtering steps helped streamline the dataset for subsequent analysis and modeling.

### Table 1: Distribution of number of unique listeners per track

| Unique Listeners Per Track | Track Counts   |
| -------------------------- | -------------- |
| [0, 5)                     | 21,869,058     |
| [5, 10)                    | 1,137,840      |
| [10, 20)                   | 512,031        |
| [20, 30)                   | 156,828        |
| [30, 40)                   | 73,404         |
| [40, 50)                   | 41,066         |
| ≥50                        | 100,969        |

### Table 2: Distribution of number of unique tracks interacted per user

| Unique Tracks Interacted Per User | User Counts |
| --------------------------------- | ----------- |
| [0, 10)                          | 236         |
| [10, 25)                         | 111         |
| [25, 50)                         | 156         |
| [50, 100)                        | 158         |
| [100, 500)                       | 981         |
| ≥500                              | 6,550       |

## 2. TRAIN/VALIDATION PARTITIONING

To support evaluating the average performance across users, we randomly partitioned the interactions of each user into train and validation datasets. To ensure each user has at least one record in the training set and the validation set, we grouped the dataset by 'user_id'. Within the subgroups of 'user_id', each record was assigned its row number and partitioned into either the training set or validation set based on the total number of records each subgroup had. 

For example, if user_1 interacted with 5 songs, those records were assigned row numbers from 1 to 5; the records with row numbers less than 4 (5*0.8) were assigned to the training set, and the rest of the records were assigned to the validation set. This method ensured a random partition and an approximately 80:20 split between the training dataset and the validation dataset.

To address the cold-start user problem in our study and reserve the test set for final evaluation, we simulated the presence of cold-start users in the validation set. We accomplished this by selecting a subset of users who ranked in the bottom 10 percentile of total interactions and excluding all their interactions from the training data. This simulation allowed us to assess the performance of the cold-start model on the validation set while ensuring the test set remained untouched for comprehensive evaluation.

Once the users for the cold-start simulation were identified, we proceeded to divide our train and validation sets into two groups: users facing the cold-start problem and users not facing the cold-start problem. We followed a similar partitioning approach for our test set, with the distinction that we did not require any simulation for cold-start users. Instead, we identified cold-start users as those present in the test set but absent from the train set. This partitioning allowed us to evaluate the performance of the cold-start model separately for both user categories while maintaining the integrity of the test set.

Our popularity baseline model and latent factor model were developed using the users who did not encounter the cold-start problem. In contrast, the cold-start model specifically targeted and incorporated users who faced the cold-start challenge. This segregation allowed us to tailor the models accordingly and address the unique characteristics of each user group.

## 3. POPULARITY BASELINE MODELS

The formula we used for our popularity baseline model is given by:
P[i] ← (Σᵤ R[u,i]) / (|R[:,i]| + β)

where P[i] is the popularity score for a specific track (i), R is a matrix of implicit ratings between users (u) and tracks (i), and 𝛽 is a damping constant to prevent extreme recommendation scores for items with very few distinct user ratings. In addition, we conditioned our model on there existing an interaction between the user and track.

We will hyperparameter tune 𝛽.

## 4. BASELINE MODEL PERFORMANCE

To evaluate our model's performance, we opted for Mean Average Precision at K (MAP@K). MAP@K calculates the average precision at K for each user and then computes the mean across all users. This metric quantifies the fraction of relevant items present in the top K recommended items across the entire dataset. With K set to 100, we focused on evaluating the relevance of the top 100 songs recommended to each user. MAP@K is a widely recognized and reliable metric commonly used for assessing recommender systems. [4]

We hyperparameter tuned 𝛽 from our popularity baseline model, and the results of tuning the damping factor 𝛽 are as follows in Table 3:

| 𝛽      | Non-Cold-Start | Cold-Start |
|---------|----------------|------------|
| 1       | 6.74E-06       | 9.99E-06   |
| 100     | 3.31E-05       | 1.23E-05   |
| 1,000   | 2.47E-04       | 1.59E-04   |
| 5,000   | 3.467E-04      | 2.17E-04   |
| 10,000  | 3.58E-04       | 2.17E-04   |
| 20,000  | 3.67E-04       | 2.20E-04   |
| 100,000 | 3.76E-04       | 2.42E-04   |
| 1,000,000 | 3.84E-04     | 2.46E-04   |

**Table 3: MAP@100 Results of Beta hyperparameter tuning on the validation set**

|            | Non-Cold-Start | Cold-Start |
|------------|----------------|------------|
| Validation | 3.467E-04      | 2.17E-04   |
| Test       | 1.192E-04      | 4.618E-04  |

**Table 4: MAP@100 Results of Baseline Popularity Model based on 𝛽 = 5,000**

The relatively small values of MAP@100 in our recommendation system can be attributed to the challenge of suggesting 100 previously unheard songs that users may like. This task is inherently difficult as it requires accurate predictions for unfamiliar preferences. Additionally, we believe that our recommendation system might be better suited for an alternative evaluation metric such as Normalized Discounted Cumulative Gain (NDCG). Unlike MAP, NDCG places greater emphasis on the quality of the top-ranked tracks, aligning with how users typically engage with ranked lists. NDCG is less affected by the distribution of relevant items throughout the list.

From Table 4, while the differences were not too big, the baseline popularity model fared worse on the test data than on the validation data for the non-cold-start users. This observation and the relatively low scores motivate the exploration of alternative models that can enhance performance and exhibit improved generalizability.

## 5. LATENT FACTOR MODEL IMPLEMENTATION AND HYPERPARAMETER TUNING

We implemented the Latent Factor Model by leveraging Spark's alternating least squares (ALS) method. The latent factor model is used in recommendation systems to capture underlying user-item relationships by representing them in a latent feature space. We first fit the model using the training dataset. After fitting the model, we used the function `recommendForUserSubset()` to make the top 100 recommendations. The recommendations were compared with the ground truth using MAP@100.

Prior to hyperparameter tuning, we tried varying the 'maxIter' parameter. 'maxIter' refers to the maximum number of iterations that the algorithm will execute. If the algorithm reaches convergence before this number of iterations, it will stop; otherwise, it will stop after 'maxIter' iterations, regardless of whether convergence has been achieved. Based on the documentation from the Apache Spark website [2], ALS typically converges within 20 iterations. However, since 20 iterations took a long time to run on the cluster and did not provide a better MAP@100 score, we chose 15 as the value for 'maxIter' to ensure the model converges. Interestingly, we observed a slight decrease in MAP@100 as we increased 'maxIter' from 10 to 12 and from 12 to 15.

To tune the hyperparameters, we ran through multiple iterations varying the 'rank', 'regParam', and 'alpha'. 'rank' is the number of latent factors in the model, representing the underlying tastes and preferences of the users in a recommendation system. 'regParam' specifies the regularization parameter in ALS, controlling the magnitude of user and item factors to prevent overfitting. 'alpha' is the degree of confidence in implicit preference observations.

Due to limited resources, we adopted a one-parameter-at-a-time approach instead of an exhaustive grid search involving all possible combinations. We prioritized investigating the 'rank' parameter, as it determines the number of latent factors and is considered more critical. By exploring the rank value comprehensively, we aimed to identify the optimal setting. Conducting a grid search with multiple values for all three parameters might not have been feasible given the cluster's limitations.

We experimented with varying the 'rank' parameter using regParam=1 and alpha=1, with values of 10, 50, 100, 150, and 200. Increasing the rank led to higher MAP@100 scores, indicating improved performance. Based on these findings, we narrowed our focus to rank values of 100, 150, and 200 and proceeded to tune the regParam. Table 5 displays the experimented rank and regParam values along with the corresponding MAP@100 results. Among the tested combinations, the model with rank=200 and regParam=0.5 achieved the best MAP@100 score of 0.02. Thus, we set the rank parameter to 200 and the regParam parameter to 0.5. If time permitted, further increasing the rank could have been explored, but due to longer execution times at rank=200, we concluded the hyperparameter tuning for rank at this value.

**Table 5: Tuning for rank and regParam (alpha=1)**

| rank | regParam | 0.1   | 0.5   | 0.8   | 1     |
|------|----------|-------|-------|-------|-------|
| 100  | 0.0118   | 0.0135| 0.0129| 0.0119|
| 150  | 0.0143   | 0.0171| 0.0161| 0.0146|
| 200  | 0.0165   | 0.0200| 0.0184| 0.0165|

Table 6 demonstrates the values of alpha we tried and the resulting MAP scores while keeping the best rank and regParam constant. The best MAP@100 value is achieved with alpha 1.

**Table 6: Tuning for alpha (rank=200, regParam=0.5)**

| rank | regParam | alpha | MAP   |
|------|----------|-------|-------|
| 200  | 0.5      | 0.1   | 0.0054|
| 200  | 0.5      | 0.5   | 0.0166|
| 200  | 0.5      | 1     | 0.0200|
| 200  | 0.5      | 10    | 0.0172|






