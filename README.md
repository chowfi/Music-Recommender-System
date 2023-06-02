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

𝑃[𝑖] ← 𝑢 𝑅[𝑢, 𝑖]
            |𝑅[:, 𝑖]| + 𝛽

where P[i] is the popularity score for a specific track (i), R is a matrix of implicit ratings between users (u) and tracks (i), and 𝛽 is a damping constant to prevent extreme recommendation scores for items with very few distinct user ratings. In addition, we conditioned our model on there existing an interaction between the user and track.

We will hyperparameter tune 𝛽.




