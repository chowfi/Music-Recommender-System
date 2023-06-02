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

This data consists of implicit feedback from music listening behavior, spanning several thousand users and tens of millions of songs(~45GB, ~8000 users, ~180 million interactions, ~28 million tracks). Each observation consists of a single interaction between a user and a song. 

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



