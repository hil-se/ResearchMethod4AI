from data_reader import load_scut
from vgg_pre import VGG_Pre
import pandas as pd
import numpy as np

# Load training data
train, protected = load_scut("../data/train.csv")
X = np.array([pixel for pixel in train['pixels']])/255.0
y = np.array(train["Rating"])
# Load the previous trained model
model = VGG_Pre("./checkpoint/attractiveness.keras")

# Get the top k=10 uncertain data points from the pool
pool, protected = load_scut("../data/pool.csv")
X_pool = np.array([pixel for pixel in pool['pixels']])/255.0
inds = model.active_query(X_pool, k=10)
inds_left = list(set(list(range(len(pool))))-set(list(inds)))

# Move these k=10 points from pool.csv to train.csv
selected = pool["Filename"][inds]
selected_ratings = pool["Rating"][inds]
new_train = {}
new_train["Filename"] = list(train["Filename"]) + list(selected)
new_train["Rating"] = list(train["Rating"]) + list(selected_ratings)
df_train = pd.DataFrame(new_train)
df_train.to_csv("../data/train.csv", index=False)
new_pool = {}
new_pool["Filename"] = pool["Filename"][inds_left]
new_pool["Rating"] = pool["Rating"][inds_left]
df_pool = pd.DataFrame(new_pool)
df_pool.to_csv("../data/pool.csv", index=False)
print("Moved \n %s \n from pool.csv to train.csv to query their oracles." % str(selected))

# Give oracle labels to the newly added data points in the train.csv file
# Then run A2_train.py to update the model

