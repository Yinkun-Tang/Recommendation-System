import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

with open("evaluation_config.json", "r") as f:
    config = json.load(f)

TOP_K = config["TOP_K"]
TEST_RATIO = config["TEST_RATIO"]
RANDOM_SEED = config["RANDOM_SEED"]

user_item_matrix = pd.read_pickle('../data/processed/user_item_matrix.pkl')

np.random.seed(RANDOM_SEED)
train_matrix = user_item_matrix.copy()
test_matrix = pd.DataFrame(0, index=user_item_matrix.index, columns=user_item_matrix.columns)

for user in user_item_matrix.index:
    rated_items = user_item_matrix.loc[user].loc[user_item_matrix.loc[user] > 0].index
    n_test = max(1, int(len(rated_items) * TEST_RATIO))
    test_items = np.random.choice(rated_items, size=n_test, replace=False)
    
    train_matrix.loc[user, test_items] = 0
    test_matrix.loc[user, test_items] = user_item_matrix.loc[user, test_items]

train_matrix.to_pickle('../data/processed/eval_train_matrix.pkl')
print("Evaluation Data Preprocessing: Training matrix saved successfully.")

item_user_matrix_eval = train_matrix.T.fillna(0)
item_user_matrix_eval.to_pickle('../data/processed/eval_item_user_matrix.pkl')
print("Evaluation Data Preprocessing: Item-User matrix for evaluation saved successfully.")

item_similarity_matrix_eval = cosine_similarity(item_user_matrix_eval)
item_similarity_matrix_eval_df = pd.DataFrame(item_similarity_matrix_eval, index=item_user_matrix_eval.index, columns=item_user_matrix_eval.index)
item_similarity_matrix_eval_df.to_pickle('../data/processed/eval_item_similarity_matrix.pkl')
print("Evaluation Data Preprocessing: Item similarity matrix for evaluation saved successfully.")

user_mean_eval = item_user_matrix_eval.mean(axis=1)
adjusted_item_user_matrix_eval = item_user_matrix_eval.sub(user_mean_eval, axis=0).fillna(0)
adjusted_item_similarity_matrix_eval = cosine_similarity(adjusted_item_user_matrix_eval)
adjusted_item_similarity_matrix_eval_df = pd.DataFrame(adjusted_item_similarity_matrix_eval, index=item_user_matrix_eval.index, columns=item_user_matrix_eval.index)
adjusted_item_similarity_matrix_eval_df.to_pickle('../data/processed/eval_adjusted_item_similarity_matrix.pkl')
print("Evaluation Data Preprocessing: Adjusted item similarity matrix for evaluation saved successfully.")

train_values = train_matrix.values.astype(float)

user_means = np.true_divide(train_values.sum(axis=1), (train_values != 0).sum(axis=1))
centered = train_values - user_means[:, None]
centered[train_values == 0] = 0

norms = np.linalg.norm(centered, axis=1)
norms[norms == 0] = 1
similarity = centered @ centered.T / (norms[:, None] * norms[None, :])
similarity = np.nan_to_num(similarity)

user_correlation_matrix_eval = pd.DataFrame(similarity, index=train_matrix.index, columns=train_matrix.index)
user_correlation_matrix_eval.to_pickle('../data/processed/eval_user_correlation_matrix.pkl')
print("Evaluation Data Preprocessing: User correlation matrix for evaluation saved successfully.")

popularity_eval = train_matrix.astype(bool).sum(axis=0).reset_index()
popularity_eval.columns = ['MovieID', 'NumRatings']
popularity_eval.to_pickle('../data/processed/eval_popularity.pkl')
print("Evaluation Data Preprocessing: Popularity data for evaluation saved successfully.")