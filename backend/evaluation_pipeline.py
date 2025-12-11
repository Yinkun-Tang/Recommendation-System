import os
import sys
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from recommenders import UserBasedCFRecommender, ItemBasedCFRecommender, ContentBasedRecommender, HybridRecommender
from evaluation_methods import precision_recall_f1_hit, run_evaluation_pipeline

MODEL_CLASSES = {
    "HybridRecommender": HybridRecommender,
    "UserBasedCFRecommender": UserBasedCFRecommender,
    "ItemBasedCFRecommender": ItemBasedCFRecommender,
    "ContentBasedRecommender": ContentBasedRecommender
}

with open("evaluation_config.json", "r") as f:
    config = json.load(f)

TOP_K = config["TOP_K"]
TEST_RATIO = config["TEST_RATIO"]
RANDOM_SEED = config["RANDOM_SEED"]
model_configs = config["MODELS"]

models = {}
for model_name, params in model_configs.items():
    model_class = MODEL_CLASSES.get(model_name)
    if model_class:
        models[model_name] = model_class(**params)
    else:
        raise ValueError(f"Unknown model class: {model_name}")

user_item_matrix = pd.read_pickle('../data/processed/user_item_matrix.pkl')

results_df = run_evaluation_pipeline(
    user_item_matrix=user_item_matrix,
    models=models,
    top_k=TOP_K,
    test_ratio=TEST_RATIO,
    random_seed=RANDOM_SEED
)

print(results_df)

