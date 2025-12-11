def precision_recall_f1_hit(recommended, actual, k=10):
    recommended_k = recommended[:k]
    actual = set(actual[actual > 0].index)
    if not actual:
        return 0, 0, 0, 0
    
    hits = len(set(recommended_k) & actual)
    precision = hits / k
    recall = hits / len(actual)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    hit_rate = 1 if hits > 0 else 0
    return precision, recall, f1, hit_rate

def run_evaluation_pipeline(user_item_matrix, models, top_k=10, test_ratio=0.2, random_seed=42):
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from collections import defaultdict
    
    np.random.seed(random_seed)
    
    train_matrix = user_item_matrix.copy()
    test_matrix = pd.DataFrame(0, index=user_item_matrix.index, columns=user_item_matrix.columns)
    
    for user in user_item_matrix.index:
        rated_items = user_item_matrix.loc[user][user_item_matrix.loc[user] > 0].index.tolist()
        n_test_items = max(1, int(len(rated_items) * test_ratio))
        test_items = np.random.choice(rated_items, size=n_test_items, replace=False)
        
        train_matrix.loc[user, test_items] = 0
        test_matrix.loc[user, test_items] = user_item_matrix.loc[user, test_items]
        
    results = defaultdict(list)
        
    for model_name, model in tqdm(models.items(), desc="Models", position=0):
        precisions = []
        recalls = []
        f1s = []
        hit_rates = []
        
        recommended_items_all = set()
        
        for user in tqdm(train_matrix.index, desc=f"{model_name}", position=1, leave=False):
            recommended_items = model.recommend(user)
            actual_items = test_matrix.loc[user]
            
            precision, recall, f1, hit_rate = precision_recall_f1_hit(recommended_items, actual_items, k=top_k)
            
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            hit_rates.append(hit_rate)
            
            recommended_items_all.update(recommended_items)
        
        user_coverage = np.mean(hit_rates)
        catalog_coverage = len(recommended_items_all) / user_item_matrix.shape[1]
        
        results["Model"].append(model_name)
        results["Precision"].append(np.mean(precisions))
        results["Recall"].append(np.mean(recalls))
        results["F1-Score"].append(np.mean(f1s))
        results["User Coverage"].append(user_coverage)
        results["Catalog Coverage"].append(catalog_coverage)
    
    return pd.DataFrame(results)