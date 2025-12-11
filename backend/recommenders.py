import pandas as pd
import numpy as np
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'processed'

class RecommenderBase:
    def __init__(self, data_path=DATA_PATH, eval_mode=False):
        # Collaborative filtering matrices
        self.user_item_matrix = pd.read_pickle(data_path / 'user_item_matrix.pkl')
        self.item_user_matrix = pd.read_pickle(data_path / 'item_user_matrix.pkl')
        self.item_similarity_matrix = pd.read_pickle(data_path / 'item_similarity_matrix_df.pkl')
        self.adjusted_item_similarity_matrix = pd.read_pickle(data_path / 'adjusted_item_similarity_matrix_df.pkl')
        
        # Content-based matrices
        self.content_similarity_matrix_one_hot = pd.read_pickle(data_path / 'content_similarity_matrix_one_hot.pkl')
        self.content_similarity_matrix_tfidf = pd.read_pickle(data_path / 'content_similarity_matrix_tfidf.pkl')
        
        # Popularity data
        self.popularity = pd.read_pickle(data_path / 'popularity.pkl')
        
        # User and movie metadata
        self.users = pd.read_pickle(data_path / 'users.pkl')
        self.movies = pd.read_pickle(data_path / 'movies.pkl')
        
        # User correlation matrix for user-based CF
        self.user_correlation_matrix = pd.read_pickle(data_path / 'user_correlation_matrix.pkl')
        
        if eval_mode:
            self.user_item_matrix = pd.read_pickle(data_path / 'eval_train_matrix.pkl')
            self.item_user_matrix = pd.read_pickle(data_path / 'eval_item_user_matrix.pkl')
            self.item_similarity_matrix = pd.read_pickle(data_path / 'eval_item_similarity_matrix.pkl')
            self.adjusted_item_similarity_matrix = pd.read_pickle(data_path / 'eval_adjusted_item_similarity_matrix.pkl')
            self.user_correlation_matrix = pd.read_pickle(data_path / 'eval_user_correlation_matrix.pkl')
            self.popularity = pd.read_pickle(data_path / 'eval_popularity.pkl')
        
# User-Based Collaborative Filtering Recommender
class UserBasedCFRecommender(RecommenderBase):
    def __init__(self, top_k=10, alpha=1.0, eval_mode=False):
        super().__init__(eval_mode=eval_mode)
        self.top_k = top_k
        self.alpha = alpha
    
    def recommend(self, user_id):
        if user_id not in self.user_item_matrix.index:
            # New User (Cold Start): Recommend most popular items
            if self.popularity is not None:
                return self.popularity.sort_values(by='NumRatings', ascending=False).head(self.top_k)['MovieID'].tolist()
            else:
                return []
        
        # Top K Nearest Neighbors
        neighbors = self.user_correlation_matrix[user_id].sort_values(ascending=False).iloc[1:self.top_k+1].index
        
        # Predict scores based on neighbors' ratings
        pred_scores = self.user_item_matrix.loc[neighbors].mean(axis=0)
        already_rated = self.user_item_matrix.loc[user_id]
        pred_scores = pred_scores[already_rated.isna() | (already_rated == 0)]
        
        # Optional: Adjust scores with alpha (popularity hybridization)
        if self.popularity is not None:
            pop_scores = self.popularity.set_index('MovieID')['NumRatings']
            pop_scores = pop_scores / pop_scores.max()  # normalization
            pred_scores = pred_scores / pred_scores.max()  # normalization
            pop_scores_for_candidates = pop_scores.reindex(pred_scores.index).fillna(0)
            pred_scores = self.alpha * pred_scores + (1 - self.alpha) * pop_scores_for_candidates
        
        top_items = pred_scores.sort_values(ascending=False).head(self.top_k).index.tolist()
        return top_items

# Item-Based Collaborative Filtering Recommender
class ItemBasedCFRecommender(RecommenderBase):
    def __init__(self, adjusted=False, top_k=10, eval_mode=False):
        super().__init__(eval_mode=eval_mode)
        self.scores = None
        self.top_k = top_k
        self.sim_matrix = self.adjusted_item_similarity_matrix if adjusted else self.item_similarity_matrix
    
    def recommend(self, user_id, top_k=None):
        actual_used_top_k = top_k if top_k else self.top_k
        
        if user_id not in self.user_item_matrix.index:
            # New User (Cold Start): Recommend most popular items
            if self.popularity is not None:
                return self.popularity.sort_values(by='NumRatings', ascending=False).head(actual_used_top_k)['MovieID'].tolist()
            else:
                return []
        
        scores = {}
        
        user_vectors = self.user_item_matrix.loc[user_id].values
        sim_matrix = self.sim_matrix.values
        scores_array = user_vectors @ sim_matrix
        
        scores_array[user_vectors > 0] = 0
        
        scores_array = np.nan_to_num(scores_array)
        
        scores = pd.Series(scores_array, index=self.user_item_matrix.columns)
        
        scores = scores[scores.notna()]
        scores = scores.fillna(0)
        
        self.scores = scores.to_dict()
        
        top_items = sorted(self.scores, key=self.scores.get, reverse=True)[:actual_used_top_k]
        return top_items
    
    def get_scores(self):
        return self.scores

# Content-Based Recommender
class ContentBasedRecommender(RecommenderBase):
    def __init__(self, use_tfidf=False, top_k=10, eval_mode=False):
        super().__init__(eval_mode=eval_mode)
        self.scores = None
        self.top_k = top_k
        self.sim_matrix = self.content_similarity_matrix_tfidf if use_tfidf else self.content_similarity_matrix_one_hot 
    
    def recommend(self, user_id, top_k=None):
        actual_used_top_k = top_k if top_k else self.top_k
        
        if user_id not in self.user_item_matrix.index:
            # New User (Cold Start): Recommend most popular items
            if self.popularity is not None:
                return self.popularity.sort_values(by='NumRatings', ascending=False).head(actual_used_top_k)['MovieID'].tolist()
            else:
                return []
        
        scores = {}
            
        user_vectors = self.user_item_matrix.loc[user_id].values
        
        liked_indices = np.where(user_vectors > 0)[0]
        if len(liked_indices) == 0:
            return self.popularity.sort_values(by='NumRatings', ascending=False).head(actual_used_top_k)['MovieID'].tolist() if self.popularity is not None else []
        
        sim_matrix = self.sim_matrix.loc[self.user_item_matrix.columns, self.user_item_matrix.columns].values
        scores_array = sim_matrix[liked_indices].sum(axis=0)
        scores_array[liked_indices] = 0
        
        scores_array = np.nan_to_num(scores_array)
        
        scores = pd.Series(scores_array, index=self.user_item_matrix.columns)
        
        scores = scores[scores.notna()]
        scores = scores.fillna(0)
        
        self.scores = scores.to_dict()
        
        top_items = sorted(self.scores, key=self.scores.get, reverse=True)[:actual_used_top_k]
        return top_items
    
    def get_scores(self):
        return self.scores

# Hybrid Recommender
class HybridRecommender(RecommenderBase):
    def __init__(self, alpha=0.8, top_k=10, candidate_factor=5, eval_mode=False):
        super().__init__(eval_mode=eval_mode)
        self.alpha = alpha
        self.top_k = top_k
        self.candidate_factor = candidate_factor
        self.n_candidates = top_k * candidate_factor
        self.item_cf = ItemBasedCFRecommender(eval_mode=eval_mode)
        self.content_based = ContentBasedRecommender(eval_mode=eval_mode)
    
    def recommend(self, user_id):
        if user_id not in self.user_item_matrix.index:
            # New User (Cold Start): Recommend most popular items
            if self.popularity is not None:
                return self.popularity.sort_values(by='NumRatings', ascending=False).head(self.top_k)['MovieID'].tolist()
            else:
                return []
        # Get recommendation scores from both collaborative filtering and content-based
        self.item_cf.recommend(user_id, top_k=self.n_candidates)
        self.content_based.recommend(user_id, top_k=self.n_candidates)
        
        item_cf_scores = self.item_cf.get_scores()
        content_scores = self.content_based.get_scores()
        
        combined_scores = {}
        for item in set(item_cf_scores) | set(content_scores):
            combined_scores[item] = (self.alpha * item_cf_scores.get(item, 0) + (1 - self.alpha) * content_scores.get(item, 0))
        
        top_items = sorted(combined_scores, key=combined_scores.get, reverse=True)[:self.top_k]
        return top_items