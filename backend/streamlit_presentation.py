import streamlit as st
import json
import pandas as pd
import numpy as np
from recommenders import UserBasedCFRecommender, ItemBasedCFRecommender, ContentBasedRecommender, HybridRecommender
from evaluation_methods import precision_recall_f1_hit, run_evaluation_pipeline

movies = pd.read_pickle('../data/processed/movies.pkl').set_index('MovieID')
users = pd.read_pickle('../data/processed/users.pkl')["UserID"].tolist()

st.title("MovieLens Recommender System")

mode = st.radio(
    "Select Mode:",
    ['User-Recommendation Mode', 'Recommender Evaluation Mode']
)

if mode == 'User-Recommendation Mode':
    recommender_type = st.selectbox(
        "Select Recommender Type:", [
            "User-Based Collaborative Filtering",
            "Item-Based Collaborative Filtering",
            "Content-Based Filtering",
            "Hybrid Recommender"
        ]
    )
    
    user_id = st.number_input("Enter User ID:", min_value=1)
    st.markdown(
    "<p style='font-size:14px; font-style:italic;'>"
    "Note: Valid User IDs range from {} to {}."
    "</p>".format(min(users), max(users)),
    unsafe_allow_html=True
    )

    
    if recommender_type == "User-Based Collaborative Filtering":
        top_k = st.slider("Select number of recommendations (top_k):", min_value=1, max_value=20, value=10)
        alpha = st.slider("Select alpha value for optional popularity hybridization recommendation:", min_value=0.0, max_value=1.0, value=0.0)
        recommender = UserBasedCFRecommender(top_k=top_k, alpha=alpha, eval_mode=False)
        
        if st.button("Get Recommendations"):
            try:
                with st.spinner('Loading...'):
                    recommendations = recommender.recommend(user_id)
                    recs_detailed = []
                    for ind, mid in enumerate(recommendations):
                        recs_detailed.append({"Rank": ind+1, "Movie Title": movies.loc[mid]["Title"]})
                    recs_df = pd.DataFrame(recs_detailed)
                st.success('Recommendations Generated!')
                st.dataframe(recs_df, hide_index=True)
            except:
                st.error("An error occurred while generating recommendations. Please check the User ID and try again.")
    elif recommender_type == "Item-Based Collaborative Filtering":
        top_k = st.slider("Select number of recommendations (top_k):", min_value=1, max_value=20, value=10)
        adjusted = st.checkbox("Use Adjusted Cosine Similarity?", value=False)
        recommender = ItemBasedCFRecommender(adjusted=adjusted, top_k=top_k, eval_mode=False)
        
        if st.button("Get Recommendations"):
            try:
                with st.spinner('Loading...'):
                    recommendations = recommender.recommend(user_id)
                    recs_detailed = []
                    for ind, mid in enumerate(recommendations):
                        recs_detailed.append({"Rank": ind+1, "Movie Title": movies.loc[mid]["Title"]})
                    recs_df = pd.DataFrame(recs_detailed)
                st.success('Recommendations Generated!')
                st.dataframe(recs_df, hide_index=True)
            except:
                st.error("An error occurred while generating recommendations. Please check the User ID and try again.")
    elif recommender_type == "Content-Based Filtering":
        top_k = st.slider("Select number of recommendations (top_k):", min_value=1, max_value=20, value=10)
        use_tfidf = st.checkbox("Use TF-IDF for content similarity?", value=False)
        recommender = ContentBasedRecommender(use_tfidf=use_tfidf, top_k=top_k, eval_mode=False)
        
        if st.button("Get Recommendations"):
            try:
                with st.spinner('Loading...'):
                    recommendations = recommender.recommend(user_id)
                    recs_detailed = []
                    for ind, mid in enumerate(recommendations):
                        recs_detailed.append({"Rank": ind+1, "Movie Title": movies.loc[mid]["Title"]})
                    recs_df = pd.DataFrame(recs_detailed)
                st.success('Recommendations Generated!')
                st.dataframe(recs_df, hide_index=True)
            except:
                st.error("An error occurred while generating recommendations. Please check the User ID and try again.")
    elif recommender_type == "Hybrid Recommender":
        top_k = st.slider("Select number of recommendations (top_k):", min_value=1, max_value=20, value=10)
        alpha = st.slider("Select alpha value for hybrid recommendation:", min_value=0.0, max_value=1.0, value=0.8)
        candidate_factor = st.slider("Select candidate factor:", min_value=1, max_value=10, value=5)
        recommender = HybridRecommender(alpha=alpha, top_k=top_k, candidate_factor=candidate_factor, eval_mode=False)
        
        if st.button("Get Recommendations"):
            try:
                with st.spinner('Loading...'):
                    recommendations = recommender.recommend(user_id)
                    recs_detailed = []
                    for ind, mid in enumerate(recommendations):
                        recs_detailed.append({"Rank": ind+1, "Movie Title": movies.loc[mid]["Title"]})
                    recs_df = pd.DataFrame(recs_detailed)
                st.success('Recommendations Generated!')
                st.dataframe(recs_df, hide_index=True)
            except:
                st.error("An error occurred while generating recommendations. Please check the User ID and try again.")
elif mode == 'Recommender Evaluation Mode':
    MODEL_CLASSES = {
    "HybridRecommender": HybridRecommender,
    "UserBasedCFRecommender": UserBasedCFRecommender,
    "ItemBasedCFRecommender": ItemBasedCFRecommender,
    "ContentBasedRecommender": ContentBasedRecommender
    }
    
    config_check = st.checkbox("Please confirm that configurations for evaluation are set in the backend/evaluation_config.json file.", value=False)
    if config_check:
        data_preprocessing_check = st.checkbox("Please confirm that data preprocessing has been completed for evaluation purpose.", value=False)
        if data_preprocessing_check:
            if st.button("Run Evaluation"):   
                with st.spinner('Running evaluation...'):
                    
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
                    
                st.success('Evaluation Completed!')
                st.dataframe(results_df, hide_index=True)