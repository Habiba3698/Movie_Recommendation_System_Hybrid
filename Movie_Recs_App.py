import streamlit as st
import pandas as pd
import joblib
class MinimalTrainset:
    def __init__(self, user_map, item_map, global_mean):
        self._raw2inner_id_users = user_map
        self._raw2inner_id_items = item_map
        self.global_mean = global_mean
        # Tells SVD the min and max rating possible
        self.rating_scale = (1, 5) 

    def knows_user(self, rid):
        try:
            return int(rid) in self._raw2inner_id_users
        except (ValueError, TypeError):
            return False

    def knows_item(self, rid):
        try:
            return int(rid) in self._raw2inner_id_items
        except (ValueError, TypeError):
            return False

    def to_inner_uid(self, rid):
        try:
            return self._raw2inner_id_users[int(rid)]
        except (KeyError, ValueError):
            raise ValueError(f'User {rid} unknown')
    
    def to_inner_iid(self, rid):
        try:
            return self._raw2inner_id_items[int(rid)]
        except (KeyError, ValueError):
            raise ValueError(f'Item {rid} unknown')

# load our resources (dataframes and models) once and cache them      
st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")
st.title("🎥 Hybrid Movie Recommendation System")

@st.cache_resource  # so it doesn't reload on every interaction
def load_resources():
    # Load movie content and similarity
    movie_content = pd.read_parquet("data/clean/movie_content_clean.parquet")
    movie_similarity_df = joblib.load("Models/movie_similarity_df.pkl")
    # Load SVD model
    svd = joblib.load("Models/svd_model.pkl")
    # Load ratings data
    ratings = pd.read_parquet("data/clean/ratings_clean.parquet")
    if not hasattr(svd, 'trainset') or svd.trainset is None:
        # We recreate the maps from your existing dataframes
        user_map = {id: id for id in ratings['userId'].unique()}
        item_map = {id: id for id in movie_content['movieId'].unique()}
        global_mean = ratings['rating'].mean()
        # Attach the Minimal version to the model
        svd.trainset = MinimalTrainset(user_map, item_map, global_mean)

    return movie_content, movie_similarity_df, svd, ratings

movie_content, movie_similarity_df, svd, ratings = load_resources()

# converts movie IDs to titles and genres for display
def movie_ids_to_titles(movie_ids):
    return (
        movie_content[movie_content['movieId'].isin(movie_ids)]
        [['movieId', 'title', 'genres']]
        .drop_duplicates()
        .reset_index(drop=True)
    )

# function for content based recommendations using cosine similarity
def recommend_by_content(movie_id, top_n=10):
    # Dictionary lookup instead of .index
    movie_scores = movie_similarity_df.get(movie_id, {})
    if not movie_scores:
        return []
    
    # Sort the dictionary values and take top_n
    top_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [m_id for m_id, score in top_movies]

# function for collaborative filtering recommendations using SVD
def recommend_by_svd(user_id, n=10):
    watched = ratings[ratings['userId'] == user_id]['movieId'].unique()
    all_movies = movie_content['movieId'].unique()
    unseen = [m for m in all_movies if m not in watched]

    predictions = [(movie_id, svd.predict(user_id, movie_id).est) for movie_id in unseen]
    predictions.sort(key=lambda x: x[1], reverse=True)

    return [movie_id for movie_id, _ in predictions[:n]]


st.subheader("Input")
user_id = st.number_input("Enter your User ID", min_value=1, step=1)
# let user select a movie they like
movie_names = movie_content['title'].sort_values().tolist()      
liked_movie_name = st.selectbox("Select a movie you like", movie_names)
# get the movie ID for the selected movie to use in recommendations
liked_movie_id = movie_content.loc[
    movie_content['title'] == liked_movie_name, 'movieId'].iloc[0]

top_n = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)

if st.button("Get Recommendations"):
    # Validate movie ID
    if liked_movie_id not in movie_content['movieId'].values:
        st.error("❌ Movie ID not found.")
    else:
        existing_users = ratings['userId'].unique()

        if user_id in existing_users: # Existing User (Hybrid Approach)
            st.subheader("🎯 Personalized Recommendations for Existing User (Hybrid)")

            svd_recs = recommend_by_svd(user_id, n=20)
            content_recs = recommend_by_content(liked_movie_id, top_n=20)

            candidate_movies = list(set(svd_recs) | set(content_recs))

            scored_movies = []
            # Get similarities for the liked movie from the dictionary
            similar_to_liked = movie_similarity_df.get(liked_movie_id, {})

            for movie_id in candidate_movies:
                svd_score = svd.predict(user_id, movie_id).est
                
                content_score = similar_to_liked.get(movie_id, 0)
                # Normalizing SVD to a 0-1 scale so it matches content_score
                final_score = (0.7 * (svd_score / 5)) + (0.3 * content_score)
                scored_movies.append((movie_id, final_score))

            scored_movies.sort(key=lambda x: x[1], reverse=True)
            recommended_ids = [m for m, _ in scored_movies[:top_n]]

        else: # New User (Content-Based Only)
            st.subheader("🆕 New User Recommendations (Content-Based Only)")
            recommended_ids = recommend_by_content(liked_movie_id, top_n=top_n)

        # Display recommendations
        recommendations_df = movie_ids_to_titles(recommended_ids)
        
        if recommendations_df.empty:
            st.warning("⚠️ The selected movie is not available in the model. Try a different movie.")
        else: 
            st.dataframe(recommendations_df, use_container_width=True)
