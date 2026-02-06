import streamlit as st
import pandas as pd
import joblib

class MinimalTrainset:
    def __init__(self, user_map, item_map, global_mean):
        self._raw2inner_id_users = user_map
        self._raw2inner_id_items = item_map
        self.global_mean = global_mean
        self.rating_scale = (1, 5) # <--- CRITICAL LINE

    def knows_user(self, rid):
        try:
            return int(rid) in self._raw2inner_id_users
        except:
            return False

    def knows_item(self, rid):
        try:
            return int(rid) in self._raw2inner_id_items
        except:
            return False

    def to_inner_uid(self, rid):
        return self._raw2inner_id_users[int(rid)]
    
    def to_inner_iid(self, rid):
        return self._raw2inner_id_items[int(rid)]

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
        user_map = {id: id for id in ratings['userId'].unique()}
        item_map = {id: id for id in movie_content['movieId'].unique()}
        global_mean = ratings['rating'].mean()
        # We create the trainset here
        svd.trainset = MinimalTrainset(user_map, item_map, global_mean)
    # Force the scale even if the class is cached 
    if hasattr(svd, 'trainset'):     # ensures svd.trainset exists before this line runs
        svd.trainset.rating_scale = (1, 5)
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
    # Ensure user_id is an int
    user_id = int(user_id)

    # check if SVD knows this user
    if user_id not in svd.trainset._raw2inner_id_users:
        return []       # if not go to content-based
    
    watched = ratings[ratings['userId'] == user_id]['movieId'].unique()
    all_movies = movie_content['movieId'].unique()

    # only consider movies known to the SVD model
    known_items = set(int(k) for k in svd.trainset._raw2inner_id_items.keys())
    
    # ensures m is in known items and not watched
    unseen = [m for m in all_movies if int(m) not in [int(w) for w in watched] and int(m) in known_items]

    predictions = [(int(movie_id), svd.predict(user_id, int(movie_id)).est) for movie_id in unseen]
    predictions.sort(key=lambda x: x[1], reverse=True)

    return [movie_id for movie_id, _ in predictions[:n]]

# Streamlit UI
st.subheader("Input")
user_id = st.number_input("Enter your User ID", min_value=1, step=1)
# let user select a movie they like and filter them to avoid KeyErrors 
valid_movie_ids = list(movie_similarity_df.keys())
filtered_content = movie_content[movie_content['movieId'].isin(valid_movie_ids)]

movie_names = filtered_content['title'].sort_values().tolist()      
liked_movie_name = st.selectbox("Select a movie you like", movie_names)

# Get the ID for selected movie from the filtered list only
liked_movie_id = filtered_content.loc[
    filtered_content['title'] == liked_movie_name, 'movieId'].iloc[0]

top_n = st.slider("Number of recommendations", min_value=5, max_value=30, value=10)

if st.button("Get Recommendations"):
    # Validate movie ID
    if liked_movie_id not in movie_content['movieId'].values:
        st.error("❌ Movie ID not found.")
    else:
        existing_users = ratings['userId'].unique()

        if user_id in existing_users: # Existing User (Hybrid Approach)
            st.subheader("🎯 Personalized Recommendations for Existing User (Hybrid)")

            svd_recs = recommend_by_svd(user_id, n=50)
            content_recs = recommend_by_content(liked_movie_id, top_n=50)  # get more candidates to allow for better hybrid scoring, we will filter down to top_n after scoring

            candidate_movies = list(set(svd_recs) | set(content_recs))

            scored_movies = []
            # Get similarities for the liked movie from the dictionary
            similar_to_liked = movie_similarity_df.get(int(liked_movie_id), {})

            # Get the list of user and movie IDs the SVD model actually knows
            known_items = set(int(k) for k in svd.trainset._raw2inner_id_items.keys())
            known_users = set(int(u) for u in svd.trainset._raw2inner_id_users.keys())

            for movie_id in candidate_movies:   # get movie and user IDs as integers for the SVD model
                m_id_int = int(movie_id)
                u_id_int = int(user_id)
                # Only predict if the SVD model knows this movie and user, otherwise skip it
                if m_id_int in known_items and u_id_int in known_users:
                    svd_score = svd.predict(u_id_int, m_id_int).est
                else:
                    # If user/movie is unknown, use the model's average rating
                    svd_score = svd.trainset.global_mean

                content_score = similar_to_liked.get(m_id_int, 0)
                # normalize SVD score to 0-1 range based on rating scale
                final_score = (0.7 * (svd_score / 5)) + (0.3 * content_score)
                scored_movies.append((m_id_int, final_score))
                    
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
