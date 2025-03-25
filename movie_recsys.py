# Import essential libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, NMF, KNNBasic, Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split
from surprise.accuracy import rmse, mae
import joblib
import os

# ---------------------------
# 1. Dataset Management
# ---------------------------
# Check if dataset exists locally and download if missing
if not os.path.exists('ml-100k/u.item'):
    # Download and extract MovieLens 100k dataset
    !wget -q https://files.grouplens.org/datasets/movielens/ml-100k.zip
    !unzip -q ml-100k.zip

def load_data():
    """
    Loads and preprocesses MovieLens dataset.
    
    Returns:
        tuple: (movies DataFrame, ratings DataFrame)
    """
    # Load movie metadata with genre information
    movies = pd.read_csv(
        'ml-100k/u.item',
        sep='|',
        encoding='latin-1',
        names=['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
               'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
               'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
               'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    )
    
    # Load user ratings
    ratings = pd.read_csv(
        'ml-100k/u.data',
        sep='\t',
        names=['user_id', 'item_id', 'rating', 'timestamp']
    )
    
    # Remove unnecessary columns
    movies.drop(columns=['video_release_date', 'IMDb_URL', 'unknown'], inplace=True)
    ratings.drop(columns=['timestamp'], inplace=True)
    
    return movies, ratings

# Load dataset
movies, ratings = load_data()

# ---------------------------
# 2. Content-Based Filtering
# ---------------------------
def content_based_preprocessing(movies_df):
    """
    Prepares movie data for content-based recommendations.
    
    Converts genre information into TF-IDF vectors.
    
    Args:
        movies_df (DataFrame): Movies DataFrame with genre columns
    
    Returns:
        tuple: (TF-IDF matrix, TF-IDF vectorizer)
    """
    # Extract genre columns
    genres = movies_df.iloc[:, 4:-1].columns.tolist()
    
    # Convert genre flags to string representation
    movies_df['genres_str'] = movies_df[genres].apply(
        lambda x: ' '.join(x[x == 1].index), axis=1
    )
    
    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres_str'])
    
    return tfidf_matrix, tfidf

# Generate TF-IDF matrix and similarity matrix
tfidf_matrix, tfidf = content_based_preprocessing(movies)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ---------------------------
# 3. Collaborative Filtering
# ---------------------------
def train_collaborative_filtering(ratings_df):
    """
    Trains a collaborative filtering model using Surprise library.
    
    Args:
        ratings_df (DataFrame): User-item ratings
    
    Returns:
        SVD: Trained SVD model
    """
    # Configure rating scale (1-5)
    reader = Reader(rating_scale=(1, 5))
    
    # Load data into Surprise format
    data = Dataset.load_from_df(ratings_df[['user_id', 'item_id', 'rating']], reader)
    
    # Split data into training and testing sets
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    # Initialize and train SVD model
    model = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)
    model.fit(trainset)
    
    # Evaluate model performance
    predictions = model.test(testset)
    print(f"RMSE: {rmse(predictions):.4f}")
    print(f"MAE: {mae(predictions):.4f}")
    
    return model

# Train SVD model
svd_model = train_collaborative_filtering(ratings)

# ---------------------------
# 4. Hybrid Recommendations
# ---------------------------
def hybrid_recommendation(user_id, item_id, model, alpha=0.7):
    """
    Combines collaborative and content-based predictions.
    
    Args:
        user_id (int): Target user
        item_id (int): Target item
        model (SVD): Trained SVD model
        alpha (float): Weight for collaborative filtering (default=0.7)
    
    Returns:
        float: Hybrid rating prediction
    """
    # Get collaborative prediction
    svd_pred = model.predict(user_id, item_id).est
    
    # Get user's rated items
    user_ratings = ratings[ratings['user_id'] == user_id]
    
    # Handle cold start case
    if user_ratings.empty:
        return svd_pred
    
    # Calculate user's average genre preferences
    user_genres = movies.iloc[user_ratings['item_id'].values - 1, 4:-1].mean().to_dict()
    
    # Get target movie's genres
    movie_genres = movies[movies['item_id'] == item_id].iloc[0, 4:-1].to_dict()
    
    # Compute content similarity score
    content_score = sum(
        user_genres.get(genre, 0) * movie_genres.get(genre, 0) 
        for genre in user_genres
    )
    
    # Combine predictions
    return alpha * svd_pred + (1 - alpha) * content_score

# Example usage
print(f"Hybrid Rating: {hybrid_recommendation(196, 242, svd_model):.2f}")

# ---------------------------
# 5. Model Comparison
# ---------------------------
def compare_models(ratings_df):
    """
    Evaluates multiple recommendation algorithms using cross-validation.
    
    Args:
        ratings_df (DataFrame): User-item ratings
    """
    # Configure rating scale
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['user_id', 'item_id', 'rating']], reader)
    
    # Define models to compare
    models = {
        'SVD': SVD(),
        'NMF': NMF(),
        'KNN': KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
    }
    
    # Evaluate each model
    for name, model in models.items():
        print(f"\nEvaluating {name}")
        cross_validate(model, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

# Compare models
compare_models(ratings)

# ---------------------------
# 6. Model Serialization
# ---------------------------
# Save trained models
joblib.dump(svd_model, 'svd_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

# Example loading
loaded_model = joblib.load('svd_model.pkl')

