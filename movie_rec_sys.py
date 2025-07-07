# =================================================================
# MODULE 1: SETUP & CONFIGURATION
# =================================================================

# Import necessary libraries
import numpy as np                # For numerical operations
import pandas as pd               # For data manipulation
import matplotlib.pyplot as plt   # For visualization
import seaborn as sns             # For enhanced visualization
from sklearn.feature_extraction.text import TfidfVectorizer  # For text processing
from sklearn.metrics.pairwise import cosine_similarity       # For similarity calculation
from surprise import SVD, NMF, KNNBasic, KNNWithMeans       # Recommendation algorithms
from surprise import SlopeOne, BaselineOnly                  # More recommendation algorithms
from surprise import Dataset, Reader                         # For data handling
from surprise.model_selection import cross_validate, train_test_split  # For evaluation
import joblib                     # For saving models
import os                         # For file operations
import re                         # For regular expressions
import urllib.request             # For downloading data
import zipfile                    # For extracting zip files
import time                       # For timing operations
from tqdm.notebook import tqdm    # For progress bars
from wordcloud import WordCloud   # For text visualization
from collections import Counter   # For counting occurrences
import warnings                   # For handling warnings
warnings.filterwarnings('ignore') # Suppress warnings for cleaner output

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Create directories for outputs
os.makedirs('visualizations', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Configure global parameters
RANDOM_SEED = 42      # For reproducibility
TEST_SIZE = 0.2       # 20% of data for testing
RATING_SCALE = (1, 5) # Ratings from 1 to 5

print("Setup complete. Libraries and configurations initialized.")

# =================================================================
# MODULE 2: DATA MANAGEMENT
# =================================================================

def download_dataset(url='https://files.grouplens.org/datasets/movielens/ml-100k.zip', 
                     destination='ml-100k.zip'):
    """
    Downloads and extracts the MovieLens dataset if not already available.
    
    Parameters:
        url (str): URL to download the dataset from
        destination (str): Local path to save the downloaded zip file
    """
    # Check if the dataset directory already exists
    if not os.path.exists('ml-100k'):
        print("Downloading MovieLens 100k dataset...")
        
        # Download the file
        urllib.request.urlretrieve(url, destination)
        
        # Extract the zip file
        with zipfile.ZipFile(destination, 'r') as zip_ref:
            zip_ref.extractall()
        
        print("Dataset downloaded and extracted successfully!")
    else:
        print("Dataset already exists locally.")

def extract_year(title):
    """
    Extracts the release year from a movie title.
    
    Parameters:
        title (str): Movie title with year in parentheses
        
    Returns:
        int or None: Extracted year or None if not found
    """
    # Use regular expression to find year in format (YYYY)
    match = re.search(r'\((\d{4})\)', title)
    return int(match.group(1)) if match else None

def load_data():
    """
    Loads and preprocesses the MovieLens dataset.
    
    Returns:
        tuple: (movies DataFrame, ratings DataFrame, users DataFrame)
    """
    try:
        print("Loading MovieLens 100k dataset...")
        
        # Load movie metadata with genre information
        # The file has a pipe (|) separator and no header, so we provide column names
        movies = pd.read_csv(
            'ml-100k/u.item',
            sep='|',
            encoding='latin-1',
            names=['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
                'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        )
        
        # Load user ratings
        # The file has a tab separator and no header
        ratings = pd.read_csv(
            'ml-100k/u.data',
            sep='\t',
            names=['user_id', 'movie_id', 'rating', 'timestamp']
        )
        
        # Load user information
        # The file has a pipe separator and no header
        users = pd.read_csv(
            'ml-100k/u.user',
            sep='|',
            names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
        )
        
        print("Data loaded successfully. Preprocessing data...")
        
        # ---- Preprocess Movies ----
        # Convert release date to datetime
        movies['release_date'] = pd.to_datetime(movies['release_date'], 
                                               errors='coerce', format='%d-%b-%Y')
        
        # Extract year from title (e.g., "Toy Story (1995)")
        movies['year'] = movies['title'].apply(lambda x: extract_year(x))
        
        # Extract decade for additional insights
        movies['decade'] = movies['year'].apply(lambda x: (x // 10) * 10 if x else None)
        
        # Create a cleaner title (without year)
        movies['clean_title'] = movies['title'].apply(lambda x: re.sub(r'\s*\(\d{4}\)$', '', x))
        
        # Create a consolidated genre string for text analysis
        genre_cols = movies.columns[5:24]  # Columns from 'unknown' to 'Western'
        movies['genre_list'] = movies.apply(
            lambda x: ' '.join([genre for genre, is_genre in zip(genre_cols, x[genre_cols]) if is_genre == 1]), 
            axis=1
        )
        
        # ---- Preprocess Ratings ----
        # Convert timestamp to datetime
        ratings['date'] = pd.to_datetime(ratings['timestamp'], unit='s')
        
        # Add day of week and hour of day for temporal analysis
        ratings['day_of_week'] = ratings['date'].dt.day_name()
        ratings['hour_of_day'] = ratings['date'].dt.hour
        
        # ---- Data Validation ----
        # Check for duplicate movie IDs
        duplicates = movies['movie_id'].duplicated().sum()
        if duplicates > 0:
            print(f"Warning: Found {duplicates} duplicate movie IDs")
            
        # Check for ratings with non-existent movies
        invalid_ratings = ratings[~ratings['movie_id'].isin(movies['movie_id'])]
        if len(invalid_ratings) > 0:
            print(f"Warning: Found {len(invalid_ratings)} ratings for non-existent movies")
            # Remove invalid ratings
            ratings = ratings[ratings['movie_id'].isin(movies['movie_id'])]
            
        print(f"Processed {len(movies)} movies, {len(ratings)} ratings, and {len(users)} users.")
        return movies, ratings, users
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None, None

# =================================================================
# MODULE 3: EXPLORATORY DATA ANALYSIS
# =================================================================

def analyze_data(movies, ratings, users):
    """
    Performs comprehensive exploratory data analysis on the dataset.
    
    Parameters:
        movies (DataFrame): Movies data
        ratings (DataFrame): Ratings data
        users (DataFrame): Users data
        
    Returns:
        dict: Key statistics about the dataset
    """
    print("\n===== EXPLORATORY DATA ANALYSIS =====")
    start_time = time.time()
    
    # ---- Basic Statistics ----
    print("\n--- RATINGS STATISTICS ---")
    print(f"Total ratings: {len(ratings)}")
    print(f"Unique users: {ratings['user_id'].nunique()}")
    print(f"Unique movies: {ratings['movie_id'].nunique()}")
    print(f"Rating distribution:\n{ratings['rating'].value_counts().sort_index()}")
    print(f"Average rating: {ratings['rating'].mean():.2f}")
    
    # Create a ratings matrix (sparse)
    # This matrix has users as rows and movies as columns
    ratings_matrix = ratings.pivot_table(index='user_id', columns='movie_id', values='rating')
    
    # Calculate matrix sparsity (percentage of empty cells)
    sparsity = 1 - ratings_matrix.count().sum() / (ratings_matrix.shape[0] * ratings_matrix.shape[1])
    print(f"\nRatings matrix shape: {ratings_matrix.shape}")
    print(f"Sparsity: {sparsity * 100:.2f}%")
    
    # ---- Cold Start Analysis ----
    # Cold start is a common problem in recommendation systems where we lack data
    # for new users or items
    print("\n--- COLD START ANALYSIS ---")
    
    # Count ratings per user and per movie
    user_rating_counts = ratings.groupby('user_id')['rating'].count()
    movie_rating_counts = ratings.groupby('movie_id')['rating'].count()
    
    # Define cold users/items (few ratings)
    cold_users = user_rating_counts[user_rating_counts < 5].index
    cold_movies = movie_rating_counts[movie_rating_counts < 5].index
    
    print(f"Cold users (< 5 ratings): {len(cold_users)} ({len(cold_users)/len(user_rating_counts)*100:.1f}%)")
    print(f"Cold movies (< 5 ratings): {len(cold_movies)} ({len(cold_movies)/len(movie_rating_counts)*100:.1f}%)")
    
    # ---- Anomaly Detection ----
    print("\n--- ANOMALY DETECTION ---")
    
    # Users with unusually high rating counts (>2 std devs from mean)
    outlier_users = user_rating_counts[user_rating_counts > user_rating_counts.mean() + 2*user_rating_counts.std()]
    print(f"Users with unusually high rating counts: {len(outlier_users)}")
    
    # Movies with unusually high rating counts
    outlier_movies = movie_rating_counts[movie_rating_counts > movie_rating_counts.mean() + 2*movie_rating_counts.std()]
    print(f"Movies with unusually high rating counts: {len(outlier_movies)}")
    
    # Most rated movies
    top_movies = movie_rating_counts.sort_values(ascending=False).head(10)
    print("\nTop 10 most rated movies:")
    for i, (movie_id, count) in enumerate(zip(top_movies.index, top_movies.values)):
        movie_title = movies[movies['movie_id'] == movie_id]['title'].values[0]
        print(f"{i+1}. {movie_title}: {count} ratings")
    
    # ---- Temporal Analysis ----
    print("\n--- TEMPORAL ANALYSIS ---")
    
    # Ratings over time (by month)
    ratings_by_month = ratings.groupby(ratings['date'].dt.to_period('M')).size()
    print(f"Date range: {ratings_by_month.index.min()} to {ratings_by_month.index.max()}")
    print(f"Most active month: {ratings_by_month.idxmax()} with {ratings_by_month.max()} ratings")
    
    # Ratings by day of week
    ratings_by_dow = ratings.groupby('day_of_week').size()
    print(f"Most active day: {ratings_by_dow.idxmax()} with {ratings_by_dow.max()} ratings")
    
    # Ratings by hour of day
    ratings_by_hour = ratings.groupby('hour_of_day').size()
    print(f"Most active hour: {ratings_by_hour.idxmax()} with {ratings_by_hour.max()} ratings")
    
    # ---- User Demographics ----
    print("\n--- USER DEMOGRAPHICS ---")
    
    # Gender distribution
    gender_counts = users['gender'].value_counts()
    print(f"Gender distribution: {gender_counts.to_dict()}")
    
    # Age distribution
    age_groups = pd.cut(users['age'], bins=[0, 18, 25, 35, 50, 100], 
                       labels=['<18', '18-24', '25-35', '36-50', '50+'])
    age_distribution = age_groups.value_counts().sort_index()
    print(f"Age distribution:\n{age_distribution}")
    
    # Top occupations
    occupation_counts = users['occupation'].value_counts().head(10)
    print(f"Top 10 occupations:\n{occupation_counts}")
    
    # ---- Genre Analysis ----
    print("\n--- GENRE ANALYSIS ---")
    
    # Genre distribution
    genre_cols = movies.columns[5:24]  # Columns for genres
    genre_counts = movies[genre_cols].sum().sort_values(ascending=False)
    print("Genre distribution:")
    for genre, count in genre_counts.items():
        print(f"{genre}: {count} movies ({count/len(movies)*100:.1f}%)")
    
    # Average ratings by genre
    genre_avg_ratings = {}
    for genre in genre_cols:
        # Get movies in this genre
        genre_movies = movies[movies[genre] == 1]['movie_id'].tolist()
        # Get ratings for these movies
        genre_ratings = ratings[ratings['movie_id'].isin(genre_movies)]['rating']
        # Calculate average rating if there are ratings
        if len(genre_ratings) > 0:
            genre_avg_ratings[genre] = genre_ratings.mean()
    
    # Sort genres by average rating
    sorted_genre_ratings = {k: v for k, v in sorted(genre_avg_ratings.items(), key=lambda item: item[1], reverse=True)}
    print("\nGenres by average rating:")
    for genre, avg_rating in list(sorted_genre_ratings.items())[:5]:  # Top 5 genres
        print(f"{genre}: {avg_rating:.2f}")
    
    # ---- Create Visualizations ----
    print("\n--- GENERATING VISUALIZATIONS ---")
    
    # 1. Rating Distribution
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    sns.countplot(x='rating', data=ratings, palette='viridis')
    plt.title('Rating Distribution', fontsize=14)
    plt.xlabel('Rating')
    plt.ylabel('Count')
    
    # 2. Ratings per user
    plt.subplot(2, 2, 2)
    sns.histplot(user_rating_counts, bins=30, kde=True, color=colors[1])
    plt.title('Ratings per User', fontsize=14)
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Users')
    
    # 3. Genre distribution
    plt.subplot(2, 2, 3)
    top_genres = genre_counts.head(10)
    sns.barplot(x=top_genres.values, y=top_genres.index, palette='viridis')
    plt.title('Top 10 Genres', fontsize=14)
    plt.xlabel('Number of Movies')
    
    # 4. Average rating by age group
    plt.subplot(2, 2, 4)
    # Merge ratings with user age data
    age_ratings = ratings.merge(users[['user_id', 'age']], on='user_id')
    age_ratings['age_group'] = pd.cut(age_ratings['age'], 
                                    bins=[0, 18, 25, 35, 50, 100], 
                                    labels=['<18', '18-24', '25-35', '36-50', '50+'])
    # Calculate average rating by age group
    age_avg = age_ratings.groupby('age_group')['rating'].mean().sort_index()
    sns.barplot(x=age_avg.index, y=age_avg.values, palette='viridis')
    plt.title('Average Rating by Age Group', fontsize=14)
    plt.xlabel('Age Group')
    plt.ylabel('Average Rating')
    
    plt.tight_layout()
    plt.savefig('visualizations/eda_basic.png', dpi=300)
    plt.close()
    
    # 5. Create a genre word cloud
    genre_data = {}
    for genre, count in genre_counts.items():
        genre_data[genre] = count
    
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                         max_words=100, contour_width=3, contour_color='steelblue')
    wordcloud.generate_from_frequencies(genre_data)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Movie Genres Distribution', fontsize=16)
    plt.savefig('visualizations/genre_wordcloud.png', dpi=300)
    plt.close()
    
    # 6. Temporal patterns
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    # Convert to list for plotting
    months = [str(m) for m in ratings_by_month.index]
    plt.plot(months, ratings_by_month.values, marker='o')
    plt.title('Ratings by Month', fontsize=14)
    plt.xlabel('Month')
    plt.ylabel('Number of Ratings')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 1, 2)
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_counts = ratings['day_of_week'].value_counts().reindex(dow_order)
    sns.barplot(x=dow_counts.index, y=dow_counts.values, palette='viridis')
    plt.title('Ratings by Day of Week', fontsize=14)
    plt.xlabel('Day')
    plt.ylabel('Number of Ratings')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('visualizations/temporal_patterns.png', dpi=300)
    plt.close()
    
    # Print elapsed time
    elapsed_time = time.time() - start_time
    print(f"\nExploratory data analysis completed in {elapsed_time:.2f} seconds.")
    
    # Return key statistics for later use
    return {
        'sparsity': sparsity,
        'cold_start_movies_pct': len(cold_movies)/len(movie_rating_counts)*100,
        'cold_start_users_pct': len(cold_users)/len(user_rating_counts)*100 if len(user_rating_counts) > 0 else 0,
        'avg_rating': ratings['rating'].mean(),
        'rating_count': len(ratings),
        'user_count': ratings['user_id'].nunique(),
        'movie_count': ratings['movie_id'].nunique()
    }

# =================================================================
# MODULE 4: FEATURE ENGINEERING
# =================================================================

def engineer_features(movies, ratings, users):
    """
    Creates enhanced features for improved recommendations.
    
    Parameters:
        movies (DataFrame): Movies data
        ratings (DataFrame): Ratings data
        users (DataFrame): Users data
        
    Returns:
        tuple: Enhanced dataframes with new features
    """
    print("\n===== FEATURE ENGINEERING =====")
    start_time = time.time()
    
    # ---- 1. Enhanced Movie Features ----
    movies_enhanced = movies.copy()
    
    # Make sure year is extracted from release date if missing from title
    movies_enhanced['year'] = movies_enhanced.apply(
        lambda x: x['year'] if x['year'] else 
                 (x['release_date'].year if pd.notnull(x['release_date']) else None), 
        axis=1
    )
    
    # ---- 2. Enhanced User Features ----
    users_enhanced = users.copy()
    
    # One-hot encode categorical user features
    users_enhanced = pd.get_dummies(users_enhanced, columns=['gender', 'occupation'], drop_first=False)
    
    # Bin ages into meaningful groups
    users_enhanced['age_group'] = pd.cut(users_enhanced['age'], 
                                        bins=[0, 18, 25, 35, 50, 100], 
                                        labels=['<18', '18-24', '25-35', '36-50', '50+'])
    users_enhanced = pd.get_dummies(users_enhanced, columns=['age_group'], drop_first=False)
    
    # ---- 3. Enhanced Rating Features ----
    ratings_enhanced = ratings.copy()
    
    # Add user average rating and standard deviation
    user_avg = ratings.groupby('user_id')['rating'].mean().reset_index()
    user_avg.columns = ['user_id', 'user_avg_rating']
    
    user_std = ratings.groupby('user_id')['rating'].std().fillna(0).reset_index()
    user_std.columns = ['user_id', 'user_rating_std']
    
    # Add movie average rating and count
    movie_avg = ratings.groupby('movie_id')['rating'].mean().reset_index()
    movie_avg.columns = ['movie_id', 'movie_avg_rating']
    
    movie_count = ratings.groupby('movie_id')['rating'].count().reset_index()
    movie_count.columns = ['movie_id', 'movie_rating_count']
    
    # Add temporal features
    # Calculate days since the rating was made
    current_time = ratings['timestamp'].max()
    ratings_enhanced['days_since_rating'] = (current_time - ratings['timestamp']) / (60*60*24)
    
    # Merge all features with ratings
    ratings_enhanced = ratings_enhanced.merge(user_avg, on='user_id', how='left')
    ratings_enhanced = ratings_enhanced.merge(user_std, on='user_id', how='left')
    ratings_enhanced = ratings_enhanced.merge(movie_avg, on='movie_id', how='left')
    ratings_enhanced = ratings_enhanced.merge(movie_count, on='movie_id', how='left')
    
    # Calculate interaction features
    # How much a user's rating differs from their average
    ratings_enhanced['rating_diff_from_user_avg'] = ratings_enhanced['rating'] - ratings_enhanced['user_avg_rating']
    # How much a user's rating differs from the movie's average
    ratings_enhanced['rating_diff_from_movie_avg'] = ratings_enhanced['rating'] - ratings_enhanced['movie_avg_rating']
    
    # Calculate popularity percentile for movies
    movie_count['popularity_percentile'] = movie_count['movie_rating_count'].rank(pct=True)
    
    print(f"Added {len(ratings_enhanced.columns) - len(ratings.columns)} new features to ratings data")
    print(f"Added {len(users_enhanced.columns) - len(users.columns)} new features to users data")
    
    elapsed_time = time.time() - start_time
    print(f"Feature engineering completed in {elapsed_time:.2f} seconds.")
    
    return movies_enhanced, ratings_enhanced, users_enhanced, movie_count




# =================================================================
# MODULE 5: CONTENT-BASED FILTERING
# =================================================================

def build_content_based_recommender(movies):
    """
    Builds a content-based recommender using movie attributes.
    
    Parameters:
        movies (DataFrame): Enhanced movies DataFrame
    
    Returns:
        tuple: (TF-IDF matrix, cosine similarity matrix, movie indices)
    """
    print("\n===== BUILDING CONTENT-BASED RECOMMENDER =====")
    start_time = time.time()
    
    # Step 1: Prepare the content features
    # Combine relevant features into a single content string
    movies['content'] = movies['genre_list'].fillna('') + ' ' + \
                        movies['clean_title'].fillna('') + ' ' + \
                        movies['year'].fillna('').astype(str)
    
    print(f"Example content string: {movies['content'].iloc[0]}")
    
    # Step 2: Create TF-IDF vectorizer
    # TF-IDF (Term Frequency-Inverse Document Frequency) converts text to numbers
    # - It gives higher weight to important terms and lower weight to common terms
    # - stop_words='english' removes common words like 'the', 'and', etc.
    # - ngram_range=(1, 2) includes both single words and pairs of words
    tfidf = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),  # Include bigrams for better context
        min_df=2,            # Ignore terms that appear in fewer than 2 documents
        max_df=0.8,          # Ignore terms that appear in more than 80% of documents
        max_features=5000    # Limit features to reduce dimensionality
    )
    
    # Step 3: Generate TF-IDF matrix
    # This converts our text content into a numerical matrix
    tfidf_matrix = tfidf.fit_transform(movies['content'])
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # Save the feature names for later explanation
    feature_names = tfidf.get_feature_names_out()
    
    # Step 4: Calculate cosine similarity
    # This measures how similar each movie is to every other movie
    print("Calculating cosine similarity matrix...")
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print(f"Cosine similarity matrix shape: {cosine_sim.shape}")
    
    # Step 5: Create a mapping of movie IDs to matrix indices
    # This allows us to look up a movie by ID
    indices = pd.Series(movies.index, index=movies['movie_id']).drop_duplicates()
    
    # Save the model components
    print("Saving content-based model components...")
    joblib.dump(tfidf, 'models/content_tfidf_vectorizer.pkl')
    joblib.dump(indices, 'models/content_movie_indices.pkl')
    np.save('models/content_cosine_sim.npy', cosine_sim)
    
    # Visualize top features
    feature_importance = np.sum(tfidf_matrix.toarray(), axis=0)
    top_features_idx = np.argsort(feature_importance)[-20:]
    top_features = [feature_names[i] for i in top_features_idx]
    
    plt.figure(figsize=(12, 6))
    plt.barh(top_features, feature_importance[top_features_idx])
    plt.title('Top 20 Important Features in Content-Based Filtering')
    plt.xlabel('Importance Score')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('visualizations/content_feature_importance.png', dpi=300)
    plt.close()
    
    elapsed_time = time.time() - start_time
    print(f"Content-based recommender built in {elapsed_time:.2f} seconds.")
    
    return tfidf_matrix, cosine_sim, indices

def get_content_based_recommendations(movie_id, cosine_sim, indices, movies, n=10):
    """
    Generate content-based movie recommendations.
    
    Parameters:
        movie_id (int): Movie ID to get recommendations for
        cosine_sim (ndarray): Cosine similarity matrix
        indices (Series): Mapping of movie IDs to matrix indices
        movies (DataFrame): Movies data
        n (int): Number of recommendations to return
        
    Returns:
        DataFrame: Recommendations with similarity scores and explanations
    """
    # Step 1: Get the index of the movie
    try:
        idx = indices[movie_id]
    except KeyError:
        print(f"Movie ID {movie_id} not found in the dataset.")
        return pd.DataFrame()
    
    # Step 2: Get similarity scores and sort them
    # Enumerate adds an index to each similarity score
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort by similarity (highest first)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Step 3: Get top n similar movies (excluding the movie itself)
    sim_scores = sim_scores[1:n+1]  # Skip first one (it's the movie itself)
    movie_indices = [i[0] for i in sim_scores]  # Get indices
    similarity_scores = [i[1] for i in sim_scores]  # Get scores
    
    # Step 4: Create a DataFrame with recommendations
    recommendations = movies.iloc[movie_indices][['movie_id', 'title', 'genre_list', 'year']]
    recommendations['similarity_score'] = similarity_scores
    
    # Step 5: Add simple explanation based on genres
    reference_movie = movies.iloc[idx]
    recommendations['explanation'] = recommendations.apply(
        lambda x: f"Similar to '{reference_movie['title']}' in genres: {x['genre_list']}", 
        axis=1
    )
    
    return recommendations


# =================================================================
# MODULE 6: COLLABORATIVE FILTERING
# =================================================================
def prepare_surprise_data(ratings):
    """
    Prepares data for Surprise library.
    
    Parameters:
        ratings (DataFrame): Ratings data
        
    Returns:
        tuple: (full dataset, trainset, testset)
    """
    # Step 1: Define the format of the ratings
    # The Reader class helps Surprise understand our rating scale
    reader = Reader(rating_scale=RATING_SCALE)
    
    # Step 2: Load the data into Surprise's Dataset format
    # We only need user_id, movie_id, and rating columns
    data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)
    
    # Step 3: Split the data into training and test sets
    # 80% for training, 20% for testing
    trainset, testset = train_test_split(data, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    
    # Use trainset.n_ratings instead of trying to get length of a generator
    print(f"Data prepared: {trainset.n_ratings} training ratings and {len(testset)} test ratings")
    return data, trainset, testset


def evaluate_collaborative_models(data):
    """
    Evaluates multiple collaborative filtering models.
    
    Parameters:
        data (Dataset): Surprise dataset object
        
    Returns:
        dict: Model evaluation results
    """
    print("\n===== EVALUATING COLLABORATIVE FILTERING MODELS =====")
    start_time = time.time()
    
    # Step 1: Define models to evaluate
    # We'll try different algorithms to see which performs best
    models = {
        'SVD': SVD(random_state=RANDOM_SEED),  # Matrix factorization
        'NMF': NMF(random_state=RANDOM_SEED),  # Non-negative matrix factorization
        'KNN_User': KNNBasic(k=50, sim_options={'name': 'pearson', 'user_based': True}, verbose=False),  # User-based collaborative filtering
        'KNN_Item': KNNBasic(k=50, sim_options={'name': 'pearson', 'user_based': False}, verbose=False),  # Item-based collaborative filtering
        'Baseline': BaselineOnly(),  # Simple baseline using averages
        'SlopeOne': SlopeOne()  # Simple but effective algorithm
    }
    
    # Step 2: Prepare results dictionary
    results = {}
    
    # Step 3: Evaluate each model with 5-fold cross-validation
    for name, model in tqdm(models.items(), desc="Evaluating models"):
        # Cross-validation splits the data into 5 parts and evaluates on each
        cv_results = cross_validate(model, data, measures=['RMSE', 'MAE'], 
                                   cv=5, verbose=False, n_jobs=-1)
        
        # Store results - using numpy functions to calculate statistics
        results[name] = {
            'RMSE': np.mean(cv_results['test_rmse']),  # Root Mean Square Error
            'MAE': np.mean(cv_results['test_mae']),    # Mean Absolute Error
            'RMSE_std': np.std(cv_results['test_rmse']),  # Standard deviation
            'MAE_std': np.std(cv_results['test_mae']),    # Standard deviation
            'Fit_time': np.mean(cv_results['fit_time'])   # Average training time
        }
        
        # Print results
        print(f"{name}: RMSE = {results[name]['RMSE']:.4f} (±{results[name]['RMSE_std']:.4f}), "
              f"MAE = {results[name]['MAE']:.4f} (±{results[name]['MAE_std']:.4f})")
    
    # Step 4: Visualize model comparison
    plt.figure(figsize=(12, 6))
    
    # Plot RMSE (lower is better)
    plt.subplot(1, 2, 1)
    models_names = list(results.keys())
    rmse_values = [results[model]['RMSE'] for model in models_names]
    rmse_errors = [results[model]['RMSE_std'] for model in models_names]
    
    plt.bar(models_names, rmse_values, yerr=rmse_errors, capsize=10, color=colors[:len(models_names)])
    plt.title('RMSE Comparison')
    plt.ylabel('RMSE (lower is better)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot MAE (lower is better)
    plt.subplot(1, 2, 2)
    mae_values = [results[model]['MAE'] for model in models_names]
    mae_errors = [results[model]['MAE_std'] for model in models_names]
    
    plt.bar(models_names, mae_values, yerr=mae_errors, capsize=10, color=colors[:len(models_names)])
    plt.title('MAE Comparison')
    plt.ylabel('MAE (lower is better)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('visualizations/collaborative_model_comparison.png', dpi=300)
    plt.close()
    
    elapsed_time = time.time() - start_time
    print(f"Model evaluation completed in {elapsed_time:.2f} seconds.")
    
    return results

def train_best_models(trainset):
    """
    Trains the best performing collaborative filtering models.
    
    Parameters:
        trainset (Trainset): Surprise trainset object
        
    Returns:
        dict: Trained models
    """
    print("\n===== TRAINING BEST MODELS =====")
    start_time = time.time()
    
    # Step 1: Define the models to train
    # Based on typical performance, we'll train these three:
    svd_model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=RANDOM_SEED)
    knn_item_model = KNNWithMeans(k=50, sim_options={'name': 'pearson', 'user_based': False}, verbose=False)
    baseline_model = BaselineOnly()
    
    # Step 2: Create models dictionary
    models = {
        'SVD': svd_model,
        'KNN_Item': knn_item_model,
        'Baseline': baseline_model
    }
    
    # Step 3: Train each model
    for name, model in tqdm(models.items(), desc="Training models"):
        model.fit(trainset)
        
    # Step 4: Save trained models
    for name, model in models.items():
        joblib.dump(model, f'models/model_{name}.pkl')
    
    elapsed_time = time.time() - start_time
    print(f"Model training completed in {elapsed_time:.2f} seconds.")
    
    return models

# =================================================================
# MODULE 7: HYBRID RECOMMENDER
# =================================================================

def build_hybrid_recommender(cf_models, cosine_sim, indices, movies_df, ratings_df, alpha=0.7):
    """
    Build a hybrid recommender that combines collaborative filtering and content-based filtering.
    
    Parameters:
        cf_models (dict): Trained collaborative filtering models
        cosine_sim (ndarray): Content-based cosine similarity matrix
        indices (Series): Mapping of movie IDs to matrix indices
        movies_df (DataFrame): Movies data
        ratings_df (DataFrame): Ratings data
        alpha (float): Weight for collaborative filtering (1-alpha for content-based)
        
    Returns:
        function: Hybrid recommendation function
    """
    print("\n===== BUILDING HYBRID RECOMMENDER =====")
    
    # Step 1: Choose the best CF model (typically SVD or KNN_Item)
    best_cf_model = cf_models['SVD']  # Can be changed based on evaluation results
    
    # Step 2: Create popularity scores for fallback recommendations
    # This will be used when we don't have enough data for a user
    movie_popularity = ratings_df.groupby('movie_id')['rating'].agg(['count', 'mean'])
    movie_popularity['score'] = movie_popularity['count'] * movie_popularity['mean']
    movie_popularity = movie_popularity.sort_values('score', ascending=False)
    
    # Step 3: Define the hybrid recommendation function
    def hybrid_recommend(user_id, movie_id=None, n=10):
        """
        Generate hybrid recommendations for a user, optionally based on a specific movie.
        
        Parameters:
            user_id (int): User ID
            movie_id (int, optional): Movie ID to base recommendations on
            n (int): Number of recommendations
            
        Returns:
            DataFrame: Hybrid recommendations
        """
        # Initialize empty recommendation lists
        cf_recs = []
        cb_recs = []
        
        # Step 3.1: Get collaborative filtering recommendations
        try:
            # Get all movie IDs
            all_movie_ids = movies_df['movie_id'].unique()
            
            # Get movies the user has already rated
            user_rated = set(ratings_df[ratings_df['user_id'] == user_id]['movie_id'])
            
            # Get candidate movies (those not yet rated by the user)
            candidate_items = list(set(all_movie_ids) - user_rated)
            
            # Predict ratings for all candidate items
            cf_predictions = []
            for item_id in candidate_items:
                try:
                    pred = best_cf_model.predict(user_id, item_id)
                    cf_predictions.append((item_id, pred.est))
                except:
                    continue
            
            # Sort by predicted rating
            cf_predictions.sort(key=lambda x: x[1], reverse=True)
            
            # Get top N*2 recommendations (we'll filter later)
            cf_recs = cf_predictions[:n*2]
            
        except Exception as e:
            print(f"Error getting CF recommendations: {e}")
        
        # Step 3.2: Get content-based recommendations if a movie is provided
        if movie_id is not None:
            try:
                cb_recommendations = get_content_based_recommendations(
                    movie_id, cosine_sim, indices, movies_df, n=n*2)
                
                if not cb_recommendations.empty:
                    cb_recs = [(row['movie_id'], row['similarity_score']) 
                               for _, row in cb_recommendations.iterrows()]
            except Exception as e:
                print(f"Error getting CB recommendations: {e}")
        
        # Step 3.3: Combine recommendations
        combined_recs = {}
        
        # Add CF recommendations with weight alpha
        for item_id, score in cf_recs:
            if item_id not in user_rated:  # Ensure we don't recommend already rated items
                combined_recs[item_id] = alpha * score
        
        # Add CB recommendations with weight (1-alpha)
        if movie_id is not None:
            for item_id, score in cb_recs:
                if item_id not in user_rated:  # Ensure we don't recommend already rated items
                    if item_id in combined_recs:
                        combined_recs[item_id] += (1 - alpha) * score
                    else:
                        combined_recs[item_id] = (1 - alpha) * score
        
        # Step 3.4: Sort by combined score
        sorted_recs = sorted(combined_recs.items(), key=lambda x: x[1], reverse=True)
        
        # Step 3.5: If we don't have enough recommendations, add popular movies
        if len(sorted_recs) < n:
            popular_movies = movie_popularity.index.tolist()
            for item_id in popular_movies:
                if item_id not in user_rated and item_id not in [r[0] for r in sorted_recs]:
                    sorted_recs.append((item_id, 0))
                    if len(sorted_recs) >= n:
                        break
        
        # Step 3.6: Get top N recommendations
        top_n_recs = sorted_recs[:n]
        
        # Step 3.7: Create a DataFrame with recommendations
        rec_df = pd.DataFrame(top_n_recs, columns=['movie_id', 'score'])
        rec_df = rec_df.merge(movies_df[['movie_id', 'title', 'genre_list']], on='movie_id')
        
        # Step 3.8: Add recommendation source
        rec_df['source'] = rec_df['movie_id'].apply(
            lambda x: 'Hybrid' if (x in [r[0] for r in cf_recs] and x in [r[0] for r in cb_recs]) else
                     ('Collaborative' if x in [r[0] for r in cf_recs] else
                      ('Content' if x in [r[0] for r in cb_recs] else 'Popular'))
        )
        
        return rec_df
    
    print(f"Hybrid recommender built with alpha={alpha} (weight for collaborative filtering)")
    return hybrid_recommend






# =================================================================
# MODULE 8: EVALUATION & VISUALIZATION
# =================================================================

def evaluate_top_n_recommendations(models, testset, movies_df, n=10):
    """
    Evaluate top-N recommendations using precision, recall, and NDCG.
    
    Parameters:
        models (dict): Trained models
        testset (list): Test set for evaluation
        movies_df (DataFrame): Movies data
        n (int): Number of recommendations
        
    Returns:
        dict: Evaluation metrics
    """
    print("\n===== EVALUATING TOP-N RECOMMENDATIONS =====")
    start_time = time.time()
    
    # Step 1: Extract unique users from the test set
    test_users = list(set([uid for (uid, _, _) in testset]))
    
    # For evaluation, we'll use a sample of users
    eval_users = test_users[:50] if len(test_users) > 50 else test_users
    print(f"Evaluating recommendations for {len(eval_users)} users")
    
    # Step 2: Get all movie IDs
    all_movie_ids = movies_df['movie_id'].unique()
    
    # Step 3: Create a mapping of users to their rated items in the test set
    # We'll consider items with rating >= 4 as "relevant"
    user_relevant_items = {}
    for uid, iid, true_rating in testset:
        if uid not in user_relevant_items:
            user_relevant_items[uid] = {}
        # Items with rating >= 4 are considered relevant
        user_relevant_items[uid][iid] = true_rating >= 4
    
    # Step 4: Prepare metrics storage
    metrics = {model_name: {'precision': [], 'recall': [], 'ndcg': []} for model_name in models}
    recommendations = {model_name: {} for model_name in models}
    
    # Step 5: Generate and evaluate recommendations for each model
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        
        # For each user
        for user_id in tqdm(eval_users, desc=f"{model_name} evaluation"):
            # Get all items the user hasn't rated in the training set
            user_train_items = set()
            for item_id, _ in model.trainset.ur[model.trainset.to_inner_uid(user_id)]:
                user_train_items.add(model.trainset.to_raw_iid(item_id))
                
            # Get candidate items (those not rated in training)
            candidate_items = list(set(all_movie_ids) - user_train_items)
            
            # Predict ratings for all candidate items
            user_predictions = []
            for item_id in candidate_items:
                try:
                    pred = model.predict(user_id, item_id)
                    user_predictions.append((item_id, pred.est))
                except:
                    continue
            
            # Sort predictions by estimated rating
            user_predictions.sort(key=lambda x: x[1], reverse=True)
            
            # Get top N recommendations
            user_top_n = user_predictions[:n]
            recommendations[model_name][user_id] = [
                (iid, est, movies_df[movies_df['movie_id'] == iid]['title'].values[0] 
                 if len(movies_df[movies_df['movie_id'] == iid]) > 0 else "Unknown") 
                for iid, est in user_top_n
            ]
            
            # Calculate metrics if user has relevant items in the test set
            if user_id in user_relevant_items and any(user_relevant_items[user_id].values()):
                # Get relevant items
                relevant_items = {iid for iid, is_relevant in user_relevant_items[user_id].items() 
                                if is_relevant}
                
                # Get recommended items
                recommended_items = {iid for iid, _, _ in recommendations[model_name][user_id]}
                
                # Calculate precision (what fraction of recommended items are relevant)
                if len(recommended_items) > 0:
                    precision = len(relevant_items & recommended_items) / len(recommended_items)
                    metrics[model_name]['precision'].append(precision)
                
                # Calculate recall (what fraction of relevant items are recommended)
                if len(relevant_items) > 0:
                    recall = len(relevant_items & recommended_items) / len(relevant_items)
                    metrics[model_name]['recall'].append(recall)
                
                # Calculate NDCG (Normalized Discounted Cumulative Gain)
                # This measures the quality of ranking, giving higher weight to items ranked higher
                
                # Create a relevance vector (1 if item is relevant, 0 otherwise)
                relevance = [1 if iid in relevant_items else 0 
                            for iid, _, _ in recommendations[model_name][user_id]]
                
                # Calculate DCG (Discounted Cumulative Gain)
                dcg = sum([(2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevance)])
                
                # Calculate ideal DCG (if all relevant items were ranked first)
                ideal_relevance = sorted(relevance, reverse=True)
                idcg = sum([(2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_relevance)])
                
                # Calculate NDCG
                if idcg > 0:
                    ndcg = dcg / idcg
                    metrics[model_name]['ndcg'].append(ndcg)
    
    # Step 6: Calculate average metrics
    for model_name in models:
    # Create a static list of the original metric keys
        metric_keys = list(metrics[model_name].keys())
    
    # Iterate over the static list instead of the dictionary
        for metric in metric_keys:
            if metrics[model_name][metric]:
                metrics[model_name][f'avg_{metric}'] = np.mean(metrics[model_name][metric])
                metrics[model_name][f'users_with_{metric}'] = len(metrics[model_name][metric])
            else:
                metrics[model_name][f'avg_{metric}'] = 0
                metrics[model_name][f'users_with_{metric}'] = 0
  
    
    # Step 7: Calculate diversity metrics
    diversity_metrics = calculate_diversity_metrics(recommendations, movies_df)
    
    # Step 8: Combine metrics
    all_metrics = {model_name: {**metrics[model_name], **diversity_metrics[model_name]} 
                   for model_name in models}
    
    # Step 9: Print results
    print("\n----- Top-N Recommendation Results -----")
    for model_name in models:
        print(f"\n{model_name}:")
        print(f"Precision@{n}: {all_metrics[model_name].get('avg_precision', 0):.4f}")
        print(f"Recall@{n}: {all_metrics[model_name].get('avg_recall', 0):.4f}")
        print(f"NDCG@{n}: {all_metrics[model_name].get('avg_ndcg', 0):.4f}")
        print(f"Users with at least one relevant item: {all_metrics[model_name].get('users_with_precision', 0)}")
        print(f"Genre diversity (entropy): {all_metrics[model_name].get('genre_entropy', 0):.4f}")
        print(f"Unique recommendations ratio: {all_metrics[model_name].get('unique_ratio', 0):.4f}")
    
    # Step 10: Visualize metrics
    visualize_recommendation_metrics(all_metrics, n)
    
    elapsed_time = time.time() - start_time
    print(f"Top-N recommendation evaluation completed in {elapsed_time:.2f} seconds.")
    
    return all_metrics, recommendations

def calculate_diversity_metrics(recommendations, movies_df):
    """
    Calculate diversity metrics for recommendations.
    
    Parameters:
        recommendations (dict): Recommendations per model and user
        movies_df (DataFrame): Movies data
        
    Returns:
        dict: Diversity metrics
    """
    diversity_metrics = {}
    
    # Genre columns
    genre_cols = movies_df.columns[5:24]
    
    for model_name, user_recs in recommendations.items():
        diversity_metrics[model_name] = {}
        
        # Collect all recommended movie IDs
        all_rec_movies = []
        for user_id, recs in user_recs.items():
            all_rec_movies.extend([movie_id for movie_id, _, _ in recs])
        
        # Calculate unique recommendations ratio
        unique_movies = len(set(all_rec_movies))
        total_recs = len(all_rec_movies)
        diversity_metrics[model_name]['unique_ratio'] = unique_movies / total_recs if total_recs > 0 else 0
        
        # Calculate Gini coefficient (popularity diversity)
        # Gini coefficient measures inequality - lower means more equal distribution
        movie_counts = Counter(all_rec_movies)
        counts = sorted(movie_counts.values())
        n = len(counts)
        
        if n > 0:
            # Calculate Gini coefficient
            index = np.arange(1, n + 1)
            gini = 1 - 2 * np.sum((n + 1 - index) * counts) / (n * np.sum(counts))
            diversity_metrics[model_name]['gini_coefficient'] = gini
        else:
            diversity_metrics[model_name]['gini_coefficient'] = 0
        
        # Calculate genre diversity
        # We want to see if recommendations cover diverse genres
        genre_distribution = {}
        for movie_id in set(all_rec_movies):
            movie_row = movies_df[movies_df['movie_id'] == movie_id]
            if not movie_row.empty:
                for genre in genre_cols:
                    if movie_row[genre].values[0] == 1:
                        if genre in genre_distribution:
                            genre_distribution[genre] += 1
                        else:
                            genre_distribution[genre] = 1
        
        # Calculate genre entropy (higher means more diverse)
        if genre_distribution:
            genre_counts = np.array(list(genre_distribution.values()))
            genre_probs = genre_counts / genre_counts.sum()
            entropy = -np.sum(genre_probs * np.log2(genre_probs))
            diversity_metrics[model_name]['genre_entropy'] = entropy
        else:
            diversity_metrics[model_name]['genre_entropy'] = 0
            
    return diversity_metrics

def visualize_recommendation_metrics(metrics, n=10):
    """
    Visualize recommendation metrics for all models.
    
    Parameters:
        metrics (dict): Metrics for each model
        n (int): Number of recommendations used
    """
    plt.figure(figsize=(15, 10))
    
    # 1. Precision, Recall, NDCG
    plt.subplot(2, 2, 1)
    model_names = list(metrics.keys())
    x = np.arange(len(model_names))
    width = 0.25
    
    precision_values = [metrics[model].get('avg_precision', 0) for model in model_names]
    recall_values = [metrics[model].get('avg_recall', 0) for model in model_names]
    ndcg_values = [metrics[model].get('avg_ndcg', 0) for model in model_names]
    
    plt.bar(x - width, precision_values, width, label=f'Precision@{n}')
    plt.bar(x, recall_values, width, label=f'Recall@{n}')
    plt.bar(x + width, ndcg_values, width, label=f'NDCG@{n}')
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Recommendation Quality Metrics')
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2. Diversity Metrics
    plt.subplot(2, 2, 2)
    
    unique_ratio = [metrics[model].get('unique_ratio', 0) for model in model_names]
    gini = [metrics[model].get('gini_coefficient', 0) for model in model_names]
    entropy = [metrics[model].get('genre_entropy', 0) / 5 for model in model_names]  # Normalized for scale
    
    plt.bar(x - width, unique_ratio, width, label='Unique Ratio')
    plt.bar(x, gini, width, label='Gini Coefficient')
    plt.bar(x + width, entropy, width, label='Genre Entropy/5')
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Recommendation Diversity Metrics')
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 3. Users with metrics
    plt.subplot(2, 2, 3)
    
    users_with_precision = [metrics[model].get('users_with_precision', 0) for model in model_names]
    users_with_recall = [metrics[model].get('users_with_recall', 0) for model in model_names]
    users_with_ndcg = [metrics[model].get('users_with_ndcg', 0) for model in model_names]
    
    plt.bar(x - width, users_with_precision, width, label='Users with Precision')
    plt.bar(x, users_with_recall, width, label='Users with Recall')
    plt.bar(x + width, users_with_ndcg, width, label='Users with NDCG')
    
    plt.xlabel('Models')
    plt.ylabel('Number of Users')
    plt.title('Users with Valid Metrics')
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('visualizations/recommendation_metrics.png', dpi=300)
    plt.close()

def display_example_recommendations(hybrid_recommender, user_id, movies, ratings):
    """
    Display example recommendations for a user.
    
    Parameters:
        hybrid_recommender (function): Hybrid recommendation function
        user_id (int): User ID to generate recommendations for
        movies (DataFrame): Movies data
        ratings (DataFrame): Ratings data
    """
    print(f"\n===== EXAMPLE RECOMMENDATIONS FOR USER {user_id} =====")
    
    # Step 1: Get the user's ratings
    user_ratings = ratings[ratings['user_id'] == user_id].sort_values('rating', ascending=False)
    
    if user_ratings.empty:
        print("This user has no ratings.")
        return
    
    # Step 2: Display user's top rated movies
    print("User's top rated movies:")
    for i, (_, row) in enumerate(user_ratings.head(3).iterrows()):
        movie_title = movies[movies['movie_id'] == row['movie_id']]['title'].values[0]
        print(f"{i+1}. {movie_title} - Rating: {row['rating']}")
    
    # Step 3: Get the user's favorite movie
    favorite_movie_id = user_ratings.iloc[0]['movie_id']
    favorite_movie = movies[movies['movie_id'] == favorite_movie_id]['title'].values[0]
    
    # Step 4: Generate hybrid recommendations
    print(f"\nRecommendations based on user's profile and favorite movie ({favorite_movie}):")
    hybrid_recs = hybrid_recommender(user_id, favorite_movie_id)
    
    # Step 5: Display recommendations
    for i, (_, row) in enumerate(hybrid_recs.iterrows()):
        print(f"{i+1}. {row['title']} - {row['source']} (Score: {row['score']:.4f})")
    
    # Step 6: Generate some purely content-based recommendations
    print(f"\nMore movies similar to {favorite_movie}:")
    content_recs = hybrid_recommender(user_id, favorite_movie_id, n=5)
    content_recs = content_recs[content_recs['source'] == 'Content']
    
    for i, (_, row) in enumerate(content_recs.iterrows()):
        if i < 5:  # Limit to 5 recommendations
            print(f"{i+1}. {row['title']} (Score: {row['score']:.4f})")

# =================================================================
# MAIN EXECUTION
# =================================================================

def main():
    """Main execution function to run the entire recommendation system."""
    print("===== MOVIE RECOMMENDATION SYSTEM =====")
    
    # Step 1: Download and load the dataset
    download_dataset()
    movies, ratings, users = load_data()
    
    # Step 2: Analyze the data
    stats = analyze_data(movies, ratings, users)
    
    # Step 3: Engineer features
    movies_enhanced, ratings_enhanced, users_enhanced, movie_popularity = engineer_features(movies, ratings, users)
    
    # Step 4: Build content-based recommender
    tfidf_matrix, cosine_sim, indices = build_content_based_recommender(movies_enhanced)
    
    # Step 5: Prepare data for collaborative filtering
    data, trainset, testset = prepare_surprise_data(ratings)
    
    # Step 6: Evaluate collaborative filtering models
    model_results = evaluate_collaborative_models(data)
    
    # Step 7: Train the best models
    trained_models = train_best_models(trainset)
    
    # Step 8: Generate and evaluate Top-N recommendations
    metrics, recommendations = evaluate_top_n_recommendations(trained_models, testset, movies, n=10)
    
    # Step 9: Build hybrid recommender
    hybrid_recommender = build_hybrid_recommender(
        trained_models, cosine_sim, indices, movies, ratings, alpha=0.7)
    
    # Step 10: Generate example recommendations for a user
    example_user_id = 1  # You can change this to any user ID
    display_example_recommendations(hybrid_recommender, example_user_id, movies, ratings)
    
    # Step 11: Print summary of results
    print("\n===== RECOMMENDATION SYSTEM SUMMARY =====")
    print(f"Dataset: MovieLens 100k ({stats['user_count']} users, {stats['movie_count']} movies, {stats['rating_count']} ratings)")
    print(f"Average score: {stats['avg_rating']:.2f}")
    print(f"Matrix sparsity: {stats['sparsity']*100:.1f}%")
    print(f"Cold start: {stats['cold_start_movies_pct']:.1f}% of movies have less than 5 ratings")
    
    # Print best model results
    best_model = min(model_results, key=lambda x: model_results[x]['RMSE'])
    print(f"\nBest model: {best_model}")
    print(f"RMSE: {model_results[best_model]['RMSE']:.4f}")
    print(f"MAE: {model_results[best_model]['MAE']:.4f}")
    
    print("\nRecommendation system completed successfully!")

if __name__ == "__main__":
    main()
