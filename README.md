# MovieLens Recommendation System

## Overview

This project implements a comprehensive movie recommendation system using the MovieLens 100k dataset. The system employs multiple recommendation strategies including content-based filtering, collaborative filtering, and hybrid approaches to deliver personalized movie suggestions. It addresses common challenges in recommendation systems such as data sparsity, cold start problems, and recommendation diversity while providing detailed evaluation metrics.

## Dataset

The [MovieLens 100K Dataset](https://files.grouplens.org/datasets/movielens/ml-100k.zip) - 100,000 movie ratings from 943 users on 1,682 movies contains:
- 100,000 ratings (1-5) from 943 users on 1,682 movies
- Each user has rated at least 20 movies
- Simple demographic information for users (age, gender, occupation, zip code)
- Movie information including title, release date, and genres

Key dataset characteristics:
- Rating matrix sparsity: 93.7%
- Average rating: 3.53
- Cold start issues: 19.8% of movies have fewer than 5 ratings
- Most rated movie: Star Wars (1977) with 583 ratings
- Most active rating month: November 1997

## Features

### Data Exploration
- Comprehensive analysis of rating distributions and patterns
- User demographic analysis (gender, age, occupation)
- Temporal analysis of rating behavior
- Genre popularity and rating analysis
- Cold start and sparsity assessment

### Data Preprocessing
- Temporal feature extraction (day of week, hour of day)
- Movie metadata enhancement (year extraction, decade grouping)
- Genre consolidation for content-based filtering
- User feature engineering (demographic encoding)
- Rating normalization and interaction features

### Model Training
- Content-based filtering using TF-IDF vectorization and cosine similarity
- Collaborative filtering models with cross-validation
- Hyperparameter optimization for SVD
- Hybrid model combining multiple recommendation approaches

### Model Evaluation
- Rating prediction metrics (RMSE, MAE)
- Top-N recommendation evaluation (Precision, Recall, NDCG)
- Diversity metrics (genre entropy, unique recommendation ratio)
- User coverage analysis

### Visualization
- Rating distribution and patterns
- User demographic insights
- Movie genre distribution wordcloud
- Model performance comparisons
- Recommendation diversity analysis
- Temporal patterns in user activity

## Models Implemented

| Model Type | Algorithm | Description |
|------------|-----------|-------------|
| Content-Based | TF-IDF + Cosine Similarity | Uses movie attributes (genres, year, title) to find similar movies |
| Collaborative | SVD | Matrix factorization approach using Singular Value Decomposition |
| Collaborative | NMF | Non-negative Matrix Factorization |
| Collaborative | KNN User-Based | K-Nearest Neighbors based on user similarity |
| Collaborative | KNN Item-Based | K-Nearest Neighbors based on item similarity |
| Collaborative | Baseline | Simple baseline using global, user, and item biases |
| Collaborative | SlopeOne | Weighted slope-one algorithm |
| Hybrid | Weighted Combination | Combines content-based and collaborative filtering with configurable weights |

## Results

### Model Comparison

#### Rating Prediction Performance

| Model    | RMSE    | MAE     |
|----------|---------|---------|
| SVD      | 0.9364  | 0.7376  |
| NMF      | 0.9665  | 0.7598  |
| KNN_User | 1.0097  | 0.8018  |
| KNN_Item | 1.0360  | 0.8297  |
| Baseline | 0.9443  | 0.7485  |
| SlopeOne | 0.9441  | 0.7422  |

#### Top-N Recommendation Performance

| Model    | Precision@10 | Recall@10 | NDCG@10 | Unique Ratio | Genre Entropy |
|----------|--------------|-----------|---------|--------------|---------------|
| SVD      | 0.0776       | 0.0413    | 0.5851  | 0.2080       | 3.4705        |
| KNN_Item | 0.0020       | 0.0004    | 0.6309  | 0.1720       | 3.3115        |
| Baseline | 0.0755       | 0.0383    | 0.4932  | 0.0540       | 3.2913        |

### Best Model

SVD (Singular Value Decomposition) provides the best overall performance:
- Lowest RMSE: 0.9364
- Competitive MAE: 0.7376
- Highest Precision@10: 0.0776
- Highest Recall@10: 0.0413
- Good recommendation diversity: 3.4705 entropy score

### Feature Importance

In content-based filtering, the most important features for movie similarity are:
- Primary genres (Drama, Comedy, Action)
- Release decade
- Distinctive director/actor terms in titles
- Niche genre combinations

## Outcome

### Best Performing Model: Hybrid Recommender

The hybrid recommender combines SVD collaborative filtering (70% weight) with content-based filtering (30% weight) to achieve:
- Better cold-start handling for new movies
- Improved diversity in recommendations
- Higher user satisfaction through personalized suggestions
- Fallback to popularity-based recommendations when necessary

Example recommendations for User 1:
1. Dr. Strangelove (1963)
2. One Flew Over the Cuckoo's Nest (1975)
3. Rear Window (1954)
4. Schindler's List (1993)
5. Third Man, The (1949)

## Future Work

- **Deep Learning Integration**: Implement neural network-based recommendation models (NCF, AutoEncoders)
- **Additional Features**: Incorporate movie plots, actor networks, and director information
- **Temporal Dynamics**: Model user preference changes over time
- **Context-aware Recommendations**: Consider situational factors (time of day, weekday/weekend)
- **Explanation Framework**: Develop better recommendation explanations for users
- **Online Learning**: Implement incremental model updates for new ratings
- **Multi-criteria Ratings**: Extend the system to handle ratings on multiple aspects (acting, plot, visuals)
- **A/B Testing Framework**: Develop tools for online evaluation with real users

## Notes

- The system is designed to be modular, allowing for easy component replacement or enhancement
- All models and visualizations are automatically saved for future reference
- Dataset sparsity (93.7%) remains a challenge for accurate recommendations
- Cold start handling is particularly important as ~20% of movies have few ratings

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## License

This project is licensed under the MIT License - see the LICENSE file for details.


