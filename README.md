# Movie Recommendation System

## Models Implemented
- **Collaborative Filtering**: SVD, NMF, KNNBasic
- **Content-Based Filtering**: TF-IDF genre analysis
- **Hybrid Approach**: Weighted combination of CF and CBF

## Features
### Data Exploration
- **Dataset**: MovieLens 100k (users, items, ratings)
- **Metadata**: Movie genres, release years

### Data Preprocessing
- **Cleaning**: Removed redundant columns (timestamps, URLs)
- **Vectorization**: TF-IDF for genre-based content analysis
- **Splitting**: 80% train / 20% test split

### Model Training
- **Collaborative**: Surprise library with custom hyperparameters
- **Content-Based**: Cosine similarity between TF-IDF vectors
- **Hybrid**: Alpha-weighted combination of predictions

### Model Evaluation
- **Metrics**: RMSE, MAE
- **Validation**: Cross-validation (3 folds)
- **Comparison**: Performance benchmarking across algorithms

## Results
### Model Comparison
| Model   | RMSE   | MAE    |
|---------|--------|--------|
| SVD     | 0.9452 | 0.7459 |
| NMF     | 0.9715 | 0.7629 |
| KNN     | 1.0209 | 0.8080 |

### Best Model
**SVD** demonstrated superior performance with:
- **RMSE**: 0.9452
- **MAE**: 0.7459

### Feature Importance
- **Collaborative Filtering**: Dominant predictor (alpha=0.7)
- **Genre Analysis**: Secondary signal for hybrid approach

## Outcome
**Best Performing Model**: SVD-based collaborative filtering with hybrid enhancement

## Future Work
1. **Cold Start Handling**: Improve new user/item recommendations
2. **Deep Learning**: Implement neural collaborative filtering
3. **Real-Time Deployment**: Containerize with API endpoints

## Notes
- **Dataset**: Preprocessed MovieLens 100k
- **Serialization**: Models saved via joblib
- **Hybrid Weight**: Alpha=0.7 balances CF/CBF

## Contributing
1. Fork repository
2. Implement new algorithms
3. Submit PR with documentation

## License
MIT License
