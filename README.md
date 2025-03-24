# Smart_Recommendation_System
Personalized food recommendation model
## Overview
This project builds a Smart Food Recommendation System that provides personalized recommendations based on content-based filtering, collaborative filtering, and contextual factors like location, weather, and user sentiment.

The system utilizes:
- Content-Based Filtering: To recommend recipes based on their ingredients.
- Collaborative Filtering: To predict ratings and suggest recipes based on user preferences.
- Context-Aware Recommendations: To personalize suggestions by incorporating factors like location, weather, and user sentiment.
- Sentiment Analysis: To refine recommendations based on the user's mood or feedback.

## Approach & Methodology
Content-Based Filtering:
- Used TF-IDF vectorization on ingredients (recipes) to generate cosine similarity scores.
- The system recommends similar recipes based on the cosine similarity of their ingredients.
Collaborative Filtering:
- Matrix Factorization using SVD (Singular Value Decomposition) from the Surprise library.
- The model predicts ratings for unrated recipes, helping to recommend items based on user preferences.
Context-Aware Recommendations:
- Generated synthetic contextual data (location and weather) and incorporated them in the recommendation process.
- Personalized recommendations based on a combination of user history and context.
Sentiment Analysis:
- Used TextBlob for sentiment analysis on user feedback.
- Adjusted recommendations based on the sentiment of user-provided feedback, i.e., mood-based refinement.

## Data Preprocessing & Selection
- Datasets:
  - Recipes Dataset:
    - Dataset containing recipe details, including id, name, and ingredients of books.
    - We used the GoodBooks-10k dataset to model books as recipes for this PoC.
    - Link: GoodBooks-10k Books Dataset
      
  - Ratings Dataset:
    - Contains ratings for each recipe from different users, with fields user_id, recipe_id, and rating.
    - Link: GoodBooks-10k Ratings Dataset

- Preprocessing Steps:
  - Handling Missing Data:
    - Removed rows with missing ingredients in the recipes dataset.
    - Removed rows with missing rating values in the ratings dataset.
  - Text Preprocessing:
    - Ingredients were tokenized, and stopwords were removed during the TF-IDF vectorization.
  - Data Normalization:
    - Ratings were normalized between 1 and 5 using the Surprise library’s Reader class.

## Model Architecture & Tuning Process
#### Content-Based Filtering:
- TF-IDF Vectorizer was used to transform the recipe ingredients into numerical vectors. Cosine similarity was then calculated to identify the most similar recipes based on these vectors.
#### Collaborative Filtering:
- Used SVD (Singular Value Decomposition) model from the Surprise library to predict ratings.
Train-Test Split: The ratings data was split into training (80%) and testing (20%) datasets using the train_test_split method.
#### Sentiment Analysis:
- TextBlob was used to perform sentiment analysis on user feedback. Sentiments were categorized as Positive, Negative, or Neutral.
#### Context-Aware Recommendation:
- Generated synthetic location and weather data for the user.
- Combined the user’s historical ratings with contextual information (location and weather) for generating personalized recommendations.
#### Hyperparameter Tuning:
- For the SVD model, you can tune parameters like the number of factors (n_factors) or regularization terms to improve prediction accuracy.
- GridSearchCV could be used to find the optimal hyperparameters.

## Performance Results & Next Steps
#### Performance Evaluation:
- Content-Based Filtering: Evaluated using cosine similarity. It provided reliable recommendations for recipes with similar ingredients.
- Collaborative Filtering: The SVD model was trained, and performance was evaluated using the RMSE (Root Mean Squared Error) on the test data.
- Context-Aware Recommendations: Personalization through synthetic contextual data improved the relevance of suggestions, showing good potential for future applications.
- Sentiment Analysis: Positive, negative, and neutral feedback was used to adjust the recommendation list. This enhanced user satisfaction by tailoring the recommendations according to mood.

#### Next Steps:
- Add Real-Time Data: Implement APIs for real-time weather and location data (e.g., using a weather API).
- Improve Sentiment Analysis: Enhance the sentiment analysis model by training on a domain-specific corpus.
- Deploy the Model: Deploy the recommendation system using a Flask or FastAPI web application with a simple UI.
- Optimization: Further fine-tune the SVD and TF-IDF models, possibly by implementing matrix factorization methods like ALS (Alternating Least Squares) or k-NN for better performance.
