import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import os

class NaiveBayes:
    def __init__(self):
        self.class_priors = {}
        self.feature_probs = {}
        self.classes = None
    
    def fit(self, X, y):
        """
        Parameters:
        X: np.ndarray - Features (n_samples, n_features)
        y: np.ndarray - Target labels (n_samples,)
        """
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        
        # class priors P(y)
        for cls in self.classes:
            X_c = X[y == cls]
            self.class_priors[cls] = X_c.shape[0] / n_samples
        
        # likelihood P(x|y) for each feature
        for cls in self.classes:
            X_c = X[y == cls]
            self.feature_probs[cls] = {}
            
            for feature_idx in range(n_features):
                feature_values = np.unique(X[:, feature_idx])
                self.feature_probs[cls][feature_idx] = {}
                
                for value in feature_values:
                    # P(x|y) = Count of feature value in class / Total samples in class
                    count = np.sum(X_c[:, feature_idx] == value)
                    total = X_c.shape[0]
                    self.feature_probs[cls][feature_idx][value] = (count + 1) / (total + len(feature_values))
    
    def predict_proba(self, X):
        """
        Predict probabilities for each class.
        
        Parameters:
        X: np.ndarray - Features (n_samples, n_features)
        
        Returns:
        np.ndarray - Predicted probabilities for each class (n_samples, n_classes)
        """
        n_samples, n_features = X.shape
        probabilities = np.zeros((n_samples, len(self.classes)))
        
        for idx, sample in enumerate(X):
            for cls_idx, cls in enumerate(self.classes):
                # Start with the log prior probability P(y)
                log_prob = np.log(self.class_priors[cls])
                
                # Add log likelihoods P(x|y) for each feature
                for feature_idx in range(n_features):
                    feature_value = sample[feature_idx]
                    feature_probs = self.feature_probs[cls][feature_idx]
                    
                    # Laplace smoothing for unseen values
                    likelihood = feature_probs.get(feature_value, 1e-6)
                    log_prob += np.log(likelihood)
                
                probabilities[idx, cls_idx] = log_prob
        
        probabilities = np.exp(probabilities)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        
        return probabilities
    
    def predict(self, X):
        """
        Predict the class labels.
        
        Parameters:
        X: np.ndarray - Features (n_samples, n_features)
        
        Returns:
        np.ndarray - Predicted class labels (n_samples,)
        """
        probabilities = self.predict_proba(X)
        return self.classes[np.argmax(probabilities, axis=1)]

os.makedirs("naive_results", exist_ok=True) 

print("="*70)
print("MOVIE SUCCESS PREDICTION USING NAIVE BAYES")
print("="*70)

csv_file = '30movies_dataset.csv' 
print(f"\nLoading data from: {csv_file}")
df = pd.read_csv(csv_file)

print(f"Dataset loaded successfully!")

# VOCABULARY SIZE 
# Convert all titles to lowercase, split into words, collect into a set, because i need to count unique words, not to repeat
vocab = set()
for title in df['Title']:
    words = title.lower().split()
    vocab.update(words)

vocab_size = len(vocab)

print("\n" + "="*70)
print("VOCABULARY STATISTICS (UNIQUE WORDS IN TITLES)")
print("="*70)
print(f"Vocabulary size: {vocab_size}")
print(f"Sample words: {list(vocab)[:20]} ...")


print(f"Total records: {len(df)}")
print(f"\nColumns: {list(df.columns)}")

# Calculate success based on mean revenue
mean_revenue = df['Gross Revenue (million)'].mean()
df['Success'] = (df['Gross Revenue (million)'] >= mean_revenue).astype(int)

# Remove rows with missing values
df_original_size = len(df)
df = df.dropna()
print(f"\nRows after removing missing values: {len(df)} (removed {df_original_size - len(df)})")

print("\n" + "="*70)
print("DATASET STATISTICS")
print("="*70)
print(f"Total movies: {len(df)}")
print(f"Successful movies (revenue >= mean): {df['Success'].sum()} ({df['Success'].sum()/len(df)*100:.1f}%)")
print(f"Unsuccessful movies (revenue < mean): {len(df) - df['Success'].sum()} ({(len(df) - df['Success'].sum())/len(df)*100:.1f}%)")
print(f"Mean revenue: ${mean_revenue:,.2f}")

# Encode categorical features - Create label encoders
print("\n" + "="*70)
print("ENCODING CATEGORICAL FEATURES")
print("="*70)

label_encoder_genre = df[['Genre']].drop_duplicates().reset_index(drop=True)
label_encoder_genre['GenreId'] = label_encoder_genre.index
genre_to_id = dict(zip(label_encoder_genre['Genre'], label_encoder_genre['GenreId']))
print(f"Unique Genres: {len(genre_to_id)}")

label_encoder_director = df[['Director']].drop_duplicates().reset_index(drop=True)
label_encoder_director['DirectorId'] = label_encoder_director.index
director_to_id = dict(zip(label_encoder_director['Director'], label_encoder_director['DirectorId']))
print(f"Unique Directors: {len(director_to_id)}")

label_encoder_actor = df[['Lead Actor']].drop_duplicates().reset_index(drop=True)
label_encoder_actor['ActorId'] = label_encoder_actor.index
actor_to_id = dict(zip(label_encoder_actor['Lead Actor'], label_encoder_actor['ActorId']))
print(f"Unique Lead Actors: {len(actor_to_id)}")

label_encoder_production = df[['Production Company']].drop_duplicates().reset_index(drop=True)
label_encoder_production['ProductionId'] = label_encoder_production.index
production_to_id = dict(zip(label_encoder_production['Production Company'], label_encoder_production['ProductionId']))
print(f"Unique Production Companies: {len(production_to_id)}")

label_encoder_country = df[['Country of Origin']].drop_duplicates().reset_index(drop=True)
label_encoder_country['CountryId'] = label_encoder_country.index
country_to_id = dict(zip(label_encoder_country['Country of Origin'], label_encoder_country['CountryId']))
print(f"Unique Countries: {len(country_to_id)}")

label_encoder_language = df[['Original Language']].drop_duplicates().reset_index(drop=True)
label_encoder_language['LanguageId'] = label_encoder_language.index
language_to_id = dict(zip(label_encoder_language['Original Language'], label_encoder_language['LanguageId']))
print(f"Unique Languages: {len(language_to_id)}")

# Apply encoding to dataframe
df['GenreId'] = df['Genre'].map(genre_to_id)
df['DirectorId'] = df['Director'].map(director_to_id)
df['ActorId'] = df['Lead Actor'].map(actor_to_id)
df['ProductionId'] = df['Production Company'].map(production_to_id)
df['CountryId'] = df['Country of Origin'].map(country_to_id)
df['LanguageId'] = df['Original Language'].map(language_to_id)

# Save processed data
df.to_csv('naive_results/processed_movies_data.csv', index=False)
print("\nProcessed data saved to: 'processed_movies_data.csv'")

# Verify the DataFrame
print("\n" + "="*70)
print("DATA PREVIEW")
print("="*70)
print(df[['Title', 'Year', 'Genre', 'Director', 'Success', 'Gross Revenue (million)']].head(10))

feature_columns = ['Year', 'GenreId', 'DirectorId', 'ActorId', 'ProductionId', 
                   'Runtime (min)', 'CountryId', 'LanguageId']
X = df[feature_columns].values
y = df['Success'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n" + "="*70)
print("TRAIN-TEST SPLIT")
print("="*70)
print(f"Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Testing samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
print(f"Number of features: {X_train.shape[1]}")

# Train Naive Bayes model
print("\n" + "="*70)
print("TRAINING NAIVE BAYES MODEL")
print("="*70)
model = NaiveBayes()
model.fit(X_train, y_train)
print("Training completed!")

# Predictions and probabilities
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)
print("Predictions generated!")

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\n" + "="*70)
print("MODEL PERFORMANCE METRICS")
print("="*70)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\n" + "="*70)
print("CONFUSION MATRIX")
print("="*70)
print(conf_matrix)
print("\nInterpretation:")
print(f"True Negatives (Correctly predicted unsuccessful): {conf_matrix[0][0]}")
print(f"False Positives (Incorrectly predicted successful): {conf_matrix[0][1]}")
print(f"False Negatives (Incorrectly predicted unsuccessful): {conf_matrix[1][0]}")
print(f"True Positives (Correctly predicted successful): {conf_matrix[1][1]}")

# Classification Report
print("\n" + "="*70)
print("CLASSIFICATION REPORT")
print("="*70)
class_report = classification_report(y_test, y_pred, target_names=['Not Successful', 'Successful'])
print(class_report)

# Create reverse mappings for decoding
id_to_genre = {v: k for k, v in genre_to_id.items()}
id_to_director = {v: k for k, v in director_to_id.items()}
id_to_actor = {v: k for k, v in actor_to_id.items()}
id_to_production = {v: k for k, v in production_to_id.items()}
id_to_country = {v: k for k, v in country_to_id.items()}
id_to_language = {v: k for k, v in language_to_id.items()}

# Preparing the test DataFrame for detailed analysis
df_test = pd.DataFrame(X_test, columns=feature_columns)
df_test['success_probability'] = y_pred_proba[:, 1] 
df_test['predicted_success'] = y_pred
df_test['actual_success'] = y_test

# Map IDs back to names
df_test['Genre'] = df_test['GenreId'].map(id_to_genre)
df_test['Director'] = df_test['DirectorId'].map(id_to_director)
df_test['Lead Actor'] = df_test['ActorId'].map(id_to_actor)
df_test['Production Company'] = df_test['ProductionId'].map(id_to_production)
df_test['Country'] = df_test['CountryId'].map(id_to_country)
df_test['Language'] = df_test['LanguageId'].map(id_to_language)

# Create final results dataframe
results_df = df_test[['Year', 'Genre', 'Director', 'Lead Actor', 'Production Company', 
                      'Runtime (min)', 'Country', 'Language', 'success_probability', 
                      'predicted_success', 'actual_success']].copy()

# Add prediction correctness column
results_df['correct_prediction'] = (results_df['predicted_success'] == results_df['actual_success']).astype(int)
results_df.to_csv('naive_results/movie_success_predictions_naive_bayes.csv', index=False)

print("\n" + "="*70)
print("RESULTS SAVED")
print("="*70)
print("All predictions saved to: 'movie_success_predictions_naive_bayes.csv'")

print("\n" + "="*70)
print("PREDICTION ANALYSIS")
print("="*70)

successful_predicted = results_df[results_df['predicted_success'] == 1]
successful_actual = results_df[results_df['actual_success'] == 1]
correct_predictions = results_df[results_df['correct_prediction'] == 1]

print(f"Total test samples: {len(results_df)}")
print(f"Predicted Successful: {len(successful_predicted)}")
print(f"Actually Successful: {len(successful_actual)}")
print(f"Correct Predictions: {len(correct_predictions)} ({len(correct_predictions)/len(results_df)*100:.1f}%)")
print(f"Incorrect Predictions: {len(results_df) - len(correct_predictions)} ({(len(results_df) - len(correct_predictions))/len(results_df)*100:.1f}%)")

# Correctly predicted successful movies
correct_successful = results_df[(results_df['predicted_success'] == 1) & (results_df['actual_success'] == 1)]
print(f"Correctly Predicted as Successful: {len(correct_successful)}")

# Top 10 movies by success probability
print("\n" + "="*70)
print("TOP 10 MOVIES BY SUCCESS PROBABILITY")
print("="*70)
top_predictions = results_df.nlargest(10, 'success_probability')
print(top_predictions[['Year', 'Genre', 'Director', 'Lead Actor', 'success_probability', 
                       'predicted_success', 'actual_success', 'correct_prediction']].to_string())

# Save top predictions
top_predictions.to_csv('naive_results/top_10_success_predictions.csv', index=False)
print("\nTop 10 predictions saved to: 'top_10_success_predictions.csv'")

# Correct vs Incorrect predictions analysis
correct_pred_df = results_df[results_df['correct_prediction'] == 1]
incorrect_pred_df = results_df[results_df['correct_prediction'] == 0]

print("\n" + "="*70)
print("PROBABILITY ANALYSIS")
print("="*70)
if len(correct_pred_df) > 0:
    print(f"Average probability for CORRECT predictions: {correct_pred_df['success_probability'].mean():.4f}")
else:
    print("No correct predictions found")
    
if len(incorrect_pred_df) > 0:
    print(f"Average probability for INCORRECT predictions: {incorrect_pred_df['success_probability'].mean():.4f}")
else:
    print("No incorrect predictions found")

correct_pred_df.to_csv('naive_results/correct_predictions.csv', index=False)
incorrect_pred_df.to_csv('naive_results/incorrect_predictions.csv', index=False)
print("\nCorrect predictions saved to: 'correct_predictions.csv'")
print("Incorrect predictions saved to: 'incorrect_predictions.csv'")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nGenerated files:")
print("1. processed_movies_data.csv - Processed dataset with encodings")
print("2. movie_success_predictions_naive_bayes.csv - All predictions")
print("3. top_10_success_predictions.csv - Top 10 most likely successful movies")
print("4. correct_predictions.csv - Correctly predicted movies")
print("5. incorrect_predictions.csv - Incorrectly predicted movies")

print("="*70)