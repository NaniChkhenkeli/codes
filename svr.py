import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import os

def train_model_and_predict():
    csv_file = '90movies_dataset.csv'
    df = pd.read_csv(csv_file)
    
    categorical_columns = ['Genre', 'Director', 'Lead Actor', 'Production Company', 
                           'Country of Origin', 'Original Language']
    
    feature_columns_base = ['Year', 'Runtime (min)'] + categorical_columns
    X_base = df[feature_columns_base].copy()
    y = df['Gross Revenue (million)'].values
    
    # first splitting and then calculating mean revenue only on training data
    X_train_base, X_test_base, y_train, y_test, train_idx, test_idx = train_test_split(
        X_base, y, df.index, test_size=0.2, random_state=42
    )
    
    mean_revenue = y_train.mean()
    print(f"\nMean Revenue Threshold (from training data only): ${mean_revenue:,.2f}")
    
    # encoding categorical variables to fit on train and transform on test
    label_encoders = {}
    X_train_encoded = pd.DataFrame()
    X_test_encoded = pd.DataFrame()
    
    X_train_encoded['Year'] = X_train_base['Year'].values
    X_train_encoded['Runtime (min)'] = X_train_base['Runtime (min)'].values
    X_test_encoded['Year'] = X_test_base['Year'].values
    X_test_encoded['Runtime (min)'] = X_test_base['Runtime (min)'].values
    
    for col in categorical_columns:
        le = LabelEncoder()
        X_train_encoded[f'{col}_encoded'] = le.fit_transform(X_train_base[col].astype(str))
        
        # Handle unseen categories in test set
        test_values = X_test_base[col].astype(str).values
        encoded_test = []
        for val in test_values:
            if val in le.classes_:
                encoded_test.append(le.transform([val])[0])
            else:
                encoded_test.append(0)
        
        X_test_encoded[f'{col}_encoded'] = encoded_test
        label_encoders[col] = le
    
    # creating director average revenue feature only from training data
    train_df = X_train_encoded.copy()
    train_df['revenue'] = y_train
    train_df['Director_encoded'] = X_train_encoded['Director_encoded']
    train_df['Year'] = X_train_encoded['Year']
    
    train_df = train_df.sort_values(by=['Director_encoded', 'Year']).reset_index(drop=True)
    
    train_df['director_avg_revenue'] = (
        train_df.groupby('Director_encoded')['revenue']
        .transform(lambda x: x.expanding().mean())
    )
    
    director_stats = train_df.groupby('Director_encoded')['revenue'].mean().to_dict()
    X_test_encoded['director_avg_revenue'] = X_test_encoded['Director_encoded'].map(director_stats).fillna(mean_revenue)
    
    # adding director_avg_revenue back to training features
    X_train_encoded['director_avg_revenue'] = train_df['director_avg_revenue'].values
    
    # defining final feature columns
    feature_columns = ['Year', 'Runtime (min)', 'Genre_encoded', 'Director_encoded', 
                       'Lead Actor_encoded', 'Production Company_encoded', 
                       'Country of Origin_encoded', 'Original Language_encoded', 
                       'director_avg_revenue']
    
    X_train = X_train_encoded[feature_columns].values
    X_test = X_test_encoded[feature_columns].values
    
    # training model
    linear_svr_model = LinearSVR(C=1.0, epsilon=0.5, max_iter=5000, random_state=42)
    linear_svr_model.fit(X_train, y_train)
    
    # Predictions on test data
    y_pred_test = linear_svr_model.predict(X_test)
    y_pred_test = np.maximum(0, y_pred_test)  
    
    df_test = X_test_encoded.copy()
    df_test['predicted_revenue'] = y_pred_test
    df_test['actual_revenue'] = y_test
    
    # decoding categorical features for readability
    for col in categorical_columns:
        encoded_col = f'{col}_encoded'
        df_test[col] = df_test[encoded_col].apply(
            lambda x: label_encoders[col].classes_[int(x)] if int(x) < len(label_encoders[col].classes_) else 'Unknown'
        )
    
    # Calculate predicted and actual success
    df_test['predicted_success'] = (df_test['predicted_revenue'] >= mean_revenue).astype(int)
    df_test['actual_success'] = (df_test['actual_revenue'] >= mean_revenue).astype(int)
    
    output_columns = ['Year', 'Genre', 'Director', 'Lead Actor', 'Production Company', 
                      'Country of Origin', 'Original Language', 'Runtime (min)',
                      'predicted_revenue', 'actual_revenue', 
                      'predicted_success', 'actual_success']
    
    df_output = df_test[output_columns].copy()
    df_output = df_output.sort_values(by=['Year', 'Director']).drop_duplicates()
    df_output.to_csv('movie_revenue_predictions.csv', index=False)
    
    print("\nModel Training Complete!")
    print(f"Predicted and Actual Revenue with Success Labels saved to 'movie_revenue_predictions.csv'")
    print("\nSample predictions:")
    print(df_output.head(10))
    
    print("\nTest Set Statistics")
    print(f"Total movies in test set: {len(df_test)}")
    print(f"Predicted successes: {df_test['predicted_success'].sum()}")
    print(f"Actual successes: {df_test['actual_success'].sum()}")
    print(f"Correct predictions: {(df_test['predicted_success'] == df_test['actual_success']).sum()}")
    print(f"Test accuracy: {(df_test['predicted_success'] == df_test['actual_success']).mean():.2%}")
    
    return mean_revenue

def analyze_and_visualize():
    csv_file = "movie_revenue_predictions.csv"
    df = pd.read_csv(csv_file)
    df["predicted_revenue"] = df["predicted_revenue"].apply(lambda x: int(x))

    precision = precision_score(df["actual_success"], df["predicted_success"], zero_division=0)
    recall = recall_score(df["actual_success"], df["predicted_success"], zero_division=0)
    f1 = f1_score(df["actual_success"], df["predicted_success"], zero_division=0)
    accuracy = accuracy_score(df["actual_success"], df["predicted_success"])

    print(f"\nClassification Metrics")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Accuracy: {accuracy:.2f}")

    mae = mean_absolute_error(df["actual_revenue"], df["predicted_revenue"])
    mse = mean_squared_error(df["actual_revenue"], df["predicted_revenue"])
    rmse = np.sqrt(mse)
    print(f"\nRegression Metrics")
    print(f"MAE: ${mae:,.2f}")
    print(f"MSE: ${mse:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")

    os.makedirs("visuals_movies", exist_ok=True)

    # 1.confusion matrix for success
    cm = confusion_matrix(df["actual_success"], df["predicted_success"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Not Success", "Success"]).plot(cmap="Blues", ax=ax)
    plt.title("Confusion Matrix - Success Prediction")
    plt.tight_layout()
    plt.savefig("visuals_movies/confusion_matrix.png", dpi=300)
    plt.close()
    print("Saved: visuals_movies/confusion_matrix.png")

    # 2.scatter plot for predicted vs actual revenue
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="actual_revenue", y="predicted_revenue", data=df)
    plt.plot([df["actual_revenue"].min(), df["actual_revenue"].max()],
             [df["actual_revenue"].min(), df["actual_revenue"].max()],
             color='red', linestyle='--', label="Perfect Prediction")
    plt.xlabel("Actual Revenue ($)")
    plt.ylabel("Predicted Revenue ($)")
    plt.title("Predicted vs Actual Revenue")
    plt.legend()
    plt.tight_layout()
    plt.savefig("visuals_movies/predicted_vs_actual.png", dpi=300)
    plt.close()
    print("Saved: visuals_movies/predicted_vs_actual.png")

    # 3.histogram of prediction errors
    df["error"] = df["predicted_revenue"] - df["actual_revenue"]
    plt.figure(figsize=(8, 6))
    sns.histplot(df["error"], bins=15, kde=True, color="orange")
    plt.xlabel("Prediction Error ($)")
    plt.title("Distribution of Prediction Errors")
    plt.tight_layout()
    plt.savefig("visuals_movies/error_distribution.png", dpi=300)
    plt.close()
    print("Saved: visuals_movies/error_distribution.png")

    # 4.MAE and MSE by Genre
    genre_stats = df.groupby("Genre").agg(
        mae_per_genre=("error", lambda x: np.mean(np.abs(x))),
        mse_per_genre=("error", lambda x: np.mean(x**2)),
        movie_count=("Genre", "count")
    ).reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Genre", y="mae_per_genre", data=genre_stats, palette="viridis")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean Absolute Error ($)")
    plt.title("MAE by Genre")
    plt.tight_layout()
    plt.savefig("visuals_movies/mae_by_genre.png", dpi=300)
    plt.close()
    print("Saved: visuals_movies/mae_by_genre.png")

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Genre", y="mse_per_genre", data=genre_stats, palette="magma")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean Squared Error ($)")
    plt.title("MSE by Genre")
    plt.tight_layout()
    plt.savefig("visuals_movies/mse_by_genre.png", dpi=300)
    plt.close()
    print("Saved: visuals_movies/mse_by_genre.png")

    # 5.success rate by Genre
    genre_success = df.groupby("Genre").agg(
        actual_success_rate=("actual_success", "mean"),
        predicted_success_rate=("predicted_success", "mean"),
        movie_count=("Genre", "count")
    ).reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=genre_success.melt(id_vars="Genre",
                                value_vars=["actual_success_rate", "predicted_success_rate"],
                                var_name="Type", value_name="Success Rate"),
        x="Genre", y="Success Rate", hue="Type"
    )
    plt.xticks(rotation=45, ha="right")
    plt.title("Success Rate by Genre")
    plt.tight_layout()
    plt.savefig("visuals_movies/success_rate_by_genre.png", dpi=300)
    plt.close()
    print("Saved: visuals_movies/success_rate_by_genre.png")

    print("\n--- Summary ---")
    print(f"All visualizations saved in 'visuals_movies/'")
    print(f"Total movies analyzed: {len(df)}")
    print(f"Number of genres: {len(genre_stats)}")


def main():
    print("=" * 60)
    print("MOVIE REVENUE PREDICTION AND ANALYSIS SYSTEM")
    print("(Fixed Version - No Data Leakage)")
    print("=" * 60)
    
    # Step 1: training model and make predictions
    print("\n" + "=" * 60)
    print("STEP 1: TRAINING MODEL AND MAKING PREDICTIONS")
    print("=" * 60)
    mean_revenue = train_model_and_predict()
    
    # Step 2: analyzing results with visualizations
    print("\n" + "=" * 60)
    print("STEP 2: ANALYZING RESULTS AND CREATING VISUALIZATIONS")
    print("=" * 60)
    analyze_and_visualize()
    
    print("\n" + "=" * 60)
    print("PROCESS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()