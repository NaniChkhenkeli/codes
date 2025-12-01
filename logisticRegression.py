import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, mean_absolute_error, mean_squared_error

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegressionManual:
    def __init__(self, learning_rate=0.01, n_iterations=10000, verbose=False):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = sigmoid(linear_model)

            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if self.verbose and i % 1000 == 0:
                loss = -np.mean(y*np.log(y_predicted + 1e-15) + (1-y)*np.log(1 - y_predicted + 1e-15))
                print(f"Iteration {i}, Loss: {loss:.4f}")

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def r_squared(self, X, y):
        # McFadden's pseudo-R²
        y_pred = self.predict_proba(X)
        ll_model = np.sum(y*np.log(y_pred + 1e-15) + (1-y)*np.log(1 - y_pred + 1e-15))
        y_mean = np.mean(y)
        ll_null = np.sum(y*np.log(y_mean + 1e-15) + (1 - y)*np.log(1 - y_mean + 1e-15))
        return 1 - (ll_model / ll_null)

def main():
    csv_file = '30movies_dataset.csv'
    df = pd.read_csv(csv_file).dropna()
    mean_revenue = df['Gross Revenue (million)'].mean()
    df['Success'] = (df['Gross Revenue (million)'] >= mean_revenue).astype(int)
    print(f"Mean Revenue Threshold for Success: ${mean_revenue:.2f}")

    categorical_cols = ['Genre', 'Director', 'Lead Actor', 'Production Company', 
                        'Country of Origin', 'Original Language']
    mappings = {}
    for col in categorical_cols:
        df[col+'_id'] = df[col].astype('category').cat.codes
        mappings[col] = dict(enumerate(df[col].astype('category').cat.categories))

    feature_cols = ['Year', 'Runtime (min)'] + [col+'_id' for col in categorical_cols]
    X = df[feature_cols].values
    y = df['Success'].values

    # Normalize numeric columns (Year, Runtime)
    X[:, 0] = (X[:, 0] - X[:, 0].mean()) / (X[:, 0].std() + 1e-8)
    X[:, 1] = (X[:, 1] - X[:, 1].mean()) / (X[:, 1].std() + 1e-8)

    # Train-test split (80%-20%)
    np.random.seed(42)
    mask = np.random.rand(len(X)) < 0.8
    X_train, X_test = X[mask], X[~mask]
    y_train, y_test = y[mask], y[~mask]

    model = LogisticRegressionManual(learning_rate=0.01, n_iterations=10000, verbose=True)
    model.fit(X_train, y_train)
    print("\nModel Training Complete!")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # --- classification metrics ---
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)

    # --- regression-style metrics ---
    mae = mean_absolute_error(y_test, y_proba)
    mse = mean_squared_error(y_test, y_proba)
    rmse = np.sqrt(mse)
    r2 = model.r_squared(X_test, y_test)

    print("\n--- Classification Metrics ---")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Accuracy: {accuracy:.2f}")

    print("\n--- Regression-style Metrics ---")
    print(f"Mean Absolute Error (MAE): {mae:.3f}")
    print(f"Mean Squared Error (MSE): {mse:.3f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
    print(f"Pseudo R²: {r2:.4f}")

    print("\n--- Summary ---")
    print(f"Total movies analyzed: {len(df)}")
    print(f"Number of genres: {df['Genre'].nunique()}")

    df_test = df.iloc[~mask].copy()
    df_test['Predicted_Success'] = y_pred
    df_test['Success_Probability'] = y_proba
    df_test['Correct_Prediction'] = (df_test['Predicted_Success'] == df_test['Success']).astype(int)

    for col in categorical_cols:
        df_test[col] = df_test[col+'_id'].map({v:k for k,v in mappings[col].items()})

    # --- Plot: Success Rate by Genre ---
    genre_success = df_test.groupby('Genre')['Predicted_Success'].mean()
    if not genre_success.empty:
        plt.figure(figsize=(10,5))
        genre_success.plot(kind='bar', title='Predicted Success Rate by Genre')
        plt.ylabel('Predicted Success Rate')
        plt.tight_layout()
        plt.savefig('success_rate_by_genre.png')
        plt.close()
        print("Saved plot: success_rate_by_genre.png")

    # --- Plot: Predicted vs Actual ---
    plt.figure(figsize=(8,5))
    plt.scatter(range(len(y_test)), y_test, label='Actual', alpha=0.7)
    plt.scatter(range(len(y_test)), y_proba, label='Predicted Prob', alpha=0.7)
    plt.xlabel('Test Sample Index')
    plt.ylabel('Success / Probability')
    plt.title('Predicted vs Actual Success')
    plt.legend()
    plt.tight_layout()
    plt.savefig('predicted_vs_actual_points.png')
    plt.close()
    print("Saved plot: predicted_vs_actual_points.png")

    df_test.to_csv('movie_success_predictions_logistic.csv', index=False)
    print("\nPredictions saved to 'movie_success_predictions_logistic.csv'")

if __name__ == "__main__":
    main()
