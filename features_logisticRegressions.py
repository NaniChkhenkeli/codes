import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix, hstack
from logisticRegression import LogisticRegressionManual  
import os

# Create output folder
os.makedirs("logisticRegfeatures", exist_ok=True)

def discretize_train_test(train_col, test_col, n_bins=5):
    train_col = np.asarray(train_col, dtype=float)
    test_col = np.asarray(test_col, dtype=float)
    if np.unique(train_col).size <= 1:
        return train_col.astype(int), test_col.astype(int)
    bins = np.unique(np.quantile(train_col, np.linspace(0, 1, n_bins + 1)))
    if len(bins) <= 1:
        return train_col.astype(int), test_col.astype(int)
    train_disc = np.digitize(train_col, bins, right=True)
    test_disc = np.digitize(test_col, bins, right=True)
    return train_disc.astype(int), test_disc.astype(int)

def ensure_int_matrix(arr):
    return np.asarray(arr).astype(int)

def main():

    df = pd.read_csv('90movies_dataset.csv').dropna()

    mean_revenue = df['Gross Revenue (million)'].mean()
    df['Success'] = (df['Gross Revenue (million)'] >= mean_revenue).astype(int)

    categorical_cols = [
        'Genre', 'Director', 'Lead Actor', 'Production Company',
        'Country of Origin', 'Original Language'
    ]

    cat_mappings = {}
    for col in categorical_cols:
        df[col + '_id'] = df[col].astype('category').cat.codes
        cat_mappings[col] = dict(enumerate(df[col].astype('category').cat.categories))

    feature_groups = {
        'Year': ['Year'],
        'Runtime': ['Runtime (min)'],
        'Genre': ['Genre_id'],
        'Director': ['Director_id'],
        'LeadActor': ['Lead Actor_id'],
        'Production': ['Production Company_id'],
        'Country': ['Country of Origin_id'],
        'Language': ['Original Language_id']
    }

    keys = list(feature_groups.keys())
    combos = []
    for r in (1, 2, 3):
        combos.extend(itertools.combinations(keys, r))
    combos.append(tuple(keys))

    np.random.seed(42)
    mask = np.random.rand(len(df)) < 0.8
    train_idx, test_idx = df.index[mask], df.index[~mask]

    results = []

    # Evaluate all feature combinations
    for combo in combos:
        selected_cols = [col for key in combo for col in feature_groups[key]]
        X_train = df.loc[train_idx, selected_cols].copy().values
        X_test = df.loc[test_idx, selected_cols].copy().values

        # discretize numeric features
        for i, col in enumerate(selected_cols):
            if col in ['Year', 'Runtime (min)']:
                tr, te = discretize_train_test(X_train[:, i], X_test[:, i])
                X_train[:, i] = tr
                X_test[:, i] = te

        X_train = ensure_int_matrix(X_train)
        X_test = ensure_int_matrix(X_test)
        y_train = df.loc[train_idx, 'Success'].values
        y_test = df.loc[test_idx, 'Success'].values

        clf = LogisticRegressionManual(learning_rate=0.01, n_iterations=5000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        results.append({'combo': combo, 'accuracy': acc})
        print(f"Combo {combo} -> Accuracy: {acc:.4f}")

    # best combo
    results_df = pd.DataFrame(results).sort_values(by='accuracy', ascending=False).reset_index(drop=True)
    best_combo = results_df.iloc[0]['combo']
    print(f"\nBEST FEATURE COMBINATION: {best_combo} -> Accuracy: {results_df.iloc[0]['accuracy']:.4f}")

    # save the table of results
    results_df.to_csv("logisticRegfeatures/logistic_feature_results.csv", index=False)

    # learning curve
    selected_cols = [col for key in best_combo for col in feature_groups[key]]

    X_train_full = df.loc[train_idx, selected_cols].copy().values
    X_test_full = df.loc[test_idx, selected_cols].copy().values

    for i, col in enumerate(selected_cols):
        if col in ['Year', 'Runtime (min)']:
            tr, te = discretize_train_test(X_train_full[:, i], X_test_full[:, i])
            X_train_full[:, i] = tr
            X_test_full[:, i] = te

    X_train_full = ensure_int_matrix(X_train_full)
    X_test_full = ensure_int_matrix(X_test_full)
    y_train_full = df.loc[train_idx, 'Success'].values
    y_test_full = df.loc[test_idx, 'Success'].values

    n_train_total = X_train_full.shape[0]
    order = np.arange(n_train_total)
    np.random.seed(42)
    np.random.shuffle(order)
    train_sizes = sorted(list({max(1, int(n_train_total * f)) for f in np.linspace(0.1, 1.0, 10)}))
    learning = []

    for n in train_sizes:
        idxs = order[:n]
        clf = LogisticRegressionManual(learning_rate=0.01, n_iterations=5000)
        clf.fit(X_train_full[idxs], y_train_full[idxs])
        y_pred_test = clf.predict(X_test_full)
        acc_test = accuracy_score(y_test_full, y_pred_test)
        learning.append((n, acc_test))
        print(f"Train samples: {n} -> Test accuracy: {acc_test:.4f}")

    train_counts = [x for x, _ in learning]
    accs = [a for _, a in learning]

    plt.figure(figsize=(8, 5))
    plt.plot(train_counts, accs, marker='o')
    plt.xlabel('Number of training examples')
    plt.ylabel('Accuracy on test set')
    plt.title('Learning curve (best feature set)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('logisticRegfeatures/learning_curve_best_features.png')

    print("Learning curve saved as 'logisticRegfeatures/learning_curve_best_features.png'")

    max_acc = max(accs)
    min_samples = train_counts[accs.index(max_acc)]
    print(f"Maximum observed accuracy: {max_acc:.4f}")
    print(f"Minimum training samples needed: {min_samples}")

if __name__ == "__main__":
    main()
