import pandas as pd
import numpy as np
import itertools
import math
import matplotlib.pyplot as plt
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from naive import NaiveBayes

def discretize_train_test(train_col, test_col, n_bins=5):
    train_col = np.asarray(train_col, dtype=float)
    test_col = np.asarray(test_col, dtype=float)
    if np.unique(train_col).size <= 1:
        return train_col.astype(int), test_col.astype(int)
    bins = np.quantile(train_col, np.linspace(0, 1, n_bins+1))
    bins = np.unique(bins)
    if len(bins) <= 1:
        return train_col.astype(int), test_col.astype(int)
    train_disc = np.digitize(train_col, bins, right=True)
    test_disc = np.digitize(test_col, bins, right=True)
    return train_disc.astype(int), test_disc.astype(int)

def ensure_int_matrix(arr):
    return np.asarray(arr).astype(int)


def main():
    df = pd.read_csv('processed_movies_data.csv')

    feature_groups = {
        'TitleWords': ['Title'],
        'Year': ['Year'],
        'Genre': ['GenreId'],
        'Director': ['DirectorId'],
        'LeadActor': ['ActorId'],
        'Production': ['ProductionId'],
        'Runtime': ['Runtime (min)'],
        'Country': ['CountryId'],
        'Language': ['LanguageId']
    }

    keys = list(feature_groups.keys())
    combos = []
    for r in (1,2,3):
        combos.extend(itertools.combinations(keys,r))
    combos.append(tuple(keys))

    # Train-test split
    stratify_col = df['Success'] if df['Success'].nunique() > 1 else None
    train_idx, test_idx = train_test_split(df.index.values, test_size=0.2, random_state=42,
                                           stratify=stratify_col)
    results = []

    # Evaluate all feature combinations
    for combo in combos:
        uses_title = 'TitleWords' in combo
        selected_non_title_cols = [col for key in combo if key != 'TitleWords' for col in feature_groups[key]]

        # ---------------------------
        # If using title column
        # ---------------------------
        if uses_title:
            tfidf = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b", max_features=200)
            X_title_train = tfidf.fit_transform(df.loc[train_idx,'Title'].astype(str).values)
            X_title_test = tfidf.transform(df.loc[test_idx,'Title'].astype(str).values)

            cat_cols = [col for col in selected_non_title_cols if col not in ('Year','Runtime (min)')]
            num_cols = [col for col in selected_non_title_cols if col in ('Year','Runtime (min)')]

            X_other_train_parts = []
            X_other_test_parts = []

            if cat_cols:
                ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
                X_cat_train = ohe.fit_transform(df.loc[train_idx,cat_cols].values)
                X_cat_test = ohe.transform(df.loc[test_idx,cat_cols].values)
                X_other_train_parts.append(X_cat_train)
                X_other_test_parts.append(X_cat_test)

            if num_cols:
                for col in num_cols:
                    tr_disc, te_disc = discretize_train_test(df.loc[train_idx,col].values,
                                                             df.loc[test_idx,col].values)
                    X_other_train_parts.append(csr_matrix(tr_disc.reshape(-1,1)))
                    X_other_test_parts.append(csr_matrix(te_disc.reshape(-1,1)))

            X_train_combined = hstack([X_title_train]+X_other_train_parts, format='csr') if X_other_train_parts else X_title_train
            X_test_combined = hstack([X_title_test]+X_other_test_parts, format='csr') if X_other_test_parts else X_title_test

            clf = NaiveBayes()
            clf.fit(X_train_combined.toarray(), df.loc[train_idx,'Success'].values)
            y_pred = clf.predict(X_test_combined.toarray())
            acc = accuracy_score(df.loc[test_idx,'Success'].values, y_pred)

        # ---------------------------
        # Only non-title columns
        # ---------------------------
        else:
            X_train = df.loc[train_idx, selected_non_title_cols].values.copy()
            X_test = df.loc[test_idx, selected_non_title_cols].values.copy()
            num_cols = [col for col in selected_non_title_cols if col in ('Year','Runtime (min)')]

            if num_cols:
                for i, col in enumerate(selected_non_title_cols):
                    if col in num_cols:
                        tr_disc, te_disc = discretize_train_test(X_train[:,i], X_test[:,i])
                        X_train[:,i] = tr_disc
                        X_test[:,i] = te_disc

            X_train = ensure_int_matrix(X_train)
            X_test = ensure_int_matrix(X_test)

            clf = NaiveBayes()
            clf.fit(X_train, df.loc[train_idx,'Success'].values)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(df.loc[test_idx,'Success'].values, y_pred)

        results.append({'combo':combo,'accuracy':acc})
        print(f"Combo {combo} -> Accuracy: {acc:.4f}")

    # ---------------------------
    # Determine best combination
    # ---------------------------
    results_df = pd.DataFrame(results).sort_values(by='accuracy', ascending=False).reset_index(drop=True)
    best_combo = results_df.iloc[0]['combo']
    print(f"\nBEST FEATURE COMBINATION: {best_combo} -> Accuracy: {results_df.iloc[0]['accuracy']:.4f}")

    # ---------------------------
    # Learning curve for best combo
    # ---------------------------
    learning = [] 

    uses_title = 'TitleWords' in best_combo
    selected_non_title_cols = [col for key in best_combo if key != 'TitleWords' for col in feature_groups[key]]

    if uses_title:
        tfidf = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b", max_features=200)
        X_title_train_full = tfidf.fit_transform(df.loc[train_idx,'Title'].astype(str).values)
        X_title_test_full = tfidf.transform(df.loc[test_idx,'Title'].astype(str).values)

        cat_cols = [col for col in selected_non_title_cols if col not in ('Year','Runtime (min)')]
        num_cols = [col for col in selected_non_title_cols if col in ('Year','Runtime (min)')]

        X_other_train_parts_full = []
        X_other_test_parts_full = []

        if cat_cols:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
            X_cat_train = ohe.fit_transform(df.loc[train_idx,cat_cols].values)
            X_cat_test = ohe.transform(df.loc[test_idx,cat_cols].values)
            X_other_train_parts_full.append(X_cat_train)
            X_other_test_parts_full.append(X_cat_test)

        if num_cols:
            for col in num_cols:
                tr_disc, te_disc = discretize_train_test(df.loc[train_idx,col].values,
                                                         df.loc[test_idx,col].values)
                X_other_train_parts_full.append(csr_matrix(tr_disc.reshape(-1,1)))
                X_other_test_parts_full.append(csr_matrix(te_disc.reshape(-1,1)))

        X_train_full_combined = hstack([X_title_train_full]+X_other_train_parts_full, format='csr') if X_other_train_parts_full else X_title_train_full
        X_test_full_combined = hstack([X_title_test_full]+X_other_test_parts_full, format='csr') if X_other_test_parts_full else X_title_test_full

        # Learning curve: incremental training
        n_train_total = X_train_full_combined.shape[0]
        order = np.arange(n_train_total)
        np.random.seed(42)
        np.random.shuffle(order)

        train_sizes = sorted(list({max(1,int(math.ceil(frac*n_train_total))) for frac in np.linspace(0.1,1.0,10)}))
        for n in train_sizes:
            idxs = order[:n]
            clf = NaiveBayes()
            clf.fit(X_train_full_combined[idxs,:].toarray(), df.loc[train_idx,'Success'].values[idxs])
            y_pred_test = clf.predict(X_test_full_combined.toarray())
            acc_test = accuracy_score(df.loc[test_idx,'Success'].values, y_pred_test)
            learning.append((n, acc_test))
            print(f"Train samples: {n} -> Test accuracy: {acc_test:.4f}")

    # ---------------------------
    # Plot learning curve
    # ---------------------------
    if learning:
        train_counts = [x for x,_ in learning]
        accs = [a for _,a in learning]
        plt.figure(figsize=(8,5))
        plt.plot(train_counts, accs, marker='o')
        plt.xlabel('Number of training examples')
        plt.ylabel('Accuracy on fixed test set')
        plt.title('Learning curve (best feature set)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('accuracy_vs_training_size.png')
        print("Learning curve saved as 'accuracy_vs_training_size.png'")

        # Minimum samples needed for max accuracy
        max_acc = max(accs)
        min_samples = train_counts[accs.index(max_acc)]
        print(f"Maximum observed accuracy: {max_acc:.4f}")
        print(f"Minimum training samples needed: {min_samples}")
    else:
        print("No learning curve data available.")

if __name__ == '__main__':
    main()
