"""
Random Forest Classification Functions.

Provides functions for training a supervised Random Forest classifier
on PCA-clustered data and generating confusion matrices.

Based on GIMLI.py methodology from Rossi et al.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, balanced_accuracy_score


# -----------------------------------------------------------------------------
# Default configuration
# -----------------------------------------------------------------------------
DEFAULT_FEATURE_COLS = [
    "AMP1", "AMP2",
    "PPR2/1", "PPR3/1", "PPR4/1", "PPR5/1", "PPR6/1",
    "PPR7/1", "PPR8/1", "PPR9/1", "PPR10/1",
    "%Fail1", "%Fail2", "%Fail3"
]

DEFAULT_TARGET_COL = "HC_Cluster"

DEFAULT_PARAM_GRID = {
    "estimator__max_depth": [2, 3, 4, 5],
    "estimator__min_samples_split": [2, 5, 10],
    "estimator__min_samples_leaf": [2, 5, 10]
}
DEFAULT_GRIDSEARCH_N_JOBS = -1
DEFAULT_ESTIMATOR_N_JOBS = 1


# -----------------------------------------------------------------------------
# Internal helper functions
# -----------------------------------------------------------------------------
def _create_model(n_estimators=20, max_leaf_nodes=3, random_state=42,
                  estimator_n_jobs=DEFAULT_ESTIMATOR_N_JOBS):
    """Create a OneVsRestClassifier wrapping RandomForestClassifier.
    
    Uses class_weight='balanced' to handle imbalanced classes by adjusting
    weights inversely proportional to class frequencies.
    """
    base_rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_leaf_nodes=max_leaf_nodes,
        random_state=random_state,
        class_weight='balanced',  # Handle class imbalance
        n_jobs=estimator_n_jobs
    )
    return OneVsRestClassifier(base_rf, n_jobs=estimator_n_jobs)


def _find_best_hyperparameters(model, param_grid, X_train, y_train, cv,
                               verbose=True, n_jobs=DEFAULT_GRIDSEARCH_N_JOBS):
    """Use GridSearchCV to find the best hyperparameters.
    
    Uses balanced_accuracy as scoring metric to handle class imbalance.
    """
    grid = GridSearchCV(
        model,
        param_grid,
        n_jobs=n_jobs,
        cv=cv,
        scoring='balanced_accuracy',
        return_train_score=False,
        pre_dispatch='2*n_jobs'
    )
    grid.fit(X_train, y_train)
    
    if verbose:
        print('------------------')
        print(f'Best model score (balanced acc): {grid.best_score_:.3f}')
        print(f'Best params: {grid.best_params_}')
    
    return grid.best_estimator_, grid.best_score_


def _run_classification(X, y, n_splits, n_iter, param_grid, shuffle_labels=False, 
                        random_state=42, verbose=True,
                        gridsearch_n_jobs=DEFAULT_GRIDSEARCH_N_JOBS,
                        estimator_n_jobs=DEFAULT_ESTIMATOR_N_JOBS):
    """Run classification with StratifiedKFold cross-validation.
    
    Uses balanced_accuracy_score to handle class imbalance (average of recall
    for each class, giving equal weight to all classes regardless of size).
    """
    if shuffle_labels:
        np.random.seed(random_state)
        y = pd.Series(np.random.permutation(y.values), index=y.index)
    
    confusion_mats_all = []
    test_score_all = []
    
    for i in range(n_iter):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + i)
        
        for (train, test), j in zip(cv.split(X, y), range(1, n_splits + 1)):
            X_train, X_test = X.iloc[train], X.iloc[test]
            y_train, y_test = y.iloc[train], y.iloc[test]
            
            model = _create_model(
                random_state=random_state,
                estimator_n_jobs=estimator_n_jobs
            )
            model_best, _ = _find_best_hyperparameters(
                model, param_grid, X_train, y_train, cv,
                verbose=verbose, n_jobs=gridsearch_n_jobs
            )

            # Use balanced accuracy: average recall per class (handles imbalance)
            y_pred = model_best.predict(X_test)
            test_score = balanced_accuracy_score(y_test, y_pred)
            test_score_all.append(test_score)
            
            if verbose:
                print(f'Balanced accuracy TRIAL {i}; SPLIT {j}: {test_score:.3f}')
                print('------------------')
            
            matrix = confusion_matrix(y_test, y_pred, normalize='true')
            confusion_mats_all.append(pd.DataFrame(matrix))
    
    confusion_mat_final = pd.concat(confusion_mats_all).groupby(level=0).mean()
    
    if verbose:
        print(f'Total mean balanced accuracy: {np.mean(test_score_all):.3f}')
    
    return confusion_mat_final, test_score_all


def _plot_single_confusion_matrix(cm, labels, title, ax):
    """Plot a single confusion matrix using seaborn heatmap."""
    sns.heatmap(cm, cmap='YlGnBu', vmin=0., vmax=1.,
                xticklabels=[f'Cluster {i}' for i in labels],
                yticklabels=[f'Cluster {i}' for i in labels],
                ax=ax, annot=True, square=True, fmt='.2f')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)


def _plot_scores_boxplot(scores_actual, scores_shuffled, ax):
    """
    Plot boxplot comparing actual vs shuffled classification scores.
    
    Shows whisker bounds (minima/maxima), median, interquartile range,
    mean ± SD, and two-sided Student t-test p-value.
    """
    # Prepare data for boxplot
    data = [scores_actual, scores_shuffled]
    positions = [1, 2]
    
    # Create boxplot with whiskers at min/max
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                    showmeans=True, meanprops=dict(marker='o', markerfacecolor='red', 
                                                    markeredgecolor='red', markersize=8),
                    whis=[0, 100])  # Whiskers at min/max
    
    # Color the boxes
    colors = ['#4CAF50', '#9E9E9E']  # Green for actual, gray for shuffled
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Calculate statistics
    mean_actual = np.mean(scores_actual)
    std_actual = np.std(scores_actual)
    mean_shuffled = np.mean(scores_shuffled)
    std_shuffled = np.std(scores_shuffled)
    
    # Two-sided Student t-test
    t_stat, p_value = stats.ttest_ind(scores_actual, scores_shuffled)
    
    # Add scatter points for individual trials
    for i, (scores, pos) in enumerate(zip(data, positions)):
        x = np.random.normal(pos, 0.04, size=len(scores))
        ax.scatter(x, scores, alpha=0.5, color='black', s=20, zorder=3)
    
    # Add statistics text
    ax.text(1, mean_actual + 0.08, f'{mean_actual:.2f} ± {std_actual:.2f}', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.text(2, mean_shuffled + 0.08, f'{mean_shuffled:.2f} ± {std_shuffled:.2f}', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add significance bar and p-value
    y_max = max(max(scores_actual), max(scores_shuffled))
    bar_height = y_max + 0.15
    ax.plot([1, 1, 2, 2], [bar_height - 0.02, bar_height, bar_height, bar_height - 0.02], 
            'k-', linewidth=1.5)
    
    # Format p-value
    if p_value < 0.0001:
        p_text = f'p < 0.0001'
    elif p_value < 0.001:
        p_text = f'p = {p_value:.4f}'
    else:
        p_text = f'p = {p_value:.3f}'
    
    ax.text(1.5, bar_height + 0.01, p_text, ha='center', va='bottom', fontsize=10)
    
    # Labels and formatting
    ax.set_xticks(positions)
    ax.set_xticklabels(['Actual', 'Shuffled'])
    ax.set_ylabel('Balanced Accuracy')
    ax.set_title('Average Balanced Accuracy')
    ax.set_ylim(0, bar_height + 0.12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return t_stat, p_value


# -----------------------------------------------------------------------------
# Main public function
# -----------------------------------------------------------------------------
def random_forest_classification(df, 
                                  feature_cols=None, 
                                  target_col=None,
                                  n_splits=5, 
                                  n_iter=3,
                                  param_grid=None,
                                  random_state=42,
                                  verbose=True,
                                  gridsearch_n_jobs=DEFAULT_GRIDSEARCH_N_JOBS,
                                  estimator_n_jobs=DEFAULT_ESTIMATOR_N_JOBS,
                                  save_path=None):
    """
    Train a Random Forest classifier on PCA-clustered data and generate confusion matrices.
    
    Trains a supervised Random Forest classifier using the same methodology as GIMLI.py
    (Rossi et al.). Generates confusion matrices for actual labels and shuffled labels
    to assess classification performance vs. chance level.
    
    Parameters
    ----------
    df : DataFrame
        Input DataFrame containing feature columns and target column.
        Expected to be a PCA-clustered dataset (e.g., PCA_Data_WT_Pooled_clustered).
    
    feature_cols : list of str, optional
        List of feature column names to use for classification.
        Default: ["AMP1", "AMP2", "PPR2/1", ..., "PPR10/1", "%Fail1", "%Fail2", "%Fail3"]
    
    target_col : str, optional
        Name of the target column containing cluster labels.
        Default: "HC_Cluster"
    
    n_splits : int, optional
        Number of cross-validation splits. Default: 5
    
    n_iter : int, optional
        Number of iterations for averaging results. Default: 3
    
    param_grid : dict, optional
        Hyperparameter grid for GridSearchCV.
        Default: max_depth [2-5], min_samples_split [2,5,10], min_samples_leaf [2,5,10]
    
    random_state : int, optional
        Random seed for reproducibility. Default: 42
    
    verbose : bool, optional
        If True, print progress information. Default: True

    gridsearch_n_jobs : int, optional
        Number of parallel jobs for GridSearchCV. Default: -1 (all cores).
        This is the main speed-up control.

    estimator_n_jobs : int, optional
        Number of parallel jobs inside each RandomForest fit. Default: 1 to
        avoid nested oversubscription when GridSearchCV already runs in parallel.
    
    save_path : str, optional
        If provided, save the figure to this path.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the two confusion matrices.
    
    results : dict
        Dictionary containing:
        - 'cm_actual': Mean confusion matrix for actual labels
        - 'cm_shuffled': Mean confusion matrix for shuffled labels
        - 'scores_actual': List of test scores for actual labels
        - 'scores_shuffled': List of test scores for shuffled labels
        - 'accuracy_actual': Mean accuracy for actual labels
        - 'accuracy_shuffled': Mean accuracy for shuffled labels
    
    Example
    -------
    >>> import pandas as pd
    >>> from Func_RF import random_forest_classification
    >>> 
    >>> df = pd.read_excel("PCA_Data_WT_Pooled_clustered.xlsx")
    >>> fig, results = random_forest_classification(df)
    >>> print(f"Accuracy: {results['accuracy_actual']:.1%}")
    """
    # Set defaults
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS
    if target_col is None:
        target_col = DEFAULT_TARGET_COL
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID
    
    # Extract features and target
    X = df[feature_cols]
    y = df[target_col]
    labels = np.sort(np.unique(y))
    
    if verbose:
        print("=" * 60)
        print("Random Forest Classification of PCA Clusters")
        print("(Methodology from GIMLI.py - Rossi et al.)")
        print("=" * 60)
        print(f"\n  Samples: {X.shape[0]}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Classes: {labels}")
    
    # Run classification with actual labels
    if verbose:
        print("\n" + "=" * 60)
        print("Classification with ACTUAL labels")
        print("=" * 60)
    
    cm_actual, scores_actual = _run_classification(
        X, y, n_splits, n_iter, param_grid, 
        shuffle_labels=False, random_state=random_state, verbose=verbose,
        gridsearch_n_jobs=gridsearch_n_jobs,
        estimator_n_jobs=estimator_n_jobs
    )
    
    # Run classification with shuffled labels
    if verbose:
        print("\n" + "=" * 60)
        print("Classification with SHUFFLED labels (chance level)")
        print("=" * 60)
    
    cm_shuffled, scores_shuffled = _run_classification(
        X, y, n_splits, n_iter, param_grid, 
        shuffle_labels=True, random_state=random_state, verbose=verbose,
        gridsearch_n_jobs=gridsearch_n_jobs,
        estimator_n_jobs=estimator_n_jobs
    )
    
    # Compute summary statistics
    accuracy_actual = np.mean(scores_actual)
    accuracy_shuffled = np.mean(scores_shuffled)
    
    if verbose:
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"\nMean balanced accuracy (actual): {accuracy_actual:.3f} ± {np.std(scores_actual):.3f}")
        print(f"Mean balanced accuracy (shuffled): {accuracy_shuffled:.3f} ± {np.std(scores_shuffled):.3f}")
    
    # Create figure with 3 subplots: 2 confusion matrices + 1 boxplot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    _plot_single_confusion_matrix(cm_actual, labels, 'Actual Labels', axes[0])
    _plot_single_confusion_matrix(cm_shuffled, labels, 'Shuffled Labels (Chance)', axes[1])
    t_stat, p_value = _plot_scores_boxplot(scores_actual, scores_shuffled, axes[2])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if verbose:
            print(f"\nFigure saved to: {save_path}")
    
    if verbose:
        print(f"\nT-test: t = {t_stat:.3f}, p = {p_value:.2e}")
    
    # Prepare results dictionary
    results = {
        'cm_actual': cm_actual,
        'cm_shuffled': cm_shuffled,
        'scores_actual': scores_actual,
        'scores_shuffled': scores_shuffled,
        'accuracy_actual': accuracy_actual,
        'accuracy_shuffled': accuracy_shuffled,
        'std_actual': np.std(scores_actual),
        'std_shuffled': np.std(scores_shuffled),
        't_statistic': t_stat,
        'p_value': p_value,
        'labels': labels
    }
    
    return fig, results
