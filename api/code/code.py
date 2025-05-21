import pandas as pd
import numpy as np
import os

from core.consts import BASE_URL

import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42  # Ensures text is stored as text, not outlines
mpl.rcParams['ps.fonttype'] = 42  # Same for PS files

def z_score_normalize(df):
    """
    Normalize a DataFrame using Z-score normalization.
    """
    # Ensure the DataFrame contains only numeric data
    numeric_df = df.select_dtypes(include='number')

    # Calculate Z-score normalization
    normalized_df = (numeric_df - numeric_df.mean()) / numeric_df.std()

    return normalized_df



def process_file(input_file, output_dir):
    """
    Process the input file to normalize data and save results.
    """
    try:
        
        # Read the input CSV file
        main_df = pd.read_csv(input_file)
        
        # Check if 'condition' column exists
        if 'condition' not in main_df.columns:
            raise ValueError("The input file must contain a 'condition' column.")

        # Drop 'condition' column for normalization
        main_df_d = main_df.drop(columns=['condition'])
        
        # Normalize the data
        main_df_norm = z_score_normalize(main_df_d)
        
        # Reattach the 'condition' column
        main_df_norm['condition'] = main_df['condition']

        # Define output file paths
        normalized_file = os.path.join(output_dir, "z_score_normalized_data_of_ML_DF.csv")
        
        # Save normalized data
        main_df_norm.to_csv(normalized_file, index=False)
        
        return {
            "message": "Normalization completed successfully.",
            "normalized_file": normalized_file
        }
    except Exception as e:
        return {
            "message": "Error during normalization.",
            "error": str(e)
        }



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap  # Correct import
import os

import matplotlib as mpl

# Font settings for vector files
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# Set random seed for reproducibility
random_seed = 123

# Function for visualization using dimensionality reduction (PCA, t-SNE, UMAP)
def visualize_dimensionality_reduction(input_file, output_dir, user_info):
    try:
        # Read the input data
        df = pd.read_csv(input_file)

        # Check for 'condition' column
        if 'condition' not in df.columns:
            raise ValueError("The input file must contain a 'condition' column.")

        X = df.drop(columns=['condition'])  # Exclude the target variable
        y = df['condition']  # Target variable (condition)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # --- PCA ---
        pca = PCA(n_components=2, random_state=random_seed)
        pca_result = pca.fit_transform(X)
        pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
        pca_df['condition'] = y.values

        # Plot PCA
        pca_png = os.path.join(output_dir, "PCA_plot.png")
        pca_pdf = os.path.join(output_dir, "PCA_plot.pdf")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='PCA1', y='PCA2', hue='condition', data=pca_df, palette='viridis')
        plt.title('PCA of Data')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid()
        plt.legend(title='Condition')
        plt.savefig(pca_png)
        plt.savefig(pca_pdf)
        plt.close()

        # --- t-SNE ---
        def set_perplexity(n_samples):
            """Set appropriate perplexity based on the number of samples."""
            return min(30, max(5, n_samples // 3))

        # Get appropriate perplexity
        n_samples = X.shape[0]
        perplexity_value = set_perplexity(n_samples)

        tsne = TSNE(n_components=2, perplexity=perplexity_value, n_iter=300, random_state=random_seed)
        tsne_result = tsne.fit_transform(X)
        tsne_df = pd.DataFrame(data=tsne_result, columns=['TSNE1', 'TSNE2'])
        tsne_df['condition'] = y.values

        # Plot t-SNE
        tsne_png = os.path.join(output_dir, "tSNE_plot.png")
        tsne_pdf = os.path.join(output_dir, "tSNE_plot.pdf")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='TSNE1', y='TSNE2', hue='condition', data=tsne_df, palette='viridis')
        plt.title('t-SNE of Data')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid()
        plt.legend(title='Condition')
        plt.savefig(tsne_png)
        plt.savefig(tsne_pdf)
        plt.close()

        # --- UMAP ---
        umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=random_seed)
        umap_result = umap_model.fit_transform(X)
        umap_df = pd.DataFrame(data=umap_result, columns=['UMAP1', 'UMAP2'])
        umap_df['condition'] = y.values

        # Plot UMAP
        umap_png = os.path.join(output_dir, "UMAP_plot.png")
        umap_pdf = os.path.join(output_dir, "UMAP_plot.pdf")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='UMAP1', y='UMAP2', hue='condition', data=umap_df, palette='viridis')
        plt.title('UMAP of Data')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.grid()
        plt.legend(title='Condition')
        plt.savefig(umap_png)
        plt.savefig(umap_pdf)
        plt.close()

        # --- Combined Plot ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot PCA
        sns.scatterplot(
            x='PCA1', y='PCA2', hue='condition', data=pca_df, palette='viridis', ax=axes[0]
        )
        axes[0].set_title('PCA of Data')
        axes[0].set_xlabel('Principal Component 1')
        axes[0].set_ylabel('Principal Component 2')
        axes[0].legend(title='Condition')

        # Plot t-SNE
        sns.scatterplot(
            x='TSNE1', y='TSNE2', hue='condition', data=tsne_df, palette='viridis', ax=axes[1]
        )
        axes[1].set_title('t-SNE of Data')
        axes[1].set_xlabel('t-SNE Component 1')
        axes[1].set_ylabel('t-SNE Component 2')
        axes[1].legend(title='Condition')

        # Plot UMAP
        sns.scatterplot(
            x='UMAP1', y='UMAP2', hue='condition', data=umap_df, palette='viridis', ax=axes[2]
        )
        axes[2].set_title('UMAP of Data')
        axes[2].set_xlabel('UMAP Component 1')
        axes[2].set_ylabel('UMAP Component 2')
        axes[2].legend(title='Condition')


        # Add suptitle and adjust layout
        plt.suptitle("Dimensionality Reduction of All Features", fontsize=18, y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.98])


        # Save the combined plots
        combined_png = os.path.join(output_dir, f"dimensionality_reduction_combined_of_all_features.png")
        combined_pdf = os.path.join(output_dir, f"dimensionality_reduction_combined_of_all_features.pdf")
        plt.savefig(combined_png)
        plt.savefig(combined_pdf)
        plt.close()

        combined_png =  f"{BASE_URL}/files/{user_info['user_id']}/dimensionality_reduction_combined_of_all_features.png"
        combined_pdf =  f"{BASE_URL}/files/{user_info['user_id']}/dimensionality_reduction_combined_of_all_features.pdf"
        return {
            "message": "Dimensionality reduction visualizations created successfully.",
            "Combined": {"png": combined_png, "pdf": combined_pdf}
        }

    except Exception as e:
        return {
            "message": "Error during visualization.",
            "error": str(e)
        }



















import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_correlation_clustermap(input_file, output_dir, drop_column, user_info):
    try:
        # Read the input data
        df = pd.read_csv(input_file)

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Drop the specified column
        df_cor = df.drop(columns=[drop_column])

        # Compute the Pearson correlation matrix
        correlation_matrix = df_cor.corr(method='pearson')

        # Save the highly correlated pairs to a CSV file
        corr_pairs = correlation_matrix.unstack().reset_index()
        corr_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']
        corr_pairs = corr_pairs[
            (corr_pairs['Feature 1'] != corr_pairs['Feature 2']) & 
            (corr_pairs['Feature 1'] < corr_pairs['Feature 2'])
        ]
        corr_pairs = corr_pairs.sort_values(by='Correlation', ascending=False)
        corr_csv_path = os.path.join(output_dir, 'Highly_Correlated_Features.csv')
        corr_pairs.to_csv(corr_csv_path, index=False)

        # Create a clustermap
        clustermap = sns.clustermap(
            correlation_matrix,
            annot=False,
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            cbar_kws={"shrink": .8},
            method='average'
        )

        # Set title
        # plt.suptitle('Pearson Correlation Clustermap', fontsize=16)

        # Save the plot as a PDF and PNG file
        pdf_path = os.path.join(output_dir, 'Pearson_Correlation_Clustermap_of_All_Features.pdf')
        png_path = os.path.join(output_dir, 'Pearson_Correlation_Clustermap_of_All_Features.png')
        clustermap.savefig(pdf_path)
        clustermap.savefig(png_path)

        # Close the plot
        plt.close()

        corr_csv = f"{BASE_URL}/files/{user_info['user_id']}/Highly_Correlated_Features.csv"
        corr_pdf = f"{BASE_URL}/files/{user_info['user_id']}/Pearson_Correlation_Clustermap.pdf"
        corr_png = f"{BASE_URL}/files/{user_info['user_id']}/Pearson_Correlation_Clustermap.png"
        return {
            "message": "Correlation clustermap created successfully.",
            "output_files": {
                "correlation_csv": corr_csv,
                "correlation_pdf": corr_pdf,
                "correlation_png": corr_png
            }
        }

    except Exception as e:
        return {
            "message": "Error generating correlation clustermap.",
            "error": str(e)
        }


import os
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score

def feature_selection_and_model(input_file, output_dir, feature_ratio, user_info):
    try:
        # Load data
        df = pd.read_csv(input_file)

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Split data into features (X) and target (y)
        X = df.drop(columns=['condition'])
        y = df['condition']

        # Calculate the number of features to select based on the provided ratio
        num_features_to_select = int(X.shape[1] * feature_ratio)

        # Initialize RFE with Random Forest as the estimator
        rf_model = RandomForestClassifier(random_state=123)
        rfe = RFE(estimator=rf_model, n_features_to_select=num_features_to_select, step=1)
        rfe.fit(X, y)  # Fit RFE to the data

        # Get selected feature names
        selected_features = X.columns[rfe.support_]

        # Create a new DataFrame with selected features
        reduced_df = X[selected_features].copy()  # Retain only selected features
        reduced_df['condition'] = y  # Add the target variable back

        # Save the selected features and reduced DataFrame
        selected_features_path = os.path.join(output_dir, "selected_features_RFE_RF.csv")
        reduced_df.to_csv(selected_features_path, index=False)

        # Train and evaluate model using cross-validation (e.g., AUC score)
        rf_model_reduced = RandomForestClassifier(random_state=123)
        cv_scores = cross_val_score(rf_model_reduced, reduced_df[selected_features], y, cv=5, scoring='roc_auc')

        # Prepare output
        selected_features_csv = f"{BASE_URL}/files/{user_info['user_id']}/selected_features_RFE_RF.csv"
        result = {
            "message": "Feature selection and model training completed successfully.",
            "output_files": {
                "selected_features_csv": selected_features_csv
            },
            "model_metrics": {
                "cross_validation_auc": cv_scores.mean()
            }
        }
        return json.dumps(result)

    except Exception as e:
        return json.dumps({
            "message": "Error during feature selection and model training.",
            "error": str(e)
        })




    

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import (
    make_scorer,
    accuracy_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
    log_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from xgboost import XGBClassifier

import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score


# Define classifiers and hyperparameter grids
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=123),
    'Extra Trees': ExtraTreesClassifier(random_state=123),
    'Random Forest': RandomForestClassifier(random_state=123),
    'XGBoost' : XGBClassifier(eval_metric='logloss',
                                         use_label_encoder=False,
                                         random_state=123),
    'Gradient Boosting': GradientBoostingClassifier(random_state=123),
    'AdaBoost': AdaBoostClassifier(random_state=123)
}

param_grids = {
    'Logistic Regression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga']
    },
    'Extra Trees': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'AdaBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    }
}

# GLOBAL SCOPE
best_models = {}

# FUNCTION
def benchmark_models(input_file, output_dir, user_info):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate, cross_val_predict
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, balanced_accuracy_score,
        matthews_corrcoef, cohen_kappa_score, log_loss, make_scorer,
        roc_curve, precision_recall_curve
    )
    from sklearn.base import clone

    try:
        global best_models

        # Load the dataset
        df = pd.read_csv(input_file)
        X = df.drop(columns=['condition'])
        y = df['condition']

        # Cross-validation setup
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)

        # Define scoring metrics
        scoring = {
            'Accuracy': 'accuracy',
            'AUROC': 'roc_auc',
            'AUPRC': 'average_precision',
            'Precision': 'precision',
            'Recall': 'recall',
            'F1': 'f1',
            'Balanced_Accuracy': 'balanced_accuracy',
            'Log_Loss': 'neg_log_loss',
            'MCC': make_scorer(matthews_corrcoef),
            'Kappa': make_scorer(cohen_kappa_score)
        }

        # Storage
        results = []

        for name, clf in classifiers.items():
            print(f"\nBenchmarking {name}…")

            base_clf = clone(clf)
            tune_clf = clone(clf)

            # Default CV
            base_cv_results = cross_validate(
                estimator=base_clf,
                X=X, y=y,
                cv=outer_cv,
                scoring=scoring,
                n_jobs=-1
            )
            base_auroc = base_cv_results['test_AUROC'].mean()

            # Nested tuned CV
            grid_search = GridSearchCV(
                estimator=tune_clf,
                param_grid=param_grids[name],
                cv=inner_cv,
                scoring='roc_auc',
                n_jobs=-1
            )
            tuned_cv_results = cross_validate(
                estimator=grid_search,
                X=X, y=y,
                cv=outer_cv,
                scoring=scoring,
                n_jobs=-1
            )
            tuned_auroc = tuned_cv_results['test_AUROC'].mean()

            # Pick better model
            if tuned_auroc > base_auroc:
                chosen_cv = tuned_cv_results
                grid_search.fit(X, y)
                best_models[name] = grid_search.best_estimator_
                chosen = 'Tuned'
            else:
                chosen_cv = base_cv_results
                best_models[name] = clone(clf)
                chosen = 'Default'

            # Collect metrics
            entry = {'Model': name, 'Chosen': chosen}
            for metric in scoring:
                scores = chosen_cv[f'test_{metric}']
                entry[metric] = -scores.mean() if metric == 'Log_Loss' else scores.mean()
                entry[metric + '_std'] = scores.std()
            results.append(entry)

        # Create and save the metrics DataFrame
        metrics_df = pd.DataFrame(results).sort_values(by='AUPRC', ascending=False)
        metrics_path = os.path.join(output_dir, "ML_classifiers_benchmarking_results.csv")
        metrics_df.to_csv(metrics_path, index=False)

        # Plotting
        sorted_names = metrics_df['Model'].tolist()
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        for name in sorted_names:
            if name not in best_models:
                continue
            clf = best_models[name]
            y_proba = cross_val_predict(clf, X, y, cv=outer_cv, method='predict_proba', n_jobs=-1)[:, 1]

            precision, recall, _ = precision_recall_curve(y, y_proba)
            auprc = average_precision_score(y, y_proba)
            axes[0].plot(recall, precision, lw=1.75, label=f'{name} (AUPRC={auprc:.2f})')

            fpr, tpr, _ = roc_curve(y, y_proba)
            auroc = roc_auc_score(y, y_proba)
            axes[1].plot(fpr, tpr, lw=1.75, label=f'{name} (AUROC={auroc:.2f})')

        # PR baseline
        pos_rate = y.mean()
        axes[0].hlines(pos_rate, 0, 1, linestyles='--', color='black', label=f'Baseline={pos_rate:.2f}', zorder=1)

        axes[0].set_title('Precision–Recall Curves')
        axes[0].set_xlabel('Recall')
        axes[0].set_ylabel('Precision')
        axes[0].legend(loc='lower left', fontsize=9, frameon=True, facecolor='white')

        axes[1].plot([0, 1], [0, 1], 'k--', label='Random Chance')
        axes[1].set_title('ROC Curves')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].legend(loc='lower right', fontsize=9, frameon=True, facecolor='white')

        fig.suptitle('ML classifiers Benchmarking: AUPRC and AUROC (Nested CV)', fontsize=16, y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save the plots
        png_path = os.path.join(output_dir, 'ML_classifiers_benchmarking_curves.png')
        pdf_path = os.path.join(output_dir, 'ML_classifiers_benchmarking_curves.pdf')
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, dpi=300, bbox_inches='tight')

        # Return consistent structure
        metrics_csv = f"{BASE_URL}/files/{user_info['user_id']}/ML_classifiers_benchmarking_results.csv"
        return {
            "metrics": metrics_df.to_dict(orient="records"),
            "metrics_path": metrics_csv
        }

    except Exception as e:
        return {"error": str(e)}



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_model_and_importance_with_top10(metrics_df, best_models, reduced_df, selected_model_name, output_dir, user_info):
    """
    Analyze and visualize top 10 feature importance for a selected model.
    Saves both full feature importances and top-10 subset with a high-quality plot.
    """

    # Validate input
    if selected_model_name not in best_models:
        raise ValueError(f"Model '{selected_model_name}' not found in best_models.")

    if 'condition' not in reduced_df.columns:
        raise ValueError("Missing 'condition' column in reduced_df.")

    selected_model = best_models[selected_model_name]
    X = reduced_df.drop(columns=['condition'])
    y = reduced_df['condition']

    # Fit the model
    selected_model.fit(X, y)

    # Compute importance
    if hasattr(selected_model, 'feature_importances_'):
        importance_scores = selected_model.feature_importances_
    elif hasattr(selected_model, 'coef_'):
        importance_scores = np.abs(selected_model.coef_.flatten())
    else:
        raise AttributeError(
            f"Model '{selected_model_name}' does not support feature importance or coefficients."
        )

    # Normalize importance
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance_scores / importance_scores.sum()
    }).sort_values(by='Importance', ascending=False)

    # Extract top 10
    top10 = importance_df.head(10)
    top10_feature_names = top10['Feature'].tolist()
    columns_to_include = top10_feature_names + ['condition']
    top10_df = reduced_df[columns_to_include].copy()

    # File paths
    user_id = user_info['user_id']
    user_folder = os.path.join(output_dir, 'files', str(user_id))
    os.makedirs(user_folder, exist_ok=True)

    base_fname = selected_model_name.replace(' ', '_').lower()
    full_csv_path = os.path.join(user_folder, f"{base_fname}_feature_importance.csv")
    top10_csv_path = os.path.join(user_folder, f"top10_features_{base_fname}.csv")
    plot_png_path = os.path.join(user_folder, f"top10_feature_importance_{base_fname}.png")
    plot_pdf_path = os.path.join(user_folder, f"top10_feature_importance_{base_fname}.pdf")

    # Save full importance CSV
    importance_df.to_csv(full_csv_path, index=False)

    # Save top 10 CSV
    top10_df.to_csv(top10_csv_path, index=False)

    # Plot top 10
    top10_plot = top10[::-1]  # reverse for top-down barh
    plt.figure(figsize=(10, 6))
    plt.barh(top10_plot['Feature'], top10_plot['Importance'], color='steelblue')
    plt.title(f"Top 10 Important Features – {selected_model_name}")
    plt.xlabel("Normalized Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(plot_png_path, dpi=300, bbox_inches='tight')
    plt.savefig(plot_pdf_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Return as API-ready paths
    base_url = f"{BASE_URL}/files/{user_id}"
    return {
        "top10_features_path": f"{base_url}/top10_features_{base_fname}.csv",
        "top10_plot_path": f"{base_url}/top10_feature_importance_{base_fname}.png",
        "top10_features": top10.to_dict(orient="records"),
        "full_importance_path": f"{base_url}/{base_fname}_feature_importance.csv",
        "plot_pdf_path": f"{base_url}/top10_feature_importance_{base_fname}.pdf",
        "top10_feature_names": top10_feature_names
    }



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import os
from datetime import datetime

# Set random seed for reproducibility
random_seed = 123

# Function for visualization using dimensionality reduction (PCA, t-SNE, UMAP)
def visualize_dimensionality_reduction_feature(input_file, output_dir, user_info):
    try:
        # Read the input data
        df = pd.read_csv(input_file)

        # Check for 'condition' column
        if 'condition' not in df.columns:
            raise ValueError("The input file must contain a 'condition' column.")

        X = df.drop(columns=['condition'])  # Exclude the target variable
        y = df['condition']  # Target variable (condition)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # --- PCA ---
        pca = PCA(n_components=2, random_state=random_seed)
        pca_result = pca.fit_transform(X)
        pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
        pca_df['condition'] = y.values

        # Plot PCA
        pca_png = os.path.join(output_dir, "PCA_plot.png")
        pca_pdf = os.path.join(output_dir, "PCA_plot.pdf")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='PCA1', y='PCA2', hue='condition', data=pca_df, palette='viridis')
        plt.title('PCA of Data')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid()
        plt.legend(title='Condition')
        plt.savefig(pca_png)
        plt.savefig(pca_pdf)
        plt.close()

        # --- t-SNE ---
        def set_perplexity(n_samples):
            """Set appropriate perplexity based on the number of samples."""
            return min(30, max(5, n_samples // 3))

        # Get appropriate perplexity
        n_samples = X.shape[0]
        perplexity_value = set_perplexity(n_samples)

        tsne = TSNE(n_components=2, perplexity=perplexity_value, n_iter=300, random_state=random_seed)
        tsne_result = tsne.fit_transform(X)
        tsne_df = pd.DataFrame(data=tsne_result, columns=['TSNE1', 'TSNE2'])
        tsne_df['condition'] = y.values

        # Plot t-SNE
        tsne_png = os.path.join(output_dir, "tSNE_plot.png")
        tsne_pdf = os.path.join(output_dir, "tSNE_plot.pdf")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='TSNE1', y='TSNE2', hue='condition', data=tsne_df, palette='viridis')
        plt.title('t-SNE of Data')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid()
        plt.legend(title='Condition')
        plt.savefig(tsne_png)
        plt.savefig(tsne_pdf)
        plt.close()

        # --- UMAP ---
        umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=random_seed)
        umap_result = umap_model.fit_transform(X)
        umap_df = pd.DataFrame(data=umap_result, columns=['UMAP1', 'UMAP2'])
        umap_df['condition'] = y.values

        # Plot UMAP
        umap_png = os.path.join(output_dir, "UMAP_plot.png")
        umap_pdf = os.path.join(output_dir, "UMAP_plot.pdf")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='UMAP1', y='UMAP2', hue='condition', data=umap_df, palette='viridis')
        plt.title('UMAP of Data')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.grid()
        plt.legend(title='Condition')
        plt.savefig(umap_png)
        plt.savefig(umap_pdf)
        plt.close()

        # --- Combined Plot ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot PCA
        sns.scatterplot(
            x='PCA1', y='PCA2', hue='condition', data=pca_df, palette='viridis', ax=axes[0]
        )
        axes[0].set_title('PCA of Data')
        axes[0].set_xlabel('Principal Component 1')
        axes[0].set_ylabel('Principal Component 2')
        axes[0].legend(title='Condition')

        # Plot t-SNE
        sns.scatterplot(
            x='TSNE1', y='TSNE2', hue='condition', data=tsne_df, palette='viridis', ax=axes[1]
        )
        axes[1].set_title('t-SNE of Data')
        axes[1].set_xlabel('t-SNE Component 1')
        axes[1].set_ylabel('t-SNE Component 2')
        axes[1].legend(title='Condition')

        # Plot UMAP
        sns.scatterplot(
            x='UMAP1', y='UMAP2', hue='condition', data=umap_df, palette='viridis', ax=axes[2]
        )
        axes[2].set_title('UMAP of Data')
        axes[2].set_xlabel('UMAP Component 1')
        axes[2].set_ylabel('UMAP Component 2')
        axes[2].legend(title='Condition')

        # Add suptitle and adjust layout
        plt.suptitle("Dimensionality Reduction of Top 10 features", fontsize=18, y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        # Save the combined plots
        combined_png = os.path.join(output_dir, f"visualize_dimensions_Top_10_features.png")
        combined_pdf = os.path.join(output_dir, f"visualize_dimensions_Top_10_features.pdf")
        plt.savefig(combined_png)
        plt.savefig(combined_pdf)
        plt.close()

        combined_png =  f"{BASE_URL}/files/{user_info['user_id']}/visualize_dimensions_Top_10_features.png"
        combined_pdf =  f"{BASE_URL}/files/{user_info['user_id']}/visualize_dimensions_Top_10_features.pdf"

        return {
            "message": "Dimensionality reduction visualizations created successfully.",
            "Combined": {"png": combined_png, "pdf": combined_pdf}
        }

    except Exception as e:
        return {
            "message": "Error during visualization.",
            "error": str(e)
        }



import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, accuracy_score, matthews_corrcoef, log_loss,
    roc_curve, precision_recall_curve
)
from sklearn.base import clone

def rank_features(top10_df, selected_model, param_grids, classifiers, output_dir, user_info):
    """
    Rank top features based on single-feature model performance (AUPRC, AUROC, etc.).
    Saves CSV and plots ROC/PR curves for each.
    """

    try:
        # --- Validate inputs ---
        if selected_model not in param_grids:
            raise ValueError(f"Parameter grid not found for model: {selected_model}")
        if selected_model not in classifiers:
            raise ValueError(f"Classifier not found for model: {selected_model}")
        if 'condition' not in top10_df.columns:
            raise ValueError("Missing 'condition' column in top10_df.")

        # Prepare data
        X = top10_df.drop(columns=['condition'])
        y = top10_df['condition']

        model = classifiers[selected_model]
        param_grid = param_grids[selected_model]

        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        # Storage
        metrics_scores = []
        predictions = {}

        # --- Loop through each top feature ---
        for feature in X.columns:
            print(f"Evaluating single-feature model for: {feature}")
            X_single = X[[feature]]

            grid_search = GridSearchCV(
                estimator=clone(model),
                param_grid=param_grid,
                cv=inner_cv,
                scoring='roc_auc',
                n_jobs=-1
            )

            y_pred_proba = cross_val_predict(
                estimator=grid_search,
                X=X_single,
                y=y,
                cv=outer_cv,
                method='predict_proba',
                n_jobs=-1
            )[:, 1]

            y_pred = (y_pred_proba > 0.5).astype(int)

            metrics_scores.append({
                'Feature': feature,
                'AUPRC': average_precision_score(y, y_pred_proba),
                'AUROC': roc_auc_score(y, y_pred_proba),
                'Precision': precision_score(y, y_pred),
                'Recall': recall_score(y, y_pred),
                'F1-Score': f1_score(y, y_pred),
                'Accuracy': accuracy_score(y, y_pred),
                'MCC': matthews_corrcoef(y, y_pred),
                'LogLoss': log_loss(y, y_pred_proba)
            })

            predictions[feature] = y_pred_proba

        # --- Save metrics ---
        metrics_df = pd.DataFrame(metrics_scores).sort_values(by='AUPRC', ascending=False)

        # Save paths
        user_id = user_info['user_id']
        user_folder = os.path.join(output_dir, 'files', str(user_id))
        os.makedirs(user_folder, exist_ok=True)

        csv_path = os.path.join(user_folder, 'single_feature_metrics_ranking.csv')
        metrics_df.to_csv(csv_path, index=False)

        # --- Plotting ---
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Precision–Recall
        auprc_scores = {
            feature: average_precision_score(y, predictions[feature])
            for feature in predictions
        }
        sorted_auprc = sorted(auprc_scores.items(), key=lambda x: x[1], reverse=True)

        for feature, auprc in sorted_auprc:
            precision, recall, _ = precision_recall_curve(y, predictions[feature])
            axes[0].plot(recall, precision, lw=1.75, label=f"{feature} (AUPRC = {auprc:.2f})")

        pos_rate = y.mean()
        axes[0].hlines(pos_rate, 0, 1, linestyles='--', color='black', label=f'Baseline={pos_rate:.2f}')
        axes[0].set_title('Precision–Recall Curves (Single-Gene)')
        axes[0].set_xlabel('Recall')
        axes[0].set_ylabel('Precision')
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1.05)
        axes[0].legend(loc='lower left', fontsize=8, frameon=True, facecolor='white')

        # ROC
        for feature, _ in sorted_auprc:
            fpr, tpr, _ = roc_curve(y, predictions[feature])
            auroc = roc_auc_score(y, predictions[feature])
            axes[1].plot(fpr, tpr, lw=1.75, label=f"{feature} (AUROC = {auroc:.2f})")

        axes[1].plot([0, 1], [0, 1], 'k--', label='Random Chance')
        axes[1].set_title('ROC Curves (Single-Gene)')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1.05)
        axes[1].legend(loc='lower right', fontsize=8, frameon=True, facecolor='white')

        fig.suptitle('AUPRC and AUROC of Single-Gene Models', fontsize=16, y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save figures
        plot_png = os.path.join(user_folder, 'single_feature_model_performance_landscape.png')
        plot_pdf = os.path.join(user_folder, 'single_feature_model_performance_landscape.pdf')
        fig.savefig(plot_png, dpi=300, bbox_inches='tight')
        fig.savefig(plot_pdf, dpi=300, bbox_inches='tight')
        plt.close()

        # Return URLs
        base_url = f"{BASE_URL}/files/{user_id}"
        return json.dumps({
            "message": "Feature ranking and plotting completed successfully.",
            "ranking_file": f"{base_url}/single_feature_metrics_ranking.csv",
            "plot_png": f"{base_url}/single_feature_model_performance_landscape.png",
            "plot_pdf": f"{base_url}/single_feature_model_performance_landscape.pdf",
            "metrics": metrics_df.to_dict(orient="records")
        })

    except Exception as e:
        return json.dumps({
            "message": "Error during feature ranking and plotting.",
            "error": str(e)
        })



import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, f1_score,
    roc_curve, precision_recall_curve, precision_score, recall_score,
    matthews_corrcoef, log_loss
)

def evaluate_model_with_features(top10_df, top10_df_array, selected_model, param_grids, classifiers, output_dir, user_info):
    """
    Evaluate the performance of models using top-N features (10 to 1), save plots and metrics, and select the best feature subset.
    """
    try:
        # Prepare cross-validation
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        # Storage
        roc_curves = []
        pr_curves = []
        performance_metrics = []

        for n_features in range(10, 0, -1):
            print(f"\nEvaluating model with top {n_features} genes...")

            selected_features = list(top10_df_array[:n_features])
            current_df = top10_df[selected_features + ['condition']]

            X = current_df.drop(columns=['condition'])
            y = current_df['condition']

            # Nested model tuning
            nested_model = GridSearchCV(
                estimator=classifiers[selected_model],
                param_grid=param_grids[selected_model],
                cv=inner_cv,
                scoring='roc_auc',
                n_jobs=-1
            )

            y_pred_proba = cross_val_predict(
                nested_model, X, y,
                cv=outer_cv,
                method='predict_proba',
                n_jobs=-1
            )[:, 1]

            y_pred = (y_pred_proba > 0.5).astype(int)

            # Metrics
            roc_auc = roc_auc_score(y, y_pred_proba)
            pr_auc = average_precision_score(y, y_pred_proba)
            f1 = f1_score(y, y_pred)
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)
            mcc = matthews_corrcoef(y, y_pred)
            logloss = log_loss(y, y_pred_proba)

            fpr, tpr, _ = roc_curve(y, y_pred_proba)
            precision_curve, recall_curve, _ = precision_recall_curve(y, y_pred_proba)

            roc_curves.append((fpr, tpr, roc_auc, n_features))
            pr_curves.append((precision_curve, recall_curve, pr_auc, n_features))

            performance_metrics.append({
                'Number of Genes': n_features,
                'AUPRC': pr_auc,
                'AUROC': roc_auc,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'Accuracy': accuracy,
                'MCC': mcc,
                'Log Loss': logloss
            })

        # Save performance metrics
        metrics_df = pd.DataFrame(performance_metrics)
        metrics_df.sort_values(by='AUPRC', ascending=False, inplace=True)

        user_id = user_info['user_id']
        user_folder = os.path.join(output_dir, 'files', str(user_id))
        os.makedirs(user_folder, exist_ok=True)

        metrics_csv_path = os.path.join(user_folder, 'biomarker_algorithms_performance.csv')
        metrics_df.to_csv(metrics_csv_path, index=False)

        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        roc_curves.sort(key=lambda x: x[2], reverse=True)
        pr_curves.sort(key=lambda x: x[2], reverse=True)

        for precision, recall, pr_auc, n_features in pr_curves:
            axes[0].plot(recall, precision, lw=1.75, label=f'{n_features}-Gene Model (AUPRC = {pr_auc:.2f})')
        pos_rate = y.mean()
        axes[0].hlines(pos_rate, 0, 1, linestyles='--', color='black', label=f'Baseline={pos_rate:.2f}')
        axes[0].set_title('Precision–Recall Curves')
        axes[0].set_xlabel('Recall')
        axes[0].set_ylabel('Precision')
        axes[0].legend(loc='lower left', fontsize=8, frameon=True)

        for fpr, tpr, roc_auc, n_features in roc_curves:
            axes[1].plot(fpr, tpr, lw=1.75, label=f'{n_features}-Gene Model (AUROC = {roc_auc:.2f})')
        axes[1].plot([0, 1], [0, 1], 'k--')
        axes[1].set_title('ROC Curves')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].legend(loc='lower right', fontsize=8, frameon=True)

        fig.suptitle('AUPRC and AUROC Plots of Gene-Models', fontsize=16, y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        png_path = os.path.join(user_folder, 'biomarker_algorithms_performance_metrics.png')
        pdf_path = os.path.join(user_folder, 'biomarker_algorithms_performance_metrics.pdf')
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        plt.close()

        # Select best feature subset based on AUPRC
        best_pr_curve = max(pr_curves, key=lambda x: x[2])
        best_n = best_pr_curve[3]
        selected_features = list(top10_df_array[:best_n])

        if any(f not in top10_df.columns for f in selected_features):
            raise ValueError("Some selected features are missing in top10_df.")

        final_df = top10_df[selected_features + ['condition']]
        final_df_path = os.path.join(user_folder, 'final_selected_biomarker_algorithms_df.csv')
        final_df.to_csv(final_df_path, index=False)

        # Return paths and results
        base_url = f"{BASE_URL}/files/{user_id}"
        return {
            "message": "Evaluation completed successfully.",
            "metrics_file": f"{base_url}/biomarker_algorithms_performance.csv",
            "plot_png": f"{base_url}/biomarker_algorithms_performance_metrics.png",
            "plot_pdf": f"{base_url}/biomarker_algorithms_performance_metrics.pdf",
            "selected_features_file": f"{base_url}/final_selected_biomarker_algorithms_df.csv",
            "metrics": metrics_df.to_dict(orient="records"),
            "best_n_features": best_n,
            "selected_features": selected_features
        }

    except Exception as e:
        return {"message": "Error during model evaluation.", "error": str(e)}


# Function for visualization using dimensionality reduction (PCA, t-SNE, UMAP)
def visualize_dimensionality_reduction_final(input_file, output_dir, user_info):
    try:
        # Read the input data
        df = pd.read_csv(input_file)

        # Check for 'condition' column
        if 'condition' not in df.columns:
            raise ValueError("The input file must contain a 'condition' column.")

        X = df.drop(columns=['condition'])  # Exclude the target variable
        y = df['condition']  # Target variable (condition)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # --- PCA ---
        pca = PCA(n_components=2, random_state=random_seed)
        pca_result = pca.fit_transform(X)
        pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
        pca_df['condition'] = y.values

        # Plot PCA
        pca_png = os.path.join(output_dir, "PCA_plot_final.png")
        pca_pdf = os.path.join(output_dir, "PCA_plot_final.pdf")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='PCA1', y='PCA2', hue='condition', data=pca_df, palette='viridis')
        plt.title('PCA of Data')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid()
        plt.legend(title='Condition')
        plt.savefig(pca_png)
        plt.savefig(pca_pdf)
        plt.close()

        # --- t-SNE ---
        def set_perplexity(n_samples):
            """Set appropriate perplexity based on the number of samples."""
            return min(30, max(5, n_samples // 3))

        # Get appropriate perplexity
        n_samples = X.shape[0]
        perplexity_value = set_perplexity(n_samples)

        tsne = TSNE(n_components=2, perplexity=perplexity_value, n_iter=300, random_state=random_seed)
        tsne_result = tsne.fit_transform(X)
        tsne_df = pd.DataFrame(data=tsne_result, columns=['TSNE1', 'TSNE2'])
        tsne_df['condition'] = y.values

        # Plot t-SNE
        tsne_png = os.path.join(output_dir, "tSNE_plot_final.png")
        tsne_pdf = os.path.join(output_dir, "tSNE_plot_final.pdf")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='TSNE1', y='TSNE2', hue='condition', data=tsne_df, palette='viridis')
        plt.title('t-SNE of Data')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid()
        plt.legend(title='Condition')
        plt.savefig(tsne_png)
        plt.savefig(tsne_pdf)
        plt.close()

        # --- UMAP ---
        umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=random_seed)
        umap_result = umap_model.fit_transform(X)
        umap_df = pd.DataFrame(data=umap_result, columns=['UMAP1', 'UMAP2'])
        umap_df['condition'] = y.values

        # Plot UMAP
        umap_png = os.path.join(output_dir, "UMAP_plot_final.png")
        umap_pdf = os.path.join(output_dir, "UMAP_plot_final.pdf")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='UMAP1', y='UMAP2', hue='condition', data=umap_df, palette='viridis')
        plt.title('UMAP of Data')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.grid()
        plt.legend(title='Condition')
        plt.savefig(umap_png)
        plt.savefig(umap_pdf)
        plt.close()

        # --- Combined Plot ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot PCA
        sns.scatterplot(
            x='PCA1', y='PCA2', hue='condition', data=pca_df, palette='viridis', ax=axes[0]
        )
        axes[0].set_title('PCA of Data')
        axes[0].set_xlabel('Principal Component 1')
        axes[0].set_ylabel('Principal Component 2')
        axes[0].legend(title='Condition')

        # Plot t-SNE
        sns.scatterplot(
            x='TSNE1', y='TSNE2', hue='condition', data=tsne_df, palette='viridis', ax=axes[1]
        )
        axes[1].set_title('t-SNE of Data')
        axes[1].set_xlabel('t-SNE Component 1')
        axes[1].set_ylabel('t-SNE Component 2')
        axes[1].legend(title='Condition')

        # Plot UMAP
        sns.scatterplot(
            x='UMAP1', y='UMAP2', hue='condition', data=umap_df, palette='viridis', ax=axes[2]
        )
        axes[2].set_title('UMAP of Data')
        axes[2].set_xlabel('UMAP Component 1')
        axes[2].set_ylabel('UMAP Component 2')
        axes[2].legend(title='Condition')

        # Add suptitle and adjust layout
        plt.suptitle("Dimensionality Reduction of Final Model", fontsize=18, y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        # Save the combined plots
        combined_png = os.path.join(output_dir, f"visualize_dimensions_final_model.png")
        combined_pdf = os.path.join(output_dir, f"visualize_dimensions_final_model.pdf")
        plt.savefig(combined_png)
        plt.savefig(combined_pdf)
        plt.close()

        combined_png =  f"{BASE_URL}/files/{user_info['user_id']}/visualize_dimensions_final_model.png"
        combined_pdf =  f"{BASE_URL}/files/{user_info['user_id']}/visualize_dimensions_final_model.pdf"

        return {
            "message": "Dimensionality reduction visualizations created successfully.",
            "Combined": {"png": combined_png, "pdf": combined_pdf}
        }

    except Exception as e:
        return {
            "message": "Error during visualization.",
            "error": str(e)
        }
    

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, log_loss, confusion_matrix, ConfusionMatrixDisplay
)

def evaluate_final_model(final_df_path, selected_model, param_grids, classifiers, output_dir, user_info):
    """
    Final model training, evaluation (train/test), and artifact saving for production use.
    """

    try:
        # Load data
        final_df = pd.read_csv(final_df_path)
        if 'condition' not in final_df.columns:
            raise ValueError("The input file must contain a 'condition' column.")

        X = final_df.drop(columns=['condition'])
        y = final_df['condition']

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)

        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        model = classifiers[selected_model]
        param_grid = param_grids[selected_model]

        # Nested CV on training set
        nested_grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_cv, scoring='roc_auc', n_jobs=-1)
        y_pred_proba_train = cross_val_predict(nested_grid, X_train, y_train, cv=outer_cv, method='predict_proba', n_jobs=-1)[:, 1]
        y_pred_train = (y_pred_proba_train > 0.5).astype(int)

        # Training metrics
        train_metrics = {
            'Set': 'Train (Nested CV)',
            'AUROC': roc_auc_score(y_train, y_pred_proba_train),
            'AUPRC': average_precision_score(y_train, y_pred_proba_train),
            'Accuracy': accuracy_score(y_train, y_pred_train),
            'F1-Score': f1_score(y_train, y_pred_train),
            'Precision': precision_score(y_train, y_pred_train),
            'Recall': recall_score(y_train, y_pred_train),
            'MCC': matthews_corrcoef(y_train, y_pred_train),
            'LogLoss': log_loss(y_train, y_pred_proba_train),
            'Confusion Matrix': confusion_matrix(y_train, y_pred_train).tolist()
        }

        # Final model refit on full training data
        final_grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_cv, scoring='roc_auc', n_jobs=-1)
        final_grid.fit(X_train, y_train)
        tuned_model = final_grid.best_estimator_

        # Test evaluation
        y_pred_proba_test = tuned_model.predict_proba(X_test)[:, 1]
        y_pred_test = (y_pred_proba_test > 0.5).astype(int)

        test_metrics = {
            'Set': 'Test',
            'AUROC': roc_auc_score(y_test, y_pred_proba_test),
            'AUPRC': average_precision_score(y_test, y_pred_proba_test),
            'Accuracy': accuracy_score(y_test, y_pred_test),
            'F1-Score': f1_score(y_test, y_pred_test),
            'Precision': precision_score(y_test, y_pred_test),
            'Recall': recall_score(y_test, y_pred_test),
            'MCC': matthews_corrcoef(y_test, y_pred_test),
            'LogLoss': log_loss(y_test, y_pred_proba_test),
            'Confusion Matrix': confusion_matrix(y_test, y_pred_test).tolist()
        }

        metrics_df = pd.DataFrame([train_metrics, test_metrics])

        # Create user folder
        user_id = user_info['user_id']
        user_folder = os.path.join(output_dir, 'files', str(user_id))
        os.makedirs(user_folder, exist_ok=True)

        # Save metrics CSV
        metrics_csv_path = os.path.join(user_folder, 'final_model_metrics_summary.csv')
        metrics_df.drop(columns=['Confusion Matrix']).to_csv(metrics_csv_path, index=False)

        # Save model
        model_path = os.path.join(user_folder, 'final_model.joblib')
        dump(tuned_model, model_path)

        # Plot PR and ROC curves
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # PR
        precision_train, recall_train, _ = precision_recall_curve(y_train, y_pred_proba_train)
        precision_test, recall_test, _ = precision_recall_curve(y_test, y_pred_proba_test)
        axes[0].plot(recall_train, precision_train, label=f'Train (AUPRC = {train_metrics["AUPRC"]:.2f})')
        axes[0].plot(recall_test, precision_test, label=f'Test (AUPRC = {test_metrics["AUPRC"]:.2f})', linestyle='--')
        axes[0].hlines(y.mean(), 0, 1, linestyles='--', color='black', label=f'Baseline={y.mean():.2f}')
        axes[0].set_title('Precision–Recall Curve')
        axes[0].set_xlabel('Recall')
        axes[0].set_ylabel('Precision')
        axes[0].legend(loc='lower left')

        # ROC
        fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_proba_train)
        fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_proba_test)
        axes[1].plot(fpr_train, tpr_train, label=f'Train (AUROC = {train_metrics["AUROC"]:.2f})')
        axes[1].plot(fpr_test, tpr_test, label=f'Test (AUROC = {test_metrics["AUROC"]:.2f})', linestyle='--')
        axes[1].plot([0, 1], [0, 1], 'k--')
        axes[1].set_title('ROC Curve')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].legend(loc='lower right')

        fig.suptitle('Performance of the Final Model (Train vs Test)', fontsize=15, y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        pr_roc_png = os.path.join(user_folder, 'final_model_performance.png')
        plt.savefig(pr_roc_png, dpi=300, bbox_inches='tight')
        plt.close()

        # Plot confusion matrices
        cm_train = train_metrics['Confusion Matrix']
        cm_test = test_metrics['Confusion Matrix']
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ConfusionMatrixDisplay(np.array(cm_train), display_labels=['Negative', 'Positive']).plot(cmap='Blues', ax=axes[0])
        axes[0].set_title("Train Confusion Matrix")
        ConfusionMatrixDisplay(np.array(cm_test), display_labels=['Negative', 'Positive']).plot(cmap='Oranges', ax=axes[1])
        axes[1].set_title("Test Confusion Matrix")
        fig.suptitle('Confusion Matrices of Final Model: Train vs Test', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        cm_png = os.path.join(user_folder, 'final_model_confusion_matrix.png')
        plt.savefig(cm_png, dpi=300, bbox_inches='tight')
        plt.close()

        # Return URLs
        base_url = f"{BASE_URL}/files/{user_id}"
        return {
            "message": "Final model evaluation completed successfully.",
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "model_path": f"{base_url}/final_model.joblib",
            "metrics_file": f"{base_url}/final_model_metrics_summary.csv",
            "pr_roc_plot": f"{base_url}/final_model_performance.png",
            "confusion_matrix_plot": f"{base_url}/final_model_confusion_matrix.png"
        }

    except Exception as e:
        return {"message": "Error during final model evaluation.", "error": str(e)}
