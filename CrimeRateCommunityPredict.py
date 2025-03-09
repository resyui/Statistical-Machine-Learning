#%%
!pip install ucimlrepo pandas numpy matplotlib seaborn scikit-learn statsmodels mlxtend joblib
#%%
# Import libraries
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import sklearn
from ucimlrepo import fetch_ucirepo
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from mlxtend.feature_selection import ExhaustiveFeatureSelector, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV, lasso_path, enet_path, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from collections import defaultdict, Counter
from joblib import Parallel, delayed
#%%
# ~Part 0: Data preprocessing~

# fetch the dataset from UCI server
com_and_crime = fetch_ucirepo(id=183)

# Extract features and targets
X = com_and_crime.data.features
y = com_and_crime.data.targets

# Create combined dataframe
df = pd.concat([X, y], axis=1)

# ~Part 0: Data Preprocessing~

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Show Metadata and see distribution of the data
print(com_and_crime.metadata)
print("Initial shape:", df.shape)
print(df.describe())

df.hist(bins=30, figsize=(12, 8))
plt.tight_layout()
plt.show()

# Identify columns with high missing values
missing = df.isna().mean().sort_values(ascending=False)
high_missing = missing[missing > 0.4].index
print("Columns with >40% missing values:", high_missing.tolist())

# Remove columns with >40% missing values
df.drop(columns=high_missing, inplace=True)

# Remove non-predictive columns
df.drop(columns=['communityname', 'state', 'county', 'community', 'fold'], inplace=True, errors='ignore')

# Check which column is still non-numerical after removal of columns
object_cols = [col for col in df.columns if df[col].dtype == 'object']
print(object_cols)

# Convert the column into floating and check if conversion is successful
df['OtherPerCap'] = pd.to_numeric(df['OtherPerCap'], errors='coerce')
print(df.OtherPerCap)

# Check where is the remaining missing value
df[df.isna().any(axis=1)]
pd.set_option('display.max_columns', None)

# Impute a value for the only missing value in OtherPerCap
imputer = SimpleImputer(strategy='median')
df['OtherPerCap'] = imputer.fit_transform(df[['OtherPerCap']])

# Ensure there is no more missing value
print("Remaining missing values:", df.isna().sum().sum())

# Compute the correlation matrix and plot the heatmap to get a sense of what is happening in the dataset
corr_matrix = df.corr()
print(corr_matrix)
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False)
plt.title('Correlation Heatmap')
plt.show()
#%%
# ~Part 1: Data Analysis~
# (a) (i) Compare and contrast

# Split dataset into features and target
X = df.drop(columns=['ViolentCrimesPerPop'])
y = df['ViolentCrimesPerPop']
print(X)
print(y)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_standardized = pd.DataFrame(X_scaled, columns=X.columns)
y_centered = y - y.mean()  # Center target

# 1 Least Square:
# Add constant for regression
X_const = sm.add_constant(X_standardized)
model = sm.OLS(y, X_const).fit()
significant_features = model.pvalues[model.pvalues < 0.05].index.tolist()
significant_features.remove('const')  # Remove intercept

# 2 Best Subset: use exhaustive feature selector on a smaller set of pre-selected features

# Reduce computational load by pre-selecting top 15 features via correlation
top_corr = X.corrwith(y).abs().nlargest(15).index
X_filtered = X[top_corr]

# Search subsets of size 1-10 features
efs = ExhaustiveFeatureSelector(LinearRegression(),
          min_features=1,
          max_features=10,
          scoring='r2',
          cv=0,
          print_progress=True)
efs.fit(X_filtered, y)
best_subset_features = list(efs.best_feature_names_)

# 3 Recursive Feature Elimination
rfe = RFE(LinearRegression(), n_features_to_select=10)
rfe.fit(X_standardized, y)
rfe_features = X.columns[rfe.support_].tolist()

# 4 Lasso
lasso = LassoCV(cv=5).fit(X_standardized, y)
lasso_features = X.columns[lasso.coef_ != 0].tolist()

# 5 Elastic Net
enet = ElasticNetCV(cv=5).fit(X_standardized, y)
enet_features = X.columns[enet.coef_ != 0].tolist()

# Preview result
print(
    "OLS Significance:", significant_features,
    "Best Subsets:", best_subset_features,
    "RFE:", rfe_features,
    "Lasso:", lasso_features,
    "Elastic Net:", enet_features
)

# Compile result
results = {
    "OLS Significance": significant_features,
    "Best Subsets": best_subset_features,
    "RFE": rfe_features,
    "Lasso": lasso_features,
    "Elastic Net": enet_features
}

# Aggregate feature votes from respective methods to see which features get picked the most or the least
all_features = [f for sublist in results.values() for f in sublist]
feature_counts = Counter(all_features)

# Plot feature importance consensus
plt.figure(figsize=(20, 20))
pd.Series(feature_counts).sort_values().plot(kind='barh')
plt.title('Feature Selection Consensus Across Methods')
plt.xlabel('Number of Methods Selecting Feature')
plt.ylabel('Features')
plt.show()

# Plot heatmap for showing the features picked by each method
comparison_df = pd.DataFrame(index=X_standardized.columns)
for results, features in results.items():
    comparison_df[results] = comparison_df.index.isin(features).astype(int)

plt.figure(figsize=(30, 30))
sns.heatmap(comparison_df.T, cmap='Blues')
plt.title("Feature Presence Across Methods")
plt.xlabel("Features")
plt.ylabel("Selection Methods")
plt.show()
#%%
# (a) (ii)
# Generate alpha (λ) values from 1e-3 to 1e3
alphas = np.logspace(-3, 3, 100)

# 1. Lasso Path
alphas_lasso, coefs_lasso, _ = lasso_path(X_standardized, y_centered, alphas=alphas)

# 2. Elastic Net Paths (two different α mixes, where α=0.3 (more L2), α=0.7 (more L1))
l1_ratios = [0.3, 0.7]
enet_results = []

for l1_ratio in l1_ratios:
    alphas_enet, coefs_enet, _ = enet_path(
        X_standardized, y_centered,
        l1_ratio=l1_ratio,
        alphas=alphas
    )
    enet_results.append((alphas_enet, coefs_enet))

# 3. Ridge Path
coefs_ridge = []
for alpha in alphas:
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(X_standardized, y_centered)
    coefs_ridge.append(ridge.coef_)
coefs_ridge = np.array(coefs_ridge).T

# Visualization
plt.figure(figsize=(15, 10))

# Lasso Plot
plt.subplot(221)
for coef in coefs_lasso:
    plt.plot(np.log10(alphas_lasso), coef)
plt.title("Lasso Paths\n(L1 Regularization)")
plt.xlabel("log(λ)")
plt.ylabel("Coefficient Values")

# Elastic Net Plots
for i, (l1_ratio, (alphas_enet, coefs_enet)) in enumerate(zip(l1_ratios, enet_results)):
    plt.subplot(222 + i)
    for coef in coefs_enet:
        plt.plot(np.log10(alphas_enet), coef)
    plt.title(f"Elastic Net (α={l1_ratio})\nL1/L2 Mix")
    plt.xlabel("log(λ)")
    plt.ylabel("Coefficient Values")

# Ridge Plot
plt.figure(figsize=(8, 6))
for coef in coefs_ridge:
    plt.plot(np.log10(alphas), coef)
plt.title("Ridge Paths\n(L2 Regularization)")
plt.xlabel("log(λ)")
plt.ylabel("Coefficient Values")

plt.tight_layout()
plt.show()
#%%
# Part b: Linear Method for Prediction

# Split dataset into features and target
X = df.drop(columns=['ViolentCrimesPerPop'])
y = df['ViolentCrimesPerPop']

# (b) (i) Run experiment

# Initialize results storage
results = {
    'Least Squares': [],
    'Ridge': [],
    'Best Subsets': [],
    'RFE': [],
    'Lasso': [],
    'Elastic Net': []
}

# Preprocessing function
def preprocess_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y.values.ravel()

# Experiment loop
def run_experiment(iteration):
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=iteration)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=iteration)

    # Scale data
    X_train_scaled, y_train_clean = preprocess_data(X_train, y_train)
    X_val_scaled, y_val_clean = preprocess_data(X_val, y_val)
    X_test_scaled, y_test_clean = preprocess_data(X_test, y_test)

    # Store iteration results
    iteration_results = {}

    # 1. Least Squares
    lr = LinearRegression().fit(X_train_scaled, y_train_clean)
    iteration_results['Least Squares'] = mean_squared_error(y_test_clean, lr.predict(X_test_scaled))

    # 2. Ridge Regression
    alpha_ridge = np.logspace(-3, 3, 50)
    val_scores = []
    for alpha in alpha_ridge:
        ridge = Ridge(alpha=alpha).fit(X_train_scaled, y_train_clean)
        val_scores.append(mean_squared_error(y_val_clean, ridge.predict(X_val_scaled)))
    best_alpha = alpha_ridge[np.argmin(val_scores)]
    iteration_results['Ridge'] = mean_squared_error(y_test_clean,Ridge(alpha=best_alpha).fit(X_train_scaled, y_train_clean).predict(X_test_scaled))

    # 3. Best Subsets (Limited implementation)
    corr_threshold = 0.5
    corr_values = pd.Series([np.corrcoef(X_train_scaled[:, i], y_train_clean)[0, 1]
                          for i in range(X_train_scaled.shape[1])])
    top_features = np.where(np.abs(corr_values) > corr_threshold)[0]
    print(top_features)
    sfs = SequentialFeatureSelector(LinearRegression(),
             k_features=8,
             forward=True,
             scoring='neg_mean_squared_error',
             cv=0,
             n_jobs=-1)

    sfs.fit(X_train_scaled[:, top_features], y_train_clean)
    # Map the selected indices back to the original feature indices
    best_features = top_features[list(sfs.k_feature_idx_)]

    # Train regression model with the best features from SFS
    lr_subset = LinearRegression().fit(X_train_scaled[:, best_features], y_train_clean)
    iteration_results['Best Subsets'] = mean_squared_error(y_test_clean, lr_subset.predict(X_test_scaled[:, best_features]))

    # 4. RFE
    rfe = RFE(LinearRegression(), n_features_to_select=10).fit(X_train_scaled, y_train_clean)
    iteration_results['RFE'] = mean_squared_error(y_test_clean, rfe.predict(X_test_scaled))

    # 5. Lasso
    alpha_lasso = np.logspace(-3, 1, 50)
    val_scores = []
    for alpha in alpha_lasso:
        lasso = Lasso(alpha=alpha).fit(X_train_scaled, y_train_clean)
        val_scores.append(mean_squared_error(y_val_clean, lasso.predict(X_val_scaled)))
    best_alpha = alpha_lasso[np.argmin(val_scores)]
    iteration_results['Lasso'] = mean_squared_error(y_test_clean,
                                                  Lasso(alpha=best_alpha).fit(X_train_scaled, y_train_clean).predict(X_test_scaled))

    # 6. Elastic Net
    alpha_enet = np.logspace(-3, 1, 20)
    l1_ratio = [0.3, 0.5, 0.7]
    best_score = np.inf
    best_params = {}

    for alpha in alpha_enet:
        for ratio in l1_ratio:
            enet = ElasticNet(alpha=alpha, l1_ratio=ratio).fit(X_train_scaled, y_train_clean)
            score = mean_squared_error(y_val_clean, enet.predict(X_val_scaled))
            if score < best_score:
                best_score = score
                best_params = {'alpha': alpha, 'l1_ratio': ratio}

    iteration_results['Elastic Net'] = mean_squared_error(y_test_clean,
                                                        ElasticNet(**best_params).fit(X_train_scaled, y_train_clean).predict(X_test_scaled))

    return iteration_results

# Run 10 iterations in parallel
n_iterations = 10
all_results = Parallel(n_jobs=-1)(delayed(run_experiment)(i) for i in range(n_iterations))

# Aggregate results
for result in all_results:
    for method in results:
        results[method].append(result[method])


# Calculate averages
final_results = {method: (np.mean(scores), np.std(scores)) for method, scores in results.items()}

# Display results
print("Average Test MSE ± Std Dev over 10 iterations:")
for method, (mean, std) in final_results.items():
    print(f"{method}: {mean:.4f} ± {std:.4f}")
#%%
# (b)(ii) Visualize comparison

# Organize results into a DataFrame for plotting
data = pd.DataFrame({
    'Method': ['Least Squares', 'Ridge', 'Best Subsets', 'RFE', 'Lasso', 'Elastic Net'],
    'Mean MSE': [0.0190, 0.0187, 0.0208, 0.0204, 0.0188, 0.0188],
    'Std Dev': [0.0019, 0.0019, 0.0016, 0.0020, 0.0019, 0.0019]
})
df = pd.DataFrame(data).sort_values('Mean MSE', ascending=True)

# bar plot
fig, ax = plt.subplots()
for i, (mean, std) in enumerate(zip(df['Mean MSE'], df['Std Dev'])):
    ax.text(
        mean + 0.001,
        i,
        f'{mean:.4f} ± {std:.4f}',
        ha='left',
        va='bottom',
        fontsize=10
    )
ax.barh(df['Method'], df['Mean MSE'], xerr=df['Std Dev'], color='purple', capsize=5)
ax.set_xlabel('Mean Squared Error (MSE)')
ax.set_ylabel('Method')
ax.set_title('Bar plot with error bars')
#%%
#HW2 Q2a CV
from itertools import product
from numpy.linalg import solve

def poly_kernel(X1, X2, degree):
    # Computes the polynomial kernel: (X1*X2^T + 1)^degree
    return (np.dot(X1, X2.T) + 1) ** degree

def rbf_kernel(X1, X2, gamma):
    # Computes the RBF kernel: exp(-gamma * ||x - y||^2)
    # Squared norm computed via broadcasting and using the formula |x-y|^2 = ||x||^2 + ||y||^2 - 2 x.y
    X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
    X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)
    K = X1_norm + X2_norm - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * K)

def kernel_ridge_predict(K_train, y_train, lam, K_test):
    # Train kernel ridge using closed-form: alpha = (K_train + lam*I)^{-1} y_train
    n = K_train.shape[0]
    alpha = solve(K_train + lam * np.eye(n), y_train)
    # Predictions on test set: K_test * alpha
    return np.dot(K_test, alpha)

def kfold_indices(n_samples, k):
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    return folds

def kfold_cv_kernel_ridge(X, y, k, kernel, kernel_param, lam):
    n_samples = X.shape[0]
    folds = kfold_indices(n_samples, k)
    mse_list = []

    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate(folds[:i] + folds[i+1:])
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Compute kernel matrices based on choice
        if kernel == "poly":
            K_train = poly_kernel(X_train, X_train, kernel_param)
            K_val = poly_kernel(X_val, X_train, kernel_param)
        elif kernel == "rbf":
            K_train = rbf_kernel(X_train, X_train, kernel_param)
            K_val = rbf_kernel(X_val, X_train, kernel_param)
        else:
            raise ValueError("Kernel must be either 'poly' or 'rbf'")

        # Get predictions using kernel ridge regression
        y_pred = kernel_ridge_predict(K_train, y_train, lam, K_val)
        mse = np.mean((y_val - y_pred) ** 2)
        mse_list.append(mse)

    return np.mean(mse_list)

def select_kernel_ridge_hyperparameters(X, y, k, kernel, param_grid):
    # param_grid should be a dict with keys 'kernel_param' and 'lam'
    best_score = np.inf
    best_params = None

    # iterate all combinations
    for kernel_param, lam in product(param_grid['kernel_param'], param_grid['lam']):
        score = kfold_cv_kernel_ridge(X, y, k, kernel, kernel_param, lam)
        # Uncomment the next line for progress reporting:
        # print(f"Kernel parameter: {kernel_param}, Regularization: {lam} -> MSE: {score}")
        if score < best_score:
            best_score = score
            best_params = {'kernel_param': kernel_param, 'lam': lam}
    return best_params, best_score

# Example usage:
# Split dataset into features and target
X = df.drop(columns=['ViolentCrimesPerPop'])
y = df['ViolentCrimesPerPop']

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale data
X_train_scaled, y_train_clean = preprocess_data(X_train, y_train)
X_val_scaled, y_val_clean = preprocess_data(X_val, y_val)
X_test_scaled, y_test_clean = preprocess_data(X_test, y_test)

if __name__ == "__main__":
    target_col = 'ViolentCrimesPerPop'
    X = X_train_scaled
    y = y_train_clean

    # Parameter grid for hyperparameter tuning
    param_grid_poly = {
        'kernel_param': [2, 3, 4],  # polynomial degrees
        'lam': [1e-2, 1e-1, 1, 10]
    }

    param_grid_rbf = {
        'kernel_param': [0.1, 1, 10],  # gamma values
        'lam': [1e-2, 1e-1, 1, 10]
    }

    # Choose k-folds
    k = 5

    # Select best hyperparameters for polynomial kernel
    best_poly, best_mse_poly = select_kernel_ridge_hyperparameters(X, y, k, "poly", param_grid_poly)
    print("Best polynomial kernel parameters:", best_poly, "with MSE:", best_mse_poly)

    # Select best hyperparameters for RBF kernel
    best_rbf, best_mse_rbf = select_kernel_ridge_hyperparameters(X, y, k, "rbf", param_grid_rbf)
    print("Best RBF kernel parameters:", best_rbf, "with MSE:", best_mse_rbf)
#%%
#Q2b
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

#Copy the above function here
def poly_kernel(X1, X2, degree):
    # Computes the polynomial kernel: (X1*X2^T + 1)^degree
    return (np.dot(X1, X2.T) + 1) ** degree

def rbf_kernel(X1, X2, gamma):
    # Computes the RBF kernel: exp(-gamma * ||x - y||^2)
    # Squared norm computed via broadcasting and using the formula |x-y|^2 = ||x||^2 + ||y||^2 - 2 x.y
    X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
    X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)
    K = X1_norm + X2_norm - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * K)

def kernel_ridge_predict(K_train, y_train, lam, K_test):
    # Train kernel ridge using closed-form: alpha = (K_train + lam*I)^{-1} y_train
    n = K_train.shape[0]
    alpha = solve(K_train + lam * np.eye(n), y_train)
    # Predictions on test set: K_test * alpha
    return np.dot(K_test, alpha)

def kfold_indices(n_samples, k):
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    return folds

def kfold_cv_kernel_ridge(X, y, k, kernel, kernel_param, lam):
    n_samples = X.shape[0]
    folds = kfold_indices(n_samples, k)
    mse_list = []

    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate(folds[:i] + folds[i+1:])
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Compute kernel matrices based on choice
        if kernel == "poly":
            K_train = poly_kernel(X_train, X_train, kernel_param)
            K_val = poly_kernel(X_val, X_train, kernel_param)
        elif kernel == "rbf":
            K_train = rbf_kernel(X_train, X_train, kernel_param)
            K_val = rbf_kernel(X_val, X_train, kernel_param)
        else:
            raise ValueError("Kernel must be either 'poly' or 'rbf'")

        # Get predictions using kernel ridge regression
        y_pred = kernel_ridge_predict(K_train, y_train, lam, K_val)
        mse = np.mean((y_val - y_pred) ** 2)
        mse_list.append(mse)

    return np.mean(mse_list)

def select_kernel_ridge_hyperparameters(X, y, k, kernel, param_grid):
    # param_grid should be a dict with keys 'kernel_param' and 'lam'
    best_score = np.inf
    best_params = None

    # iterate all combinations
    for kernel_param, lam in product(param_grid['kernel_param'], param_grid['lam']):
        score = kfold_cv_kernel_ridge(X, y, k, kernel, kernel_param, lam)
        # Uncomment the next line for progress reporting:
        # print(f"Kernel parameter: {kernel_param}, Regularization: {lam} -> MSE: {score}")
        if score < best_score:
            best_score = score
            best_params = {'kernel_param': kernel_param, 'lam': lam}
    return best_params, best_score

# apply on original dataset:
# Split dataset into features and target
X = df.drop(columns=['ViolentCrimesPerPop'])
y = df['ViolentCrimesPerPop']

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale data
X_train_scaled, y_train_clean = preprocess_data(X_train, y_train)
X_val_scaled, y_val_clean = preprocess_data(X_val, y_val)
X_test_scaled, y_test_clean = preprocess_data(X_test, y_test)

if __name__ == "__main__":
    target_col = 'ViolentCrimesPerPop'
    X = X_train_scaled
    y = y_train_clean

    # Parameter grid for hyperparameter tuning
    param_grid_poly = {
        'kernel_param': [2, 3, 4],  # polynomial degrees
        'lam': [1e-2, 1e-1, 1, 10]
    }

    param_grid_rbf = {
        'kernel_param': [0.1, 1, 10],  # gamma values
        'lam': [1e-2, 1e-1, 1, 10]
    }

    # Choose k-folds
    k = 5

    # Select best hyperparameters for polynomial kernel
    best_poly, best_mse_poly = select_kernel_ridge_hyperparameters(X, y, k, "poly", param_grid_poly)

    # Select best hyperparameters for RBF kernel
    best_rbf, best_mse_rbf = select_kernel_ridge_hyperparameters(X, y, k, "rbf", param_grid_rbf)

# Tune hyperparameters for a polynomial kernel using the custom function
param_grid_poly = {
    'kernel_param': [2, 3, 4],  # polynomial degrees
    'lam': [1e-2, 1e-1, 1, 10]
}
k = 5
custom_best_poly, custom_mse_poly = select_kernel_ridge_hyperparameters(X, y, k, "poly", param_grid_poly)
print("Custom CV - Polynomial kernel best parameters:", custom_best_poly, "MSE:", custom_mse_poly)

# Tune hyperparameters for an RBF kernel using the custom function
param_grid_rbf = {
    'kernel_param': [0.1, 1, 10],  # gamma values
    'lam': [1e-2, 1e-1, 1, 10]
}
custom_best_rbf, custom_mse_rbf = select_kernel_ridge_hyperparameters(X, y, k, "rbf", param_grid_rbf)
print("Custom CV - RBF kernel best parameters:", custom_best_rbf, "MSE:", custom_mse_rbf)

# Now use sklearn's GridSearchCV with KernelRidge for comparison.
# For the polynomial kernel, sklearn's KernelRidge uses parameters: alpha (regularization) and degree for the kernel.
param_grid_sklearn_poly = {
    'alpha': [1e-2, 1e-1, 1, 10],
    'degree': [2, 3, 4],
    'kernel': ['polynomial']
}
kr_poly = KernelRidge()
grid_search_poly = GridSearchCV(kr_poly, param_grid_sklearn_poly, cv=5, scoring='neg_mean_squared_error')
grid_search_poly.fit(X, y)
print("sklearn CV - Polynomial kernel best params:", grid_search_poly.best_params_, "MSE:", -grid_search_poly.best_score_)

# For the RBF kernel, sklearn's KernelRidge uses parameters: alpha, gamma (kernel parameter) and kernel.
param_grid_sklearn_rbf = {
    'alpha': [1e-2, 1e-1, 1, 10],
    'gamma': [0.1, 1, 10],
    'kernel': ['rbf']
}
kr_rbf = KernelRidge()
grid_search_rbf = GridSearchCV(kr_rbf, param_grid_sklearn_rbf, cv=5, scoring='neg_mean_squared_error')
grid_search_rbf.fit(X, y)
print("sklearn CV - RBF kernel best params:", grid_search_rbf.best_params_, "MSE:", -grid_search_rbf.best_score_)