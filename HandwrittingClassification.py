#%%
from statsmodels.graphics.tukeyplot import results
!pip install group_lasso
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from group_lasso import GroupLasso
#%%
#Load MNIST dataset and convert it into a pandas dataframe
mnist = fetch_openml("mnist_784")
df = pd.concat([mnist['data'],mnist['target']],axis=1)

# filter digits 3, 5, and 8
df = df[(df['class']=='3') | (df['class']=='5') | (df['class']=='8')]
#%%
# Convert target to integer
df['target'] = df['class'].astype(int)

# Split features and target
X = df.drop('target', axis=1).values
y = df['target'].values

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
# Create pipelines for methods needing standardized data
# Logistic Regression (One-vs-Rest)
lr_ovr = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(multi_class='ovr', max_iter=1000, random_state=42))
])
# Multinomial Regression (use LogisticRegression with multi_class='multinomial')
multinomial = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42))
])
# Linear SVM (One-vs-Rest)
svm_ovr = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LinearSVC(multi_class='ovr', max_iter=1000, random_state=42))
])

# Naive Bayes and Linear Discriminant Analysis might not need scaling
nb = GaussianNB()
lda = LinearDiscriminantAnalysis()

# Dictionary of classifiers
classifiers = {
    "Logistic Regression (OvR)": lr_ovr,
    "Multinomial Regression": multinomial,
    "Naive Bayes": nb,
    "Linear Discriminant Analysis": lda,
    "Linear SVM (OvR)": svm_ovr
}
#%%
# Train each classifier and evaluate test accuracy
results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {
        "accuracy": acc,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[3, 5, 8]),
        "predictions": y_pred
    },
    print(f"{name}: Accuracy = {acc:.4f}")
#%%
# Compute the confusion matrix
labels = [3, 5, 8]
cm = confusion_matrix(y_test, y_pred, labels=labels)
print("Confusion Matrix:")
print(cm)

# Visualize the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Digit")
plt.ylabel("True Digit")
plt.title("Confusion Matrix for MNIST Digits 3, 5, 8")
plt.show()
#%%
# Determine misclassification counts for each digit
misclassifications = {}
for i, label in enumerate(labels):
    # Misclassifications = row sum minus the diagonal element
    misclass = np.sum(cm[i]) - cm[i, i]
    misclassifications[label] = misclass
    print(f"Digit {label} is misclassified {misclass} times.")

# Identify which digit is most often misclassified
most_misclassified = max(misclassifications, key=misclassifications.get)
print(f"\nDigit most often misclassified: {most_misclassified}")
#%%
# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert y to one-hot encoding
encoder = OneHotEncoder(sparse_output=False)
y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_onehot = encoder.transform(y_test.reshape(-1, 1))
n_classes = y_train_onehot.shape[1]
n_features = X_train.shape[1]

# Initialize weight matrix W (no intercept for simplicity)
W = np.zeros((n_features, n_classes))

# Define softmax function
def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

# Choose hyperparameters
learning_rate = 1e-2
n_iter = 500
lmbda = 0.1  # regularization strength

# Function to compute the negative log-likelihood
def compute_loss(W, X, y_onehot, lmbda):
    logits = X.dot(W)
    probs = softmax(logits)
    # Avoid log(0)
    loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-8), axis=1))
    # Group-lasso penalty: sum of L2 norms for each row
    penalty = lmbda * np.sum(np.linalg.norm(W, axis=1))
    return loss + penalty

# Proximal operator for group lasso (row-wise)
def prox_operator(W, thresh):
    W_new = np.zeros_like(W)
    for i in range(W.shape[0]):
        norm_i = np.linalg.norm(W[i])
        if norm_i > thresh:
            W_new[i] = (1 - thresh / norm_i) * W[i]
        else:
            W_new[i] = 0.0
    return W_new

# Proximal gradient descent
loss_history = []
for it in range(n_iter):
    # Compute gradient (only considering the differentiable part)
    logits = X_train.dot(W)
    probs = softmax(logits)
    grad_loss = X_train.T.dot(probs - y_train_onehot) / X_train.shape[0]

    # Gradient descent step (without penalty)
    W_temp = W - learning_rate * grad_loss

    # Proximal step: apply group lasso penalty row-wise
    # thresh = learning_rate * lmbda
    W = prox_operator(W_temp, learning_rate * lmbda)

    # record the loss for monitoring
    if it % 50 == 0:
        loss = compute_loss(W, X_train, y_train_onehot, lmbda)
        loss_history.append(loss)
        print("Iteration %d, Loss: %.4f" % (it, loss))

# Evaluate accuracy on test set using the learned W
logits_test = X_test.dot(W)
probs_test = softmax(logits_test)
y_pred = np.argmax(probs_test, axis=1)
# compare with original y_test (3, 5, 8 as int)
acc = accuracy_score(y_test, encoder.categories_[0].astype(int)[y_pred])
print("\nTest Accuracy: %.4f" % acc)
#%%
# Compute coefficient norms across classes for each feature
# Remove the intercept row
W_no_intercept = W[:-1, :]
coef_norm = np.linalg.norm(W_no_intercept, axis=1)

# Get the feature names (pixel indices)
feature_names = df.drop('target', axis=1).columns

# Set a threshold to determine if the feature is selected (e.g., 1e-3)
threshold = 1e-3
selected_features = np.where(coef_norm > threshold)[0]

# Create dataframe for showing the feature names and their importance
selected_features_df = pd.DataFrame({
    'Feature': feature_names[selected_features],
    'Importance': coef_norm[selected_features]
})
selected_features_df = selected_features_df.sort_values(by='Importance', ascending=False)

# Print selected feature indices
print("Selected feature indices: \n", selected_features_df)

# Visualize the coefficient norms as an image (28x28)
coef_image = coef_norm.reshape(28, 28)
plt.figure(figsize=(6, 6))
plt.imshow(coef_image, cmap='viridis')
plt.colorbar()
plt.title('Coefficient Norms (Group-Lasso Feature Importance)')
plt.axis('off')
plt.show()