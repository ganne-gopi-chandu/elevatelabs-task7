import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions

# Load dataset
df = pd.read_csv("breast-cancer.csv")

# Drop the ID column (not useful for classification)
df = df.drop(columns=['id'])

# Encode diagnosis (target variable)
label_encoder = LabelEncoder()
df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])  # 0 for Benign, 1 for Malignant

# Select two features for decision boundary visualization
feature_cols = ['radius_mean', 'texture_mean']
X_vis = df[feature_cols]  # Only select these for visualization
y_vis = df['diagnosis']

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop(columns=['diagnosis']))  # Standardizing all features
X_vis_scaled = scaler.fit_transform(X_vis)  # Scaled 2D feature subset for plotting

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['diagnosis'], test_size=0.2, random_state=42)
X_vis_train, X_vis_test, y_vis_train, y_vis_test = train_test_split(X_vis_scaled, y_vis, test_size=0.2, random_state=42)

print("\nData Preparation Complete.")
print(f"Training Samples: {len(X_train)}, Testing Samples: {len(X_test)}")

# Train SVM with Linear Kernel
svm_linear = SVC(kernel="linear", C=1)
svm_linear.fit(X_train, y_train)

# Train SVM with RBF Kernel
svm_rbf = SVC(kernel="rbf", C=1, gamma="scale")
svm_rbf.fit(X_train, y_train)

print("SVM training complete for both kernels!")

# Convert y_vis_train to NumPy array before using plot_decision_regions
plt.figure(figsize=(6, 5))
plot_decision_regions(X_vis_train, y_vis_train.to_numpy(), clf=SVC(kernel="linear", C=1).fit(X_vis_train, y_vis_train))
plt.title("SVM Decision Boundary (Linear Kernel)")
plt.show()

plt.figure(figsize=(6, 5))
plot_decision_regions(X_vis_train, y_vis_train.to_numpy(), clf=SVC(kernel="rbf", C=1, gamma="scale").fit(X_vis_train, y_vis_train))
plt.title("SVM Decision Boundary (RBF Kernel)")
plt.show()

# Hyperparameter tuning for RBF kernel
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'gamma': ['scale', 0.1, 1, 10]  # Kernel coefficient
}

# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Display best parameters
print("\nBest Hyperparameters:", grid_search.best_params_)

# Evaluate the best model using cross-validation
best_model = grid_search.best_estimator_
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)


cv_scores_vis = cross_val_score(svm_linear, X_vis_train, y_vis_train.to_numpy(), cv=5)
accuracy_vis_percentage = cv_scores_vis.mean() * 100
print(f"Cross-Validation Accuracy (2D Features - Linear Kernel): {accuracy_vis_percentage:.2f}%")
cv_scores_vis_rbf = cross_val_score(svm_rbf, X_vis_train, y_vis_train.to_numpy(), cv=5)
accuracy_vis_rbf_percentage = cv_scores_vis_rbf.mean() * 100
print(f"Cross-Validation Accuracy (2D Features - RBF Kernel): {accuracy_vis_rbf_percentage:.2f}%")


# Train the final SVM model with best hyperparameters
best_svm = SVC(kernel="rbf", C=1, gamma='scale')
best_svm.fit(X_train, y_train)

# Evaluate accuracy on test set
test_accuracy = best_svm.score(X_test, y_test) * 100
print(f"Final Test Accuracy with Best Hyperparameters: {test_accuracy:.2f}%")


# Train a separate SVM model using only 2 selected features for visualization (with gamma='scale')
svm_vis_rbf = SVC(kernel="rbf", C=1, gamma="scale")  # Explicitly setting gamma
svm_vis_rbf.fit(X_vis_train, y_vis_train.to_numpy())  # Train on 2D data

# Define a mesh grid for plotting decision boundaries
x_min, x_max = X_vis_train[:, 0].min() - 1, X_vis_train[:, 0].max() + 1
y_min, y_max = X_vis_train[:, 1].min() - 1, X_vis_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Predict decision boundary using the newly trained 2D SVM model with RBF kernel
Z = svm_vis_rbf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot RBF Kernel Decision Boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
plt.scatter(X_vis_train[:, 0], X_vis_train[:, 1], c=y_vis_train.to_numpy(), edgecolors='k', cmap="coolwarm")
plt.xlabel("Radius Mean")
plt.ylabel("Texture Mean")
plt.title("SVM Decision Boundary (RBF Kernel - gamma='scale')")
plt.show()

