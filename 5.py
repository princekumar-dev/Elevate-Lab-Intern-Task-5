import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Loading dataset...")
df = pd.read_csv('heart.csv')

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Decision Tree...")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

plt.figure(figsize=(15, 8))
plot_tree(dt, feature_names=X.columns, class_names=['No Disease', 'Disease'], filled=True, max_depth=3)
plt.title("Decision Tree Visualization (Max Depth=3)")
plt.show()

print("\nChecking for overfitting...")
train_preds = dt.predict(X_train)
test_preds = dt.predict(X_test)
print("Unconstrained Tree - Train Accuracy:", accuracy_score(y_train, train_preds))
print("Unconstrained Tree - Test Accuracy:", accuracy_score(y_test, test_preds))

print("Controlling tree depth to fix overfitting...")
dt_controlled = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
dt_controlled.fit(X_train, y_train)
print("Controlled Tree - Train Accuracy:", accuracy_score(y_train, dt_controlled.predict(X_train)))
print("Controlled Tree - Test Accuracy:", accuracy_score(y_test, dt_controlled.predict(X_test)))

print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_test_preds = rf.predict(X_test)
print("Random Forest - Test Accuracy:", accuracy_score(y_test, rf_test_preds))

print("\nFeature Importances (Random Forest):")
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

for i in range(X.shape[1]):
    print(f"{i + 1}. {X.columns[indices[i]]}: {importances[indices[i]]:.4f}")

plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=45)
plt.title("Feature Importances")
plt.tight_layout()
plt.show()

print("\nEvaluating with Cross-Validation...")
dt_cv_scores = cross_val_score(dt_controlled, X, y, cv=5)
rf_cv_scores = cross_val_score(rf, X, y, cv=5)

print("Decision Tree CV Accuracy:", dt_cv_scores.mean())
print("Random Forest CV Accuracy:", rf_cv_scores.mean())

print("\nDone!")
