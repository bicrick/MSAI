from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
import pandas as pd

# Create a DataFrame for the training data
data = {
    'X': [0, 0, 0, 0, 1, 1, 1, 1],
    'Y': [0, 0, 1, 1, 0, 0, 1, 1],
    'Z': [0, 1, 0, 1, 0, 1, 0, 1],
    'Positive': [10, 25, 35, 35, 5, 30, 10, 15],
    'Negative': [20, 5, 15, 5, 15, 10, 10, 5]
}

df = pd.DataFrame(data)

# Define features and target variable
X = df[['X', 'Y', 'Z']]
y = df['Positive'] > df['Negative']  # Binary classification problem

# Create a decision tree classifier with max_depth=2
clf = DecisionTreeClassifier(max_depth=2, criterion='gini')
clf.fit(X, y)

# Predict on the training data
y_pred = clf.predict(X)

# Calculate overall accuracy
accuracy = accuracy_score(y, y_pred)

print("Overall Accuracy:", accuracy)

# After fitting the classifier, add these lines:
tree_rules = export_text(clf, feature_names=['X', 'Y', 'Z'])
print("Decision Tree Structure:")
print(tree_rules)
