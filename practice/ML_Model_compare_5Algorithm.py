import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Load the wine dataset
wine_data = load_wine()
X = wine_data.data  # Features
y = wine_data.target  # Labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the classifiers
models = {
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'Naive Bayes':GaussianNB(),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}

# Store cross-validation accuracies
cv_accuracies = []

# Perform cross-validation for each model and collect accuracies
for model_name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
    cv_accuracies.append(cv_scores)

# Create a DataFrame for the cross-validation results
cv_accuracy_df = pd.DataFrame(cv_accuracies).transpose()
cv_accuracy_df.columns = models.keys()

# Plot the boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=cv_accuracy_df)
plt.title('Comparison of ML Models for Wine Dataset (Accuracy)')
plt.ylabel('Accuracy')
plt.xlabel('Models')
plt.show()

# Find the model with the best mean accuracy
mean_accuracies = cv_accuracy_df.mean()
best_model = mean_accuracies.idxmax()
print(f"The best model is: {best_model} with an average accuracy of {mean_accuracies[best_model]:.4f}")

