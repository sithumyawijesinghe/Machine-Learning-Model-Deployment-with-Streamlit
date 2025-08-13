import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score


# 1. Load and Prepare Data
print("Script started")

df = pd.read_csv("data/train.csv")

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Encode categorical variables
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Feature selection
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df['Survived']

# 2. Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 3. Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("Model trained")


# 4. Evaluation

y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_acc:.4f}")

cv_scores = cross_val_score(model, X, y, cv=5)
print(f" Cross-Validation Mean Score: {cv_scores.mean():.4f}")
print(f" Cross-Validation Scores: {cv_scores}")


# 5. Feature Importance

feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n Feature Importance:")
print(feature_importances.to_string(index=False))


# 6. Save Model

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n Model saved successfully as model.pkl")


# 7. Final Results Summary

print("\n Final Model Performance Summary")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Cross-Validation Mean Accuracy: {cv_scores.mean():.4f}")
print("\nFeature Importance:")
print(feature_importances.to_string(index=False))
print("\nScript completed successfully")
import sys  