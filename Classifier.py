import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

print("Dataset shape:", df.shape)
print("\nLabel counts:")
print(df['label'].value_counts())

# Encode labels
df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})

X = df['message']
y = df['label_num']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF vectorisation
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

print("\nTrain shape:", X_train_tfidf.shape)
print("Test shape: ", X_test_tfidf.shape)

# Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
print("\nModel trained!")

# Save for use in other scripts
import pickle
with open('models/spam_classifier.pkl', 'wb') as f:
    pickle.dump({'model': model, 'vectorizer': vectorizer}, f)

print("Model saved to models/spam_classifier.pkl")