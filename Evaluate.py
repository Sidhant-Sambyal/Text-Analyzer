import pickle
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load model
with open('models/spam_classifier.pkl', 'rb') as f:
    data = pickle.load(f)

model      = data['model']
vectorizer = data['vectorizer']

# Re-load test set (same split)
import pandas as pd
from sklearn.model_selection import train_test_split

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df  = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})

_, X_test, _, y_test = train_test_split(
    df['message'], df['label_num'], test_size=0.2, random_state=42, stratify=df['label_num']
)

X_test_tfidf = vectorizer.transform(X_test)
y_pred       = model.predict(X_test_tfidf)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print("             Pred Ham  Pred Spam")
print(f"Actual Ham  {cm[0][0]:8d}  {cm[0][1]:9d}")
print(f"Actual Spam {cm[1][0]:8d}  {cm[1][1]:9d}")