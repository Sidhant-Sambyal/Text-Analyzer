# Text Classifier — Scikit-learn & TF-IDF

A spam detection classifier built with **Naive Bayes + TF-IDF** using scikit-learn. Classifies SMS messages as **SPAM** or **HAM** with ~98% accuracy.

---

## What it does

```
Your text
    ↓
TF-IDF Vectorizer   → converts words to weighted numeric scores
    ↓
MultinomialNB       → calculates P(spam | words) for each word
    ↓
Prediction          → SPAM / HAM + confidence score
```

---

## Project Structure

```
text-classifier/
├── models/              ← saved model is stored here after training
├── data/                ← drop custom datasets here
├── classifier.py        ← entry point ✅ — train & save the model
├── evaluate.py          ← accuracy, precision, recall, F1 report
├── predict.py           ← predict on new messages + interactive input
├── compare_models.py    ← compare Naive Bayes vs Logistic Regression vs SVM
└── requirements.txt     ← all dependencies
```

---

## Setup

### 1. Create a virtual environment

```bash
# Mac / Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run Order

| Step | Command | What it does |
|------|---------|--------------|
| 1 | `python classifier.py` | Entry point — trains model, saves to `models/` |
| 2 | `python evaluate.py` | Accuracy + classification report |
| 3 | `python predict.py` | Test on sample messages + type your own |
| 4 | `python compare_models.py` | Compare 3 ML algorithms side by side |

---

## Example Output

```bash
$ python classifier.py

Dataset shape: (5572, 2)

Label counts:
ham     4825
spam     747

Train shape: (4457, 5000)
Test shape:  (1115, 5000)

Model trained!
Model saved to models/spam_classifier.pkl
```

```bash
$ python predict.py

[SPAM] (99.8%) → Congratulations! You won a FREE iPhone. Claim NOW!
[HAM]  (99.2%) → Hey, are we still on for lunch tomorrow at 1pm?
[SPAM] (97.4%) → URGENT: Your account is compromised. Call us now!
[HAM]  (98.6%) → The meeting got moved to Thursday, FYI.
```

```bash
$ python compare_models.py

Model                     Accuracy  Spam Recall
--------------------------------------------------
Naive Bayes                 0.9785       0.8800
Logistic Regression         0.9857       0.9200
Linear SVM                  0.9839       0.9133
```

---

## How TF-IDF Works

**TF-IDF = Term Frequency × Inverse Document Frequency**

- **TF** — how often a word appears in this message
- **IDF** — how rare the word is across ALL messages
- Words like "the", "is", "a" appear everywhere → low IDF → low score
- Words like "winner", "free", "urgent" are rare but spam-specific → high score

```python
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)  # learn vocab + transform
X_test_tfidf  = vectorizer.transform(X_test)        # only transform — no fit!
```

> Never call `fit_transform()` on test data — that's **data leakage**.

---

## Key Concepts for AI Engineer Interviews

| Concept | One-liner |
|---|---|
| TF-IDF | Scores words by frequency × rarity across documents |
| Naive Bayes | Assumes word independence — calculates P(spam \| word) |
| Data leakage | Fitting on test data lets the model cheat — always avoid |
| Stratified split | Keeps class ratio equal in train and test sets |
| Precision | Of predicted spam, how many were actually spam? |
| Recall | Of all real spam, how many did we catch? |
| F1 Score | Harmonic mean of precision and recall |
| Confusion matrix | Table of TP, TN, FP, FN predictions |

---

## Interview Questions This Project Prepares You For

- *"What is TF-IDF and why is it used for text?"*
- *"Why is Naive Bayes good for text classification?"*
- *"What is data leakage and how do you prevent it?"*
- *"Why is accuracy not enough for imbalanced datasets?"*
- *"When would you use precision vs recall?"*
- *"How does a confusion matrix help evaluate a classifier?"*

---

## Next Steps

- [ ] Swap dataset — try product reviews or news categories
- [ ] Replace TF-IDF + Naive Bayes with HuggingFace DistilBERT
- [ ] Build a Flask API to serve predictions via REST
- [ ] Deploy to Railway or Render (free hosting)
- [ ] Add a simple React frontend

---

## Dataset

- **Name:** SMS Spam Collection
- **Source:** UCI Machine Learning Repository via [justmarkham/pycon-2016-tutorial](https://github.com/justmarkham/pycon-2016-tutorial)
- **Size:** 5,572 SMS messages
- **Labels:** ham (4,825) · spam (747)
- **License:** Public domain

---

## Dependencies

```
scikit-learn>=1.4.0    # TF-IDF, Naive Bayes, metrics
pandas>=2.2.0          # data loading and manipulation
numpy>=1.26.0          # numerical operations
matplotlib>=3.8.0      # optional: visualise results
```
