import pickle

# Load model
with open('models/spam_classifier.pkl', 'rb') as f:
    data = pickle.load(f)

model      = data['model']
vectorizer = data['vectorizer']

def predict_message(message):
    vec   = vectorizer.transform([message])
    pred  = model.predict(vec)[0]
    prob  = model.predict_proba(vec)[0]
    label = "SPAM" if pred == 1 else "HAM"
    conf  = prob[pred] * 100
    print(f"[{label}] ({conf:.1f}%) → {message[:60]}")

# Test examples
predict_message("Congratulations! You won a FREE iPhone. Claim NOW!")
predict_message("Hey, are we still on for lunch tomorrow at 1pm?")
predict_message("URGENT: Your account is compromised. Call us now!")
predict_message("The meeting got moved to Thursday, FYI.")

# Try your own
print("\n--- Try your own ---")
msg = input("Enter a message: ")
predict_message(msg)