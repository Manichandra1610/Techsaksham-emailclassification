import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('spam.csv')

# Remove non-numeric values from the target column
df = df[pd.to_numeric(df['target'], errors='coerce').notnull()]

# Convert the target variable to integers
df['target'] = df['target'].astype(int)

# Split the dataset into features and target
X = df['text']
y = df['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data into numerical data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test_vec)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Function to classify new emails
def classify_email(email_content):
    email_vec = vectorizer.transform([email_content])
    prediction = clf.predict(email_vec)
    return 'Spam' if prediction[0] == 1 else 'Not Spam'

# Example usage
print("Welcome to the Email Classification System (Spam or Not Spam)!")
print("Type 'quit' at any time to exit.\n")

while True:
    user_email = input("Enter the email content for classification (or type 'quit' to exit): ")
    if user_email.strip().lower() == 'quit':
        print("\nThank you for using the Email Classification System. Goodbye!")
        break
    
    if not user_email.strip():
        print("\nPlease enter valid email content. Empty input cannot be classified.")
        continue
    
    classification = classify_email(user_email)
    print(f"\nThe email is classified as: {classification}\n")

