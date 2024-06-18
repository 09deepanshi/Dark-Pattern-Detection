import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sqlite3
from sklearn.ensemble import IsolationForest

# Load the pickled model
with open('anomaly_detector_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Function to detect fake reviews based on keywords count
def rule_based_detection(review, keywords):
    count = 0
    for word in review.split():
        if word.lower() in keywords:
            count += 1
    return count > 2  # Adjust the threshold as needed

# Define keywords indicating fake reviews
fake_keywords = ['not', 'disappointed', 'waste', 'terrible', 'poor', 'avoid', 'horrible', 'worst', 'cheap', 'junk',
                 'trash', 'awful', 'terrible', 'useless', 'disappointing', 'bad', 'defective', 'ruined', 'flimsy',
                 'garbage', 'unsatisfactory', 'shoddy', 'faulty', 'disgusting', 'regret', 'unsatisfied', 'crap',
                 'rubbish', 'deceptive', 'subpar', 'overpriced', 'inferior', 'unusable', 'disappoint', 'lies',
                 'displeased', 'stupid', 'not happy', 'shameful', 'unsatisfying', 'sucks', 'hate', 'unreliable',
                 'unacceptable', 'fail', 'lousy', 'poorly', 'dissatisfied']

# Function to save ratings in SQLite database
def save_rating(rating):
    conn = sqlite3.connect('ratings.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS ratings (id INTEGER PRIMARY KEY, rating INTEGER)''')
    c.execute("INSERT INTO ratings (rating) VALUES (?)", (rating,))
    conn.commit()
    conn.close()

def main():
    st.title("Fake Review Detection App")
    
    menu = ["Home", "Detect", "Feedback"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.subheader("Home")
        st.write("This is a simple fake review detection app.")
    
    elif choice == "Detect":
        st.subheader("Detect Fake Reviews")
        review_text = st.text_area("Enter the review text")
        if st.button("Detect"):
            # Apply rule-based detection
            fake_by_rule = rule_based_detection(review_text, fake_keywords)
            # Feature extraction
            X = pd.DataFrame([len(review_text)], columns=['length'])
            # Predict anomalies
            anomaly_score = loaded_model.decision_function(X)
            anomaly = loaded_model.predict(X)
            if fake_by_rule or anomaly == -1:
                st.write("This review is likely to be fake.")
            else:
                st.write("This review is likely to be genuine.")
    
    elif choice == "Feedback":
        st.subheader("Leave Feedback")
        st.write("Please leave your feedback by selecting a star rating below:")
        feedback = st.slider("Star Rating", min_value=1, max_value=5)
        st.write("You've rated our service with:", feedback, "stars.")
        # Save the rating to the database
        save_rating(feedback)

if __name__ == "__main__":
    main()
