import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import joblib

# Function to detect fake reviews based on keywords count
def rule_based_detection(review, keywords):
    count = 0
    for word in review.split():
        if word.lower() in keywords:
            count += 1
    return count > 2  # Adjust the threshold as needed

# Load the trained SVM model
model = joblib.load('svm.pkl')

# Define keywords indicating fake reviews
fake_keywords = ['not', 'disappointed', 'waste', 'terrible', 'poor', 'avoid', 'horrible', 'worst', 'cheap', 'junk',
                 'trash', 'awful', 'terrible', 'useless', 'disappointing', 'bad', 'defective', 'ruined', 'flimsy',
                 'garbage', 'unsatisfactory', 'shoddy', 'faulty', 'disgusting', 'regret', 'unsatisfied', 'crap',
                 'rubbish', 'deceptive', 'subpar', 'overpriced', 'inferior', 'unusable', 'disappoint', 'lies',
                 'displeased', 'stupid', 'not happy', 'shameful', 'unsatisfying', 'sucks', 'hate', 'unreliable',
                 'unacceptable', 'fail', 'lousy', 'poorly', 'dissatisfied']

# Streamlit app
def main():
    st.title("Fake Review Detection")

    # Text input for user to enter their review
    review_input = st.text_area("Enter your review here:")

    # Button to trigger review detection
    if st.button("Detect Fake Review"):
        # Prepare input data for prediction
        review_df = pd.DataFrame({'text_': [review_input]})
        
        # Use the SVM model to predict if the review is fake
        predicted_label = model.predict(review_df['text_'])[0]

        if predicted_label == 'CG':
            st.error("This review is detected as fake.")

            # Apply rule-based detection
            is_fake_by_rule = rule_based_detection(review_input, fake_keywords)
            if is_fake_by_rule:
                st.write("Detected as fake by rule-based method.")

            # Apply anomaly-based detection
            # Feature extraction
            review_length = len(review_input)
            
            # Predict anomalies
            anomaly_score = model.decision_function([[review_length]])
            if anomaly_score < 0:
                st.write("Detected as fake by anomaly-based method.")

        else:
            st.success("This review is not detected as fake.")

if __name__ == "__main__":
    main()
