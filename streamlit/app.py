import streamlit as st
import pandas as pd
import numpy as np
import pickle
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

def main():
    st.title("Fake Review Detection App")
    
    menu = ["Home", "Detect"]
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
                
if __name__ == "__main__":
    main()
