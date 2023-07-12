import streamlit as st
import joblib
import nltk
import nltk.corpus
import nltk.stem.porter as porter
import numpy as np
import pandas as pd
import neattext.functions as nfx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

"""


# Streamlit app
def main():
    # Title and description
    st.title("Answer Prediction App")
    st.write("Enter your question and the app will predict the answer.")

    # User input
    user_question = st.text_input("Enter your question:")
    if user_question:
        # Predict answer
        answer = pipe.predict([user_question])
        rating = answer[0]  # Assuming the predicted answer is the rating
        
        # Print stars based on the rating
        stars = print_stars(rating)

        # Display the predicted rating and stars
        st.subheader("Predicted Rating:")
        st.write(rating)
        st.subheader("Stars:")
        st.write(stars)

# Run the app
if __name__ == "__main__":
    main()


"""

# Core Pkgs
import streamlit as st 
import altair as alt
import plotly.express as px 

# EDA Pkgs
import pandas as pd 
import numpy as np 
from datetime import datetime



# Track Utils
from track_utils import add_rating_details,create_rating_table


# Load the pre-trained model
model = joblib.load("emotion_classifier_pipe.pkl")

# Define the preprocessing steps
preprocessing_steps = [
    ('remove_userhandles', nfx.remove_userhandles),
    ('remove_stopwords', nfx.remove_stopwords),
    # Add more preprocessing steps if needed
]

# Create the pipeline with preprocessing and model
pipe = Pipeline([
    ('preprocess', Pipeline(preprocessing_steps)),
    ('cv', CountVectorizer()),
    ('lr', LogisticRegression())
])



# Function to print stars based on the predicted rating
def print_stars(rating):
    num_stars = int(rating)
    stars = '⭐' * num_stars
    return stars


# Fxn
def get_prediction_proba(docx):
	results = pipe.predict_proba([docx])
	return results

# Main Application
def main():
    # Title and description
    st.title("Answer Prediction App")
    st.write("Enter your question and the app will predict the answer.")
    create_rating_table()
    # User input
    user_question = st.text_input("Enter your question:")
    if user_question:
        # Predict answer
        answer = pipe.predict([user_question])
        rating = int(answer)  # Assuming the predicted answer is the rating
        probability = get_prediction_proba(user_question)
        # Print stars based on the rating
        stars = print_stars(rating)


        porbability = np.max(probability)
        add_rating_details(user_question,rating,probability)
    
        # Display the predicted rating and stars
        st.subheader("Predicted Rating:")
        st.write(rating)
        st.subheader("Stars:")
        st.write(stars)
        st.write("Confidence:{}".format(np.max(probability)))
    





if __name__ == '__main__':
	main()