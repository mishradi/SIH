import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

candidates_df = pd.read_csv("candidates.csv")
internship_df = pd.read_csv("internship.csv")

#cleaing

stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z\s]','',text, re.I|re.A)
    text = text.lower()
    tokens = text.split()
    filtered_token = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_token)

candidates_df["processed_resume"] = candidates_df['resume'].apply(preprocess_text)

internship_df["processed_discription"] = internship_df['job_description'].apply(preprocess_text)


#modeling

tfidf_vectorizer = TfidfVectorizer(max_features=5000)

internship_matrix = tfidf_vectorizer.fit_transform(internship_df["processed_discription"])
candidates_matrix = tfidf_vectorizer.transform(candidates_df["processed_resume"])

matchscores = cosine_similarity(candidates_matrix, internship_matrix)

def recommendation_internship(candidate_id, n):
    
    candidate_index = candidates_df[candidates_df['candidate_id'] == candidate_id].index[0]
    candidate_score = matchscores[candidate_index]
    top_indices = np.argsort(candidate_score)[-n:][::-1]
    top_internship = internship_df.iloc[top_indices]
    top_scores = candidate_score[top_indices]
    recommendation_df = top_internship.copy()
    recommendation_df["matchscore"] = top_scores
    return recommendation_df[['Company_name', 'job_title', 'matchscore','job_description']]




with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
    

with open('internship_tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(internship_matrix, f)

print("\nModel artifacts saved successfully.")


print(recommendation_internship(122, 10))
