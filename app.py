import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Data
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")  # Load your movies.csv file
    movies['genres'] = movies['genres'].str.replace("|", " ")  # Convert genres for TF-IDF
    return movies

movies = load_data()

# Compute TF-IDF and Cosine Similarity
@st.cache_data
def compute_similarity(movies):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim = compute_similarity(movies)

# Recommendation Function
def recommend_movie(movie_title, top_n=5):
    movies['title'] = movies['title'].str.strip().str.lower()
    movie_title = movie_title.strip().lower()

    if movie_title not in movies['title'].values:
        return ["Movie not found. Please check the spelling."]

    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    idx = indices[movie_title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.write("Enter a movie title to get similar movie recommendations.")

# Input from user
movie_input = st.text_input("Enter a Movie Title", "Toy Story (1995)")
top_n = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

if st.button("Recommend"):
    recommendations = recommend_movie(movie_input, top_n)
    if recommendations:
        st.subheader("Recommended Movies 🎥")
        for i, movie in enumerate(recommendations, start=1):
            st.write(f"{i}. {movie}")
