import pandas as pd
import pickle
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommendation System", page_icon="🎞", layout="wide")
st.title("Movie Recommendation System")

def set_bg_hack(main_bg):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url({main_bg});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

main_bg = "https://cdn.pixabay.com/photo/2017/11/24/10/43/ticket-2974645_1280.jpg"
set_bg_hack(main_bg)

@st.cache_resource
def load_data():
    with open('movie_dict.pkl', 'rb') as f:
        movie_dict = pickle.load(f)
    movies = pd.DataFrame(movie_dict)
    with open('similarity.pkl', 'rb') as f:
        similarity = pickle.load(f)
    return movies, similarity

movies, similarity = load_data()

def get_recommendations(selected_movie):
    if selected_movie not in movies['Title'].values:
        return None
    
    movie_index = movies[movies['Title'] == selected_movie].index[0]
    distances = similarity[movie_index]
    recommended_indices = distances.argsort()[-8:-1][::-1]
    return movies.iloc[recommended_indices]

st.write("Select a movie from the dropdown menu below, and click '🔍 Recommend' to get recommendations!")

selected_movie_name = st.selectbox(
    'What type of movie do you want me to recommend?',
    movies['Title'].values
)

if st.button('🔍 Recommend'):
    recommendations = get_recommendations(selected_movie_name)
    if recommendations is not None and not recommendations.empty:
        st.write("Here are some recommendations based on your choice:")
        st.markdown(
            """
                <style>
                    table {
                        background-color: black;
                        color: white;
                    }
                    th, td {
                        padding: 10px;
                        text-align: left;
                    }
                </style>
            """, unsafe_allow_html=True
        )
        st.table(recommendations[['Title', 'Genres', 'Release Date', 'IMDB Rating', 'Director', 'Cast']].reset_index(drop=True))
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.write("Sorry, no recommendations available for that movie. Please try another one.")

if st.button('❌ Clear Selection'):
    st.experimental_rerun()
