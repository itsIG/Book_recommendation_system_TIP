import streamlit as st
import pandas as pd
import numpy as np
import string
import re
from PIL import Image

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Create a Streamlit app
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Book Recommender System",
    layout="wide",
)

# Center-align subheading and image using HTML <div> tags
st.markdown(
    """
    <div style="display: flex; justify-content: center; align-items: center; margin-top: 20px; margin-bottom: 30px;">
        <img src="rec1.jpg" alt="Book Recommender System" style="max-width: 80%;">
    </div>
    """,
    unsafe_allow_html=True
)

# Add an introductory paragraph with improved styling
st.markdown(
    """
    <div style="text-align: justify; line-height: 1.5;">
        <h2 style="text-align: center; color: #FF2525;">Welcome to the Book Recommender Web-App!</h2>

        <p>
            Are you in search of your next captivating read? This application is here to assist! A clever system is utilized to analyze your preferences in a book. Aspects such as plot, characters, and themes that truly engage you are taken into consideration.
        </p>

        <p>
            Subsequently, book recommendations are provided that align with your tastes. Picture having a friend who is familiar with all your preferred genres and authors. They can suggest books that will likely resonate with you. That's precisely what is achieved here, but in the form of an app! So, if thrilling mysteries, heartwarming romances, epic fantasies, or insightful biographies are what you fancy, your preferences are catered to. Happy reading!
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Load the data into a DataFrame
df = pd.read_csv("recommender_books_with_url.csv")

# Copy df to df1
df1 = df.copy()

# text preprocessing function
def preprocess_text(text):
    # Handle NaN or float values
    if isinstance(text, float) or text is None or pd.isnull(text):
        return ''

    # Lowercase the text
    text = text.lower()

    # Remove URLs, hashtags, mentions, and special characters
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)

    # Remove numbers/digits
    text = re.sub(r'\b[0-9]+\b\s*', '', text)

    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])

    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join tokens back into a single string
    return ' '.join(tokens)

# Features to preprocess
features_to_preprocess = ['title', 'description', 'author', 'genre', 'publisher', 'language', 'rating', 'page_count']

# Data preprocessing for each feature
for feature in features_to_preprocess:
    # Apply text preprocessing to each entry in the feature
    df1[feature] = df1[feature].apply(preprocess_text)

# Concatenate the preprocessed features into a single text feature
df1['combined_features'] = df1[features_to_preprocess].apply(lambda x: ' '.join(x), axis=1)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the TF-IDF Vectorizer
tfidf_matrix = tfidf_vectorizer.fit_transform(df1['combined_features'])

# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Display the TF-IDF matrix shape
# st.write("TF-IDF Matrix shape:", tfidf_matrix.shape)

# Display the cosine similarity matrix shape and content
# st.write("Cosine Similarity Matrix shape:", cosine_sim.shape)
# st.write("Cosine Similarity Matrix:")
# st.write(cosine_sim)

# All Books heading
st.markdown(
    """
    <div style="margin-top: 20px; padding: 10px; background-color: #FF2525; border-radius: 10px; text-align:center; align-items:center;">
        <h2>All Books</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# Pagination function
# Pagination function with genre selection
def display_books_with_pagination(book_data, items_per_page=5):
    # Get unique genres
    unique_genres = df['genre'].unique()

    # Add a dropdown to select genre
    selected_genre = st.sidebar.selectbox("Select Genre", ['All'] + list(unique_genres))

    if selected_genre != 'All':
        # Filter books by the selected genre
        filtered_books = [book for book in book_data if book['genre'] == selected_genre]
        total_books = len(filtered_books)
    else:
        filtered_books = book_data
        total_books = len(filtered_books)

    num_pages = (total_books + items_per_page - 1) // items_per_page

    page = st.sidebar.slider("Section", 1, num_pages, 1)

    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page

    # st.write(f"Section {page}/{num_pages} for Genre: {selected_genre}:")
    st.write(f"Section {page}/{num_pages} for Genre➡  <span style='font-family: Cursive;'>{selected_genre}</span>", unsafe_allow_html=True)

    # Custom CSS for styling
    st.markdown(
        """
        <style>
        .book-card {
            border: 1px solid #ccc;
            padding: 12px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }
        .book-card img {
            max-width: 50%; /* Adjust the image width */
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    for i in range(start_idx, min(end_idx, total_books)):
        book = filtered_books[i]
        st.markdown(
            f"""
            <div class="book-card">
                <
