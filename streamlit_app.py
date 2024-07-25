import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page config
st.set_page_config(page_title="Berita Indonesia", page_icon="📰", layout="wide")

# Define clean_text function
def clean_text(text):
    return re.sub(r'[^\w\s\d]', '', text).lower()

# Define recommend_news function
def recommend_news(news_data, tfidf_matrix, tfidf_vectorizer, keyword, categories, sort_by, sort_order, min_similarity=0.25):
    # Perform the search and similarity scoring
    if keyword:
        keyword_vector = tfidf_vectorizer.transform([clean_text(keyword)])
        cosine_similarities = cosine_similarity(keyword_vector, tfidf_matrix).flatten()
        news_data['similarity'] = cosine_similarities
        news_data = news_data[news_data['similarity'] >= min_similarity]
    
    # Filter by categories
    news_data = news_data[news_data['category'].isin(categories)]

    # Sort results
    if sort_by == 'date':
        news_data = news_data.sort_values(by='date', ascending=(sort_order == 'oldest'))
    elif sort_by == 'alphabet':
        news_data = news_data.sort_values(by='title', ascending=(sort_order == 'A-Z'))
    elif sort_by == 'similarity' and keyword:
        news_data = news_data.sort_values(by='similarity', ascending=(sort_order == 'least relevant'))
    
    return news_data

# Define a function to handle news display with pagination
def display_news(news, items_per_page=10):
    if news.empty:
        st.write("Berita tidak ditemukan! Silahkan periksa kembali kata kunci atau kategori Anda.")
        return
    
    # Calculate total pages
    total_pages = max(1, len(news) // items_per_page + (len(news) % items_per_page > 0))

    # Create a session state for page number
    if 'page_number' not in st.session_state:
        st.session_state.page_number = 1

    # Adjust the current page number if it's out of range due to a new search
    if st.session_state.page_number > total_pages:
        st.session_state.page_number = total_pages

    # Create a number input for pagination
    page_number = st.number_input("Halaman", min_value=1, max_value=total_pages, value=st.session_state.page_number)
    st.session_state.page_number = page_number

    # Calculate the indices of the items to display on the current page
    start_index = (page_number - 1) * items_per_page
    end_index = start_index + items_per_page

    # Display the sliced items for the current page
    for idx, row in news.iloc[start_index:end_index].iterrows():
        st.text(f"Date: {row['date']} | Category: {row['category']}")
        st.markdown(f"##### [{row['title']}]({row['url']})")
        if 'similarity' in row:
            st.text(f"Similarity Score: {row['similarity']:.2f}")
        st.markdown("---")

# Define the main function for the Streamlit app
def main():
    st.title("Berita Indonesia")

    # Load data function
    @st.cache_data
    def load_data():
        data = pd.read_csv('indonesian-news-title.csv')
        data.dropna(subset=['title'], inplace=True)
        data['title_cleaned'] = data['title'].apply(clean_text)
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(data['title_cleaned'])
        return data, matrix, vectorizer

    data, matrix, vectorizer = load_data()

    # Sidebar for user input
    with st.sidebar:
        st.header("Pencarian Berita")
        keyword = st.text_input("Masukkan kata kunci berita:", "")
        category_selected = st.multiselect("Pilih Kategori:", options=data['category'].unique(), default=data['category'].unique())
        sort_by = st.selectbox("Urutkan Berdasarkan:", ["date", "alphabet", "similarity"], index=0)
        sort_order = st.selectbox("Urutan:", ["newest", "oldest"] if sort_by == "date" else ["A-Z", "Z-A"] if sort_by == "alphabet" else ["most relevant", "least relevant"], index=0)
        search = st.button("Cari")

    # Initialize session state for search results if it doesn't exist
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None

    if search:
        # Perform the search and store the results in session state
        st.session_state.search_results = recommend_news(data, matrix, vectorizer, keyword, category_selected, sort_by, sort_order)
        st.session_state.page_number = 1  # Reset page number to 1 for a new search

    # Display search results if they exist, otherwise show default sorted data
    if st.session_state.search_results is not None:
        display_news(st.session_state.search_results)
    else:
        # Sort and display news based on user selection when not searching
        sorted_data = data
        if sort_by == 'date':
            sorted_data = sorted_data.sort_values(by='date', ascending=(sort_order == 'oldest'))
        elif sort_by == 'alphabet':
            sorted_data = sorted_data.sort_values(by='title', ascending=(sort_order == 'A-Z'))
        display_news(sorted_data)

    st.sidebar.markdown("---")
    st.sidebar.markdown("[github.com/devapratama](https://github.com/devapratama)")

# Run the main function
if __name__ == "__main__":
    main()
