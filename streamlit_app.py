import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Berita Indonesia", page_icon="📰", layout="wide")

# Fungsi untuk membersihkan teks
def clean_text(text):
    return re.sub(r'[^\w\s\d]', '', text).lower()

# Fungsi rekomendasi berita dengan pencarian, sortir, dan filter kategori
def recommend_news(news_data, tfidf_matrix, tfidf_vectorizer, keyword, categories, sort_by, sort_order, min_similarity=0.25):
    if keyword:
        keyword_vector = tfidf_vectorizer.transform([clean_text(keyword)])
        cosine_similarities = cosine_similarity(keyword_vector, tfidf_matrix).flatten()
        news_data['similarity'] = cosine_similarities
        news_data = news_data[news_data['similarity'] >= min_similarity]
    
    # Filter berdasarkan kategori yang dipilih
    news_data = news_data[news_data['category'].isin(categories)]

    # Sortir hasil
    if sort_by == 'date':
        news_data = news_data.sort_values(by='date', ascending=(sort_order == 'oldest'))
    elif sort_by == 'alphabet':
        news_data = news_data.sort_values(by='title', ascending=(sort_order == 'A-Z'))
    elif sort_by == 'similarity' and keyword:
        news_data = news_data.sort_values(by='similarity', ascending=(sort_order == 'least relevant'))
    
    return news_data

# Fungsi untuk menampilkan berita dengan paginasi
def display_news(news, items_per_page=10):
    # Hitung total halaman
    total_pages = max(1, len(news) // items_per_page + (len(news) % items_per_page > 0))
    # Input untuk memilih nomor halaman
    page_number = st.number_input("Halaman", min_value=1, max_value=total_pages, value=1)
    start_item = (page_number - 1) * items_per_page
    end_item = start_item + items_per_page
    sliced_news = news.iloc[start_item:end_item]

    for idx, row in sliced_news.iterrows():
        st.text(f"Date: {row['date']} | Category: {row['category']}")
        st.markdown(f"##### [{row['title']}]({row['url']})")
        if 'similarity' in row:
            st.text(f"Similarity Score: {row['similarity']:.2f}")
        st.markdown("---")

# Fungsi utama untuk aplikasi Streamlit
def main():
    st.title("Berita Indonesia")

    @st.cache_data
    def load_data():
        data = pd.read_csv('indonesian-news-title.csv')
        data.dropna(subset=['title'], inplace=True)
        data['title_cleaned'] = data['title'].apply(clean_text)
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(data['title_cleaned'])
        return data, matrix, vectorizer

    data, matrix, vectorizer = load_data()

    with st.sidebar:
        st.header("Pencarian Berita")
        keyword = st.text_input("Masukkan kata kunci berita:", "")
        category_selected = st.multiselect("Pilih Kategori:", options=data['category'].unique(), default=data['category'].unique())

        # Tentukan opsi pengurutan
        sort_options = ["date", "alphabet"] + (["similarity"] if keyword else [])
        sort_by = st.selectbox("Urutkan Berdasarkan:", sort_options, index=0)
        sort_order_labels = ["newest", "oldest"] if sort_by == "date" else ["A-Z", "Z-A"] if sort_by == "alphabet" else ["most relevant", "least relevant"]
        sort_order = st.selectbox("Urutan:", sort_order_labels, index=0)
        
        search = st.button("Cari")

    if search:
        if not category_selected:
            st.error("Pilih minimal satu kategori.")
        else:
            results = recommend_news(data, matrix, vectorizer, keyword, category_selected, sort_by, sort_order)
            if results.empty:
                st.info('Tidak ada berita yang cukup mirip dengan kata kunci yang diberikan.')
            else:
                display_news(results)
    else:
        # Jika tidak ada kata kunci, urutkan dan tampilkan berita berdasarkan pilihan pengguna
        sorted_data = data
        if sort_by == 'date':
            sorted_data = sorted_data.sort_values(by='date', ascending=(sort_order == 'oldest'))
        elif sort_by == 'alphabet':
            sorted_data = sorted_data.sort_values(by='title', ascending=(sort_order == 'A-Z'))
        display_news(sorted_data)

    st.sidebar.markdown("---")
    st.sidebar.text("Footer: Informasi Tambahan")

if __name__ == "__main__":
    main()
