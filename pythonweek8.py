```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import streamlit as st
from collections import Counter
import re
import os

# Set seaborn style for better visualization
sns.set_style("whitegrid")

def load_data(file_path):
    """Load the CORD-19 metadata.csv file with error handling"""
    try:
        df = pd.read_csv(./metadata.csv, low_memory=False)
        print("Dataset loaded successfully!")
        return df
    except FileNotFoundError:
        print(f"Error: File {(./metadata.csv} not found. Please ensure the file is in the correct directory.")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def explore_data(df):
    """Explore basic dataset structure"""
    print("\nDataset Dimensions (rows, columns):", df.shape)
    print("\nData Types:\n", df.info())
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nBasic Statistics:\n", df.describe())

def clean_data(df):
    """Clean and prepare the dataset"""
    # Handle missing values
    # Drop rows where title or publish_time is missing
    df_cleaned = df.dropna(subset=['title', 'publish_time']).copy()
    
    # Convert publish_time to datetime and extract year
    try:
        df_cleaned['publish_time'] = pd.to_datetime(df_cleaned['publish_time'], errors='coerce')
        df_cleaned['year'] = df_cleaned['publish_time'].dt.year
    except Exception as e:
        print(f"Error converting dates: {e}")
    
    # Fill missing journal values with 'Unknown'
    df_cleaned['journal'] = df_cleaned['journal'].fillna('Unknown')
    
    # Create abstract word count column
    df_cleaned['abstract_word_count'] = df_cleaned['abstract'].apply(
        lambda x: len(str(x).split()) if pd.notnull(x) else 0
    )
    
    return df_cleaned

def analyze_data(df):
    """Perform basic data analysis"""
    # Count papers by year
    year_counts = df['year'].value_counts().sort_index()
    
    # Top 5 journals
    top_journals = df['journal'].value_counts().head(5)
    
    # Simple word frequency in titles
    titles = ' '.join(df['title'].dropna().str.lower())
    # Remove punctuation and split into words
    words = re.findall(r'\b\w+\b', titles)
    # Remove common stop words (simplified list)
    stop_words = {'and', 'of', 'in', 'to', 'the', 'for', 'on', 'with', 'by'}
    word_freq = Counter(word for word in words if word not in stop_words).most_common(10)
    
    return year_counts, top_journals, word_freq

def create_visualizations(df, year_counts, top_journals, word_freq):
    """Create required visualizations"""
    
    # 1. Line Plot: Publications over time
    plt.figure(figsize=(10, 6))
    plt.plot(year_counts.index, year_counts.values, marker='o')
    plt.title('Number of Publications Over Time')
    plt.xlabel('Year')
    plt.ylabel('Number of Papers')
    plt.savefig('publications_over_time.png')
    plt.close()
    
    # 2. Bar Chart: Top 5 journals
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_journals.values, y=top_journals.index)
    plt.title('Top 5 Journals Publishing COVID-19 Research')
    plt.xlabel('Number of Papers')
    plt.ylabel('Journal')
    plt.savefig('top_journals.png')
    plt.close()
    
    # 3. Word Cloud: Frequent words in titles
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(
        ' '.join(word for word, _ in word_freq)
    )
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Paper Titles')
    plt.savefig('wordcloud.png')
    plt.close()
    
    # 4. Histogram: Distribution of papers by source
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='source_x', hue='source_x')
    plt.title('Distribution of Papers by Source')
    plt.xlabel('Source')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.savefig('source_distribution.png')
    plt.close()

def run_streamlit_app(df, year_counts, top_journals, word_freq):
    """Create Streamlit app"""
    st.title("CORD-19 Data Explorer")
    st.write("Explore patterns in COVID-19 research papers from the CORD-19 dataset")
    
    # Show dataset sample
    st.subheader("Sample Data")
    st.write(df[['title', 'journal', 'publish_time', 'abstract_word_count']].head())
    
    # Interactive year range slider
    min_year = int(df['year'].min())
    max_year = int(df['year'].max())
    year_range = st.slider("Select Year Range", min_year, max_year, (min_year, max_year))
    
    # Filter data based on year range
    filtered_df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    
    # Display visualizations
    st.subheader("Visualizations")
    
    # Publications over time
    st.image('publications_over_time.png', caption='Publications Over Time')
    
    # Top journals
    st.image('top_journals.png', caption='Top 5 Journals')
    
    # Word cloud
    st.image('wordcloud.png', caption='Word Cloud of Paper Titles')
    
    # Source distribution
    st.image('source_distribution.png', caption='Distribution by Source')
    
    # Basic statistics
    st.subheader("Basic Statistics")
    st.write(f"Total papers in selected year range: {len(filtered_df)}")
    st.write(f"Average abstract word count: {filtered_df['abstract_word_count'].mean():.2f}")

def main():
    # File path for metadata.csv (update this path based on your local setup)
    file_path = 'metadata.csv'
    
    # Part 1: Load and explore
    df = load_data(file_path)
    if df is None:
        return
    
    explore_data(df)
    
    # Part 2: Clean and prepare
    df_cleaned = clean_data(df)
    
    # Part 3: Analyze and visualize
    year_counts, top_journals, word_freq = analyze_data(df_cleaned)
    create_visualizations(df_cleaned, year_counts, top_journals, word_freq)
    
    # Part 4: Streamlit app
    run_streamlit_app(df_cleaned, year_counts, top_journals, word_freq)
    
    # Part 5: Findings
    print("\nFindings and Observations:")
    print("- The dataset contains metadata on thousands of COVID-19 research papers.")
    print("- Publication volume peaked around 2020-2021, reflecting the global research focus on COVID-19.")
    print("- Top journals include high-impact medical and scientific publications.")
    print("- Common title words often relate to COVID-19, SARS-CoV-2, and related medical terms.")
    print("- Missing values in key columns like title and publish_time were removed to ensure data quality.")

if __name__ == "__main__":
    main()
```