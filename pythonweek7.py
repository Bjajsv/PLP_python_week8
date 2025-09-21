import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np

# Set seaborn style for better visualization
sns.set_style("whitegrid")

def load_iris_data():
    """Load Iris dataset and convert to pandas DataFrame"""
    try:
        # Load iris dataset from sklearn
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def explore_dataset(df):
    """Explore dataset structure and handle missing values"""
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # If there were missing values, we could handle them like this:
    # df = df.fillna(df.mean())  # Fill with mean for numerical columns
    # df = df.dropna()  # Drop rows with missing values
    
    return df

def analyze_data(df):
    """Perform basic statistical analysis"""
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Group by species and compute mean for each numerical column
    print("\nMean values by Species:")
    print(df.groupby('species').mean())

def create_visualizations(df):
    """Create four different types of visualizations"""
    
    # 1. Line Chart: Mean measurements across species
    plt.figure(figsize=(10, 6))
    for column in df.columns[:-1]:  # Exclude species column
        means = df.groupby('species')[column].mean()
        plt.plot(means.index, means.values, marker='o', label=column)
    plt.title('Mean Measurements Across Iris Species')
    plt.xlabel('Species')
    plt.ylabel('Mean Value (cm)')
    plt.legend()
    plt.savefig('line_chart.png')
    plt.close()
    
    # 2. Bar Chart: Average sepal length by species
    plt.figure(figsize=(8, 6))
    means = df.groupby('species')['sepal length (cm)'].mean()
    sns.barplot(x=means.index, y=means.values)
    plt.title('Average Sepal Length by Species')
    plt.xlabel('Species')
    plt.ylabel('Sepal Length (cm)')
    plt.savefig('bar_chart.png')
    plt.close()
    
    # 3. Histogram: Distribution of petal length
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x='petal length (cm)', bins=20)
    plt.title('Distribution of Petal Length')
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Count')
    plt.savefig('histogram.png')
    plt.close()
    
    # 4. Scatter Plot: Sepal length vs Petal length
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', 
                   hue='species', size='species')
    plt.title('Sepal Length vs Petal Length')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.legend()
    plt.savefig('scatter_plot.png')
    plt.close()

def main():
    # Load dataset
    df = load_iris_data()
    
    if df is not None:
        # Explore dataset
        df = explore_dataset(df)
        
        # Analyze data
        analyze_data(df)
        
        # Create visualizations
        create_visualizations(df)
        
        # Print findings
        print("\nFindings and Observations:")
        print("- The dataset contains 150 samples with 4 numerical features and 1 categorical feature (species).")
        print("- There are three species: setosa, versicolor, and virginica.")
        print("- Setosa generally has smaller measurements compared to versicolor and virginica.")
        print("- Petal measurements show more distinct separation between species than sepal measurements.")
        print("- The scatter plot shows clear clustering of species based on sepal and petal length.")
    else:
        print("Failed to load dataset. Analysis terminated.")

if __name__ == "__main__":
    main()