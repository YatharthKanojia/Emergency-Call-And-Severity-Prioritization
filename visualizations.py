import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Function to load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    return df

# Function to plot bar graph of severity counts
def plot_severity_distribution(df):
    severity_counts = df['severity'].value_counts()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=severity_counts.index, y=severity_counts.values, palette='viridis')
    plt.xlabel('Severity Level')
    plt.ylabel('Count')
    plt.title('Severity Level Distribution')
    plt.show()

# Function to plot pie chart for severity distribution
def plot_severity_pie_chart(df):
    severity_counts = df['severity'].value_counts()
    plt.figure(figsize=(8, 8))
    severity_counts.plot.pie(autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99'])
    plt.ylabel('')
    plt.title('Severity Distribution (Pie Chart)')
    plt.show()

if __name__ == "__main__":
    file_path = r"C:\Users\hp\OneDrive\Documents\Project-DisasterManagement\socialmedia-disaster-tweets-DFE_updated.csv"
    df = load_data(file_path)

    if 'severity' in df.columns:
        plot_severity_distribution(df)
        plot_severity_pie_chart(df)
    else:
        print("Error: 'severity' column does not exist in the DataFrame.")
