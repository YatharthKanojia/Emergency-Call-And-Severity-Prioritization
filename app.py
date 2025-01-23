import pandas as pd
import re
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import joblib
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Function to clean text data
def clean_text(text):
    """
    Clean the input text by removing URLs, mentions, hashtags, special characters, and stopwords.
    """
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)  # Remove @mentions and hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove special characters and numbers
    text = text.lower()  # Convert to lowercase
    text = text.strip()  # Remove leading/trailing whitespaces
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]  # Remove stopwords
    return ' '.join(words)

# Load the pre-trained Logistic Regression model and TF-IDF vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    """
    Load the pre-trained Logistic Regression model and TF-IDF vectorizer.
    """
    model = joblib.load(r"C:\Users\hp\OneDrive\Documents\Project-DisasterManagement\logistic_regression_model.pkl")  # Load model
    tfidf = joblib.load(r"C:\Users\hp\OneDrive\Documents\Project-DisasterManagement\tfidf_vectorizer.pkl")
    return model, tfidf

# Load the model and TF-IDF vectorizer
model, tfidf = load_model_and_vectorizer()

# Streamlit App Layout
st.title('Disaster Management Severity Prediction')
st.write('This app predicts severity levels based on user-provided comments or uploaded datasets and displays a classification report.')

# Sidebar for options
option = st.sidebar.radio("Choose an option", ["Predict Severity for Comments", "Analyze Uploaded Dataset"])

# Option 1: Predict severity for real-time comments
if option == "Predict Severity for Comments":
    st.header("Real-Time Severity Prediction")
    
    # Input text box for real-time comment
    user_comment = st.text_area("Enter your comment below to predict its severity:")

    if user_comment:
        # Step 1: Clean the user-provided comment
        cleaned_comment = clean_text(user_comment)
        
        # Step 2: Transform the comment using the pre-trained TF-IDF vectorizer
        transformed_comment = tfidf.transform([cleaned_comment])
        
        # Step 3: Predict the severity using the pre-trained model
        predicted_severity = model.predict(transformed_comment)[0]
        
        # Step 4: Display the results
        st.write(f"**Entered Comment:** {user_comment}")
        st.write(f"**Predicted Severity Level:** {predicted_severity}")

        # Provide additional information or context for severity levels
        if predicted_severity == 'High':
            st.error("This comment indicates a **High Severity** situation. Immediate action is required!")
        elif predicted_severity == 'Medium':
            st.warning("This comment indicates a **Medium Severity** situation. Action is recommended.")
        else:
            st.success("This comment indicates a **Low Severity** situation. No immediate action needed.")

# Option 2: Analyze an uploaded dataset
elif option == "Analyze Uploaded Dataset":
    st.header("Dataset Analysis for Severity Predictions")
    st.write("Upload a dataset to predict severity levels for multiple entries and view the classification report.")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        # Load dataset
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Dataset")
        st.write(df.head())

        # Check if 'text' column exists
        if 'text' not in df.columns:
            st.error("The dataset must have a 'text' column.")
        else:
            # Step 1: Clean the text in the dataset
            df['cleaned_text'] = df['text'].apply(lambda x: clean_text(x) if isinstance(x, str) else '')
            
            # Step 2: Transform the text using TF-IDF vectorizer
            X_tfidf = tfidf.transform(df['cleaned_text'])
            
            # Step 3: Predict severity for each row
            df['predicted_severity'] = model.predict(X_tfidf)
            
            # Display the dataset with predicted severity
            st.write("### Dataset with Predicted Severity")
            st.write(df[['text', 'predicted_severity']])
            
            # Optionally calculate classification report if 'severity' column exists
            if 'severity' in df.columns:
                st.write("### Classification Report")
                report = classification_report(
                    df['severity'], df['predicted_severity'], zero_division=1, output_dict=True
                )
                st.write(pd.DataFrame(report).T)
            
            # Visualize the Confusion Matrix
            st.write("### Confusion Matrix")
            cm = confusion_matrix(df['severity'], df['predicted_severity'])
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['High', 'Medium', 'Low'], yticklabels=['High', 'Medium', 'Low'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)

            # Bar Chart for Severity Count
            st.write("### Severity Distribution (Bar Chart)")
            severity_counts = df['predicted_severity'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 6))
            severity_counts.plot(kind='bar', color=['red', 'yellow', 'green'])
            ax.set_title('Predicted Severity Distribution')
            ax.set_xlabel('Severity Level')
            ax.set_ylabel('Count')
            st.pyplot(fig)

            # Pie Chart for Severity Distribution
            st.write("### Severity Distribution (Pie Chart)")
            fig, ax = plt.subplots(figsize=(8, 6))
            severity_counts.plot(kind='pie', autopct='%1.1f%%', colors=['red', 'yellow', 'green'], ax=ax, legend=False)
            ax.set_title('Predicted Severity Distribution')
            st.pyplot(fig)

            # Option to download the updated dataset
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predicted Dataset",
                data=csv,
                file_name='predicted_severity_dataset.csv',
                mime='text/csv',
            )
