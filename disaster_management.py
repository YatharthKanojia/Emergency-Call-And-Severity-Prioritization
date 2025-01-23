import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from queue import PriorityQueue
from sklearn.utils import shuffle

nltk.download('stopwords')

df = pd.read_csv(r"C:\Users\hp\Downloads\socialmedia-disaster-tweets-DFE.csv", encoding='ISO-8859-1')  # Update with actual file path

stop_words = set(stopwords.words('english'))

# Function to clean text data
def clean_text(text):
    # Remove URLs, special characters, numbers, and convert to lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)  # remove @mentions and hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)  # remove special characters and numbers
    text = text.lower()  # convert to lowercase
    text = text.strip()  # remove leading/trailing whitespaces
    
    words = text.split()
    
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    
    return ' '.join(words)

df['cleaned_text'] = df['text'].apply(lambda x: clean_text(x) if isinstance(x, str) else '')

def assign_severity(text):
    if 'urgent' in text or 'emergency' in text or 'critical' in text or 'fast' in text :
        return 'High'
    elif 'help' in text or 'support' in text or 'assistance' in text:
        return 'Medium'
    else:
        return 'Low'

df['severity'] = df['cleaned_text'].apply(assign_severity)

# Remove some of the 'Low' severity instances
low_severity_df = df[df['severity'] == 'Low']
non_low_severity_df = df[df['severity'] != 'Low']

low_severity_df_sampled = low_severity_df.sample(frac=0.5, random_state=42)

df_balanced = pd.concat([non_low_severity_df, low_severity_df_sampled])

tfidf = TfidfVectorizer(max_features=5000)

X_tfidf = tfidf.fit_transform(df_balanced['cleaned_text'])

# Split data into training and testing sets with severity as target
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df_balanced['severity'], test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_resampled, y_train_resampled)

# Predict on the test data
y_pred = model.predict(X_test)

# Print the classification report and accuracy
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))  
print("Accuracy:", accuracy_score(y_test, y_pred))

priority_queue = PriorityQueue()

severity_priority_map = {'High': 1, 'Medium': 2, 'Low': 3}
for i, row in df.iterrows():
    priority = severity_priority_map[row['severity']]
    priority_queue.put((priority, i, row['text']))

# Process items in priority order and count severity levels
severity_count = {'High': 0, 'Medium': 0, 'Low': 0}

while not priority_queue.empty():
    item = priority_queue.get()
    severity_level = item[0]
    
    # Increment the count for the severity level
    if severity_level == 1:
        severity_count['High'] += 1
    elif severity_level == 2:
        severity_count['Medium'] += 1
    elif severity_level == 3:
        severity_count['Low'] += 1

# Print a summary after processing all calls
print("\nProcessing summary by severity priority:")
for level, count in severity_count.items():
    print(f"Processed {count} calls with severity level: {level}")
