INTRODUCTION
This project is a Disaster Management Call Prioritization System that leverages Natural Language Processing (NLP) and machine learning to classify and prioritize emergency incidents based on severity. It uses Logistic Regression for severity classification, a priority queue to process critical cases first, and Streamlit for an interactive user interface to visualize and analyze the results.

FEATURES:
* Text Preprocessing: Cleans and prepares input text data for analysis using NLP techniques.
* Severity Classification: Uses Logistic Regression to classify emergency incidents into High, Medium, and Low severity levels.
* Priority Queue System: Overrides the traditional First-Come-First-Serve (FCFS) approach to prioritize high-severity cases for immediate attention.
* Real-Time and Batch Processing: Allows real-time predictions for individual cases and bulk predictions for datasets.
* Visual Insights: Displays confusion matrices and severity distribution graphs to evaluate the system’s performance.
* Exportable Results: Enables saving the processed data with severity classifications and priority scores as a CSV file.

PREREQUISITES:
* Python 3.x
* Streamlit
* Scikit-learn
* NLTK
* Pandas
* Matplotlib
* NumPy

FILES AND THEIR FUNCTIONS:
* disaster_management.py: This file contains the functionality for preprocessing and transforming the text data. It handles tasks such as tokenization, stopword removal, and vectorization using the TF-IDF technique. These steps ensure that the raw input text is cleaned and converted into a format suitable for machine learning algorithms. Also this file implements the priority queue system. It assigns a priority score to each emergency call based on its severity classification (High, Medium, Low). The system uses this priority score to process critical cases first, overriding the traditional First-Come-First-Serve (FCFS) method.

* app.py: This is the main file for the Streamlit-based web application. It serves as the user interface, allowing users to input real-time text data or upload datasets for batch processing. The app visualizes predictions, displays severity distributions, provides confusion matrices for evaluation, and allows downloading processed results with priority scores.

* visualizations.py: This file contains functions to visualize the processed data. It generates charts and graphs, such as severity distribution plots and confusion matrices, to help users understand and analyze the predictions and the system's performance effectively.

This system is designed to enhance disaster management efforts by ensuring critical emergencies are addressed with priority, thereby improving response efficiency and saving lives.
