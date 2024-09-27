# Ticketing-System-Task

## Overview

This project involves processing a dataset of customer tickets, applying text cleaning (for both Arabic and English text), performing feature extraction, and building machine learning models for classification and clustering. The final model identifies key clusters of ticket requests based on text embeddings and evaluates models such as Logistic Regression and Random Forest.

## 1. Data Preprocessing
### 1.1 Date Handling

The dataset includes columns containing dates (Date, 1st-Response Date, and Close Date). These columns are converted into a datetime format, handling both numeric Excel-style dates and string dates.

If a column contains numeric data, it is interpreted as days since the Excel date origin (1899-12-30).
For other formats, non-parsable entries are set to NaT (Not a Time).

## 1.2 Missing Values

Missing values are handled appropriately based on column relevance:


For 'Close Date', NaT values are kept and not used in the response time calculation.

## 1.3 Text Cleaning

The text cleaning process addresses both Arabic and English entries:

* Arabic Text: We apply custom normalization (e.g., unifying different forms of letters) and remove unwanted characters.
* English Text: We tokenize and remove stopwords using NLTK’s English stopwords.
* General Cleaning: All text undergoes removal of HTML tags, extra spaces, and irrelevant characters.

## 1.4 Applying Cleaning Functions

Text cleaning is applied to columns that contain descriptive information (Request Type, Subject, Request Detail, Notes). This ensures both Arabic and English text are processed and normalized.

# 2. Exploratory Data Analysis (EDA)
## 2.1 Major Insights

Five insights were drawn from the dataset:

Insight 1: Most common request types.
Insight 2: Distribution of priorities across tickets.
Insight 3: Average response time between first response and ticket closure.
Insight 4: Most active companies based on ticket submission.
Insight 5: Ticket status distribution.
## 2.2 Word Cloud Visualization

A word cloud was generated based on the Request Detail column, highlighting the most frequently used words in the requests.

## 2.3 Correlation Analysis

Correlation analysis was performed on numerical features, including the Response Time, using a heatmap to visualize relationships.

## 2.4 Data Distribution

Visualizations were created for:

* Response Time Distribution: Histogram of the number of days it took to close tickets.
* Priority Distribution: Bar plot showing the frequency of different ticket priorities.

## 2.5 Statistical Summary

A statistical summary of the data (mean, count, unique values, etc.) was produced using the describe function.

# 3. Feature Engineering
## 3.1 Combining Text Features

Request Detail and Notes columns were concatenated into a new column combined_text for more comprehensive text processing.

## 3.2 Feature Extraction Methods
Three feature extraction techniques were applied to the text data:

TF-IDF: Transforms the text into a matrix of term frequencies, weighted by inverse document frequency.
Count Vectorizer: Transforms text into a matrix based on raw word counts.
BERT Embeddings: Uses BERT (Bidirectional Encoder Representations from Transformers) to convert text into contextual embeddings, representing the semantic meaning of the text.
## 3.3 Creating Feature DataFrames

Each feature extraction technique generates a feature matrix. These matrices are combined into a single DataFrame, which is used for subsequent modeling.

# 4. Clustering
## 4.1 Silhouette Score for Optimal Clusters
To determine the optimal number of clusters, the silhouette score was calculated for various cluster sizes (2 to 9). The optimal number of clusters was selected based on the highest silhouette score.

## 4.2 KMeans Clustering
KMeans clustering was applied using the optimal number of clusters. The resulting clusters were assigned to the dataset, and the cluster distribution was analyzed.

# 5. Modeling
## 5.1 Train-Test Split
The dataset was split into training and testing sets (80/20 split), using the Cluster labels as the target variable.

## 5.2 Logistic Regression
A Logistic Regression model was trained to predict the cluster labels. The model's performance was evaluated on the test set.

## 5.3 Random Forest Classifier
A Random Forest Classifier was trained and tested similarly to the Logistic Regression model.

## 5.4 BERT-Based Classification (Planned)
BERT embeddings were used to further enhance the classification with more complex models like BERT-based sequence classification..

# 6. Model Evaluation
## 6.1 Model Performance Metrics
* Accuracy Score: Measures the overall correctness of predictions.
* Precision Score: Evaluates the accuracy of the positive predictions.
* Recall Score: Measures the completeness of positive predictions.
* Classification Report: A comprehensive report showing precision, recall, and F1 scores.
* Confusion Matrix: A matrix visualizing the true versus predicted clusters.
# 7. Visualizations
* Word Cloud: Displays the most common words in the Request Detail feature.
* Heatmap: Visualizes the correlation between numerical features.
* Cluster Silhouette Scores: Shows silhouette scores for varying cluster numbers.
* Distribution Plots: Visualizes the distribution of response times and priorities in the dataset.

# 8. Files
* Dataset: The dataset used in this project is stored as Book1.xlsx.
* Report Notebook: All code and visualizations are part of the implementation.
* Models: Logistic Regression and Random Forest models are trained and saved using the joblib library for later use.
# 9. Conclusion
The report documents a comprehensive data processing pipeline, feature extraction, and model development workflow. Clustering and classification models were built to identify key patterns in ticket data, and insights were drawn from both the cleaned data and the resulting clusters.

