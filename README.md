## Project Overview

This project performs sentiment analysis on customer reviews collected from platforms like Amazon and Flipkart. It uses NLP techniques such as tokenization, stopword removal, stemming, and TF-IDF vectorization to convert raw text into numerical features.

A Multinomial Naive Bayes classifier is trained to classify each review as positive, negative, or neutral. Model performance is evaluated using metrics like accuracy, precision, recall, and visualized through a confusion matrix.

This internship project demonstrates applied skills in text classification, natural language processing, and machine learning.

Key features include a full preprocessing pipeline, effective feature extraction, robust classification, and detailed evaluation.

## 🛠️ Technologies Used
- **Python 3.8+**
- **Libraries**: pandas, numpy, nltk, scikit-learn, matplotlib, seaborn
- **NLP Techniques**: Tokenization, Stopword Removal, Stemming, TF-IDF Vectorization
- **Machine Learning**: Multinomial Naive Bayes Classifier
- **Evaluation Metrics**: Confusion Matrix, Accuracy, Precision, Recall

## ⚙️ Features

- **Multiclass Sentiment Classification**: Automatically categorizes reviews into positive, negative, and neutral classes.
- **Advanced Text Preprocessing**: Implements tokenization, stopword removal using NLTK’s English list, stemming via PorterStemmer, and normalization.
- **TF-IDF Vectorization**: Converts text data into numerical vectors using TF-IDF with support for unigrams and bigrams to capture context.
- **Naive Bayes Classifier**: Applies a Multinomial Naive Bayes algorithm optimized for high-dimensional sparse text datasets.
- **Performance Reporting**: Generates detailed classification reports including accuracy, precision, recall, F1-score, and confusion matrix heatmaps.
- **Modular and Scalable Codebase**: Structured notebook with reusable functions facilitating extensions or integration with larger datasets or other algorithms.
- **Custom Review Prediction**: Enables sentiment prediction on new/unseen customer reviews, demonstrating real-world application.
- ## 🚀 How to Run the Project

1. Clone the repository:
2. Change directory:
3. (Optional) Create and activate a virtual environment:
4. Install required packages:
5. Download NLTK datasets (if not already downloaded):
6. Open `sentiment_analysis_project.ipynb` in Jupyter Notebook or VS Code.
7. Run all cells sequentially to preprocess data, train the model, and evaluate performance.
8. Use the final code cells to classify any new customer reviews easily.

- ## 🔄 Methodology

### 1. Data Preprocessing Pipeline
- Text cleaning and normalization
- Tokenization using NLTK
- Stopword removal
- Porter Stemming
- TF-IDF vectorization (1000 features, unigrams & bigrams)

### 2. Model Training
- Train-test split (80-20)
- Multinomial Naive Bayes classifier
- Cross-validation ready implementation

### 3. Evaluation
- Confusion matrix visualization
- Accuracy, precision, recall metrics
- Feature importance analysis
- Real-world testing capabilities
- ## 📊 Results & Insights

- Achieved an **accuracy of approximately 100%** on the test set, demonstrating reliable sentiment classification.
- Confusion matrix reveals strong performance in distinguishing positive and negative sentiments, with some overlap in the neutral class, as expected due to its ambiguity.
- Feature importance analysis exposed terms strongly indicative of each sentiment, validating linguistic intuition (e.g., "excellent", "love" for positive; "poor", "disappointed" for negative).
- The modular pipeline allows easy adaptation to larger datasets or alternative machine learning models.
- ## 🚀 How to Run

### NLTK Data Setup


### Prerequisites

### Execution
1. Clone this repository
2. Install dependencies
3. Open `analysis_of_reviews.ipynb` in Jupyter Notebook/VS Code
4. Run all cells sequentially

   ## 📋 Implementation Highlights
- Complete text preprocessing pipeline
- TF-IDF feature extraction with optimal parameters
- Naive Bayes classification with Laplace smoothing
- Comprehensive evaluation metrics
- Visualization of results
- New review prediction capability
### 🔮 Future Enhancements
- Implement additional ML algorithms (SVM, Logistic Regression)
- Add cross-validation for robust performance assessment
- Expand dataset for improved model generalization
- Deploy as web application
- Add real-time sentiment analysis capabilities
- 
- ## 👤 Author
**[Mimansha]** - IBM Internship Candidate
- Project Duration: [07/2025]
- Contact: [chaudharymimansha126@gmail.com]
### 📄 License
This project is created for educational purposes as part of IBM Internship program.  



