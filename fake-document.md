# Let's create the documentation as a text file

documentation = """
# Fake News Detection - Python Script Documentation

## **Overview**

This script is designed to detect fake news from a dataset of news articles. The pipeline includes data loading, cleaning, feature extraction, model training, evaluation, and deployment. The script leverages machine learning algorithms such as Logistic Regression and Random Forest, and includes deep learning models (LSTM) as optional enhancements. It also provides a REST API for deployment and an interactive web app using Streamlit.

---

# Fake News Detection - Python Script Documentation (Extended)

## **Project Motivation**
With the rise of misinformation on social media and digital platforms, it is increasingly important to have tools that can automatically detect fake news. This project aims to reduce the spread of false information by providing an effective way to identify and classify news articles as fake or true.

## **Technologies Used**
- **Python**: The main programming language used for data analysis and machine learning tasks.
- **Scikit-learn**: A popular machine learning library for model training and evaluation.
- **TensorFlow/Keras**: Used for training deep learning models like LSTM.
- **NLTK**: Natural Language Toolkit used for text preprocessing tasks like tokenization and lemmatization.
- **Flask**: A lightweight web framework to build and deploy the model as a REST API.
- **Streamlit**: A framework to create interactive web applications for data science projects.

## **Data Exploration Details**
The dataset contains over 40,000 articles split into fake and true categories. Key insights include:
- A noticeable imbalance in the number of fake versus true news articles.
- The presence of certain patterns in the content and subject matter (e.g., fake news often comes from sensationalized topics).

## **Model Evaluation**
We used accuracy, precision, recall, and F1-score to evaluate the performance of our models. While accuracy gives a general sense of the model’s performance, precision and recall are more suitable for evaluating the model's performance on imbalanced datasets.

## **Hyperparameter Tuning**
GridSearchCV was used to fine-tune the hyperparameters of the Random Forest model. This process involved testing different combinations of hyperparameters such as the number of trees and depth of the trees to optimize the model's performance.

## **Model Comparison**
- **Logistic Regression**: While fast and effective, it was less powerful at capturing complex patterns compared to other models.
- **Random Forest**: Provided better accuracy due to its ability to capture complex relationships in the data.
- **LSTM**: This deep learning model showed the most promise in understanding contextual information from the text, though it required more computational resources and was slower to train.

## **Deployment Details**
- **Flask API**: A simple API is deployed to accept news article text via a POST request and return a prediction of whether the article is fake or true.
- **Streamlit Web App**: An interactive web application where users can input text directly and see predictions in real time.

## **Limitations and Future Work**
While the model performs well on the training dataset, there are certain limitations:
- The current model relies on traditional text features like TF-IDF, which may not capture more complex semantics present in the text.
- Future work could involve exploring more advanced NLP techniques, such as transformer models like BERT or GPT, to improve performance.

## **Ethical Considerations**
While the model aims to detect fake news, it is important to recognize that biases in the training data could affect its accuracy. Automated classification of news articles raises ethical concerns, such as the risk of misclassifying legitimate news articles as fake.

## **Future Enhancements**
- **Sentiment Analysis**: Could be added to detect the tone of news articles and identify whether the content is sensationalized.
- **Real-time Data Processing**: Integrating with live news sources (e.g., Twitter or news APIs) for real-time fake news detection.

"""

# Combine the original documentation with the additional sections
full_documentation = documentation + additional_documentation

# Write the extended documentation to a file
extended_file_path = "/mnt/data/Extended_Fake_News_Detection_Documentation.txt"

with open(extended_file_path, "w") as doc_file:
    doc_file.write(full_documentation)

extended_file_path


## **Datasets**

### **Fake.csv**
- **Rows:** 23,481
- **Columns:** 4 (`title`, `text`, `subject`, `date`)
- **Label:** 0 (Fake news)

### **True.csv**
- **Rows:** 21,417
- **Columns:** 4 (`title`, `text`, `subject`, `date`)
- **Label:** 1 (True news)

**Note:**
- Both datasets follow the same structure and contain news articles, their associated categories (`subject`), and publication dates.
- `text` is the main content for analysis, while the `label` column (0 for fake, 1 for true) serves as the target variable.

---

## **Features**

### **1. Data Preprocessing**
The script implements advanced text cleaning procedures:
- **Text cleaning:** Removal of special characters, punctuation, and URLs.
- **Stopwords removal**: Filters out common words that do not contribute to text classification.
- **Tokenization:** Splits text into individual words.
- **Lemmatization:** Reduces words to their base form (e.g., "running" → "run").
- **Optional spell correction**: Corrects misspelled words using the `autocorrect` library.

### **2. Exploratory Data Analysis (EDA)**
Visualizations to better understand the dataset:
- **Distribution of fake vs. true news articles**
- **Word clouds**: Separate clouds for fake and true news.
- **Article length distribution**: Displays the word count distribution of articles.
- **Subject distribution**: Frequency of different subjects.
- **Publication dates over time**: Examines trends in article publication.

### **3. Feature Engineering**
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Extracts meaningful features from the text. The script uses **unigrams, bigrams, and trigrams**.
- **Word Embeddings (optional)**: Uses Word2Vec (via `gensim`) for capturing semantic meaning from words, if opted.

### **4. Model Training**
- **Classical Models**:
  - **Logistic Regression**
  - **Random Forest Classifier**: Both models are trained using TF-IDF features and evaluated using traditional metrics (accuracy, precision, recall, F1-score).
- **Deep Learning Models (Optional)**:
  - **LSTM (Long Short-Term Memory)**: For sequential text data, using Keras.
  
  The script also implements **hyperparameter tuning** using GridSearchCV for Random Forest to find the optimal parameters.

### **5. Evaluation Metrics**
- **Accuracy**
- **Precision, Recall, F1-Score**
- **Confusion Matrix**
- **ROC-AUC Curve**
- **Precision-Recall Curve**
- **SHAP (SHapley Additive exPlanations)**: Provides explainability for model predictions, helping to understand which features contribute to the model's decision-making process.

### **6. Deployment**
- **Flask API**: A simple API to make predictions by sending a POST request with news article text. The model predicts whether the article is fake or true.
  - *API Endpoint*: `/predict`
  - Accepts JSON input (`{"text": "Your article content here"}`)
  
- **Streamlit Web App**: An interactive UI to input a news article and display the predicted label (Fake/True).

### **7. Multilingual Support (Optional)**
- The script can detect the language of input text and filter out non-English articles using the `langdetect` library.

---

## **Installation and Setup**

### **1. Install Dependencies**
Ensure that the following Python libraries are installed:
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud tensorflow keras flask streamlit shap langdetect autocorrect gensim
