# Natural-language-processing
---------

In today's world, one of the biggest sources of information is text data, which is unstructured in nature, finding customer sentiments from product reviews or feedback, extracting opinions from social media data, document classification, news articles category classification are a few examples of text analytics and natural language processing. Find overview and details of projects below

---------

Project 1: Sentiment classification of movies review dataset

Project 2: News article descriptions classification using bidirectional LSTMs

News description classification web-app link: [Click here](https://avikumart-natural-language-processing-news-clf-app-news--0st9q1.streamlitapp.com/)

Published article on Analytics Vidhya: [Click here](https://www.analyticsvidhya.com/blog/2022/09/understanding-word-embeddings-and-building-your-first-rnn-model/)

-----------------------

## 1. Overview of Sentiment classification project:
Finding Insights from text data is not as straight forward as structured data and it need extensive data pre-precessing. pre-precessing inclues removing punctuations, removing stop-words and cleaning words to it root format so that vectorized data contains meaningful features that maps target variable well enough.

In this notebook, we will use Count vectorizer & TF-IDF vectorizer along with stemming and n-grams to pre-precess and vectorize the movie reviews data before we build sentiment classification models for training and testing.

## Obejectives and tasks:
- Load and explore dataset
- Pre-process dataset using stemming and lammatization
- Vecotrize text documents using Count vectorizer, TF-IDF vectorizer and n-grams methods
- Build classification models using skalean's Naive Bayes and Tree based models
- Evaluate the model accuracy on test dataset.

## Approches:
- Pre-processed dataset using stop-words removal and stemming technique of NLTK
- Built various models using count vectorizer, TF-IDF and n-grams vectorizer to classify movies reviews into positive or negative 
- Evaluated models using classificaiton report and metrics like precision, recall, accuracy and F1-score

## Results:
1) I built 5 different models using Count vectorization and TF-IDF vectorization techniques to extract features from text data.
2) During model evaluation, Logistic regression model recieved highest accuracy of 99% and with Naive-Bayes model we recieved accuracy of 98%.
-------------------

## 2. Overview of News article descriptions classification project:
In this notebook, we are going to explore and solve news classification problem to classify 41 types of news headlines and news descriptions.
![image](https://user-images.githubusercontent.com/88608935/190999860-33bb6afd-998c-4a9d-8a7c-71985931f6a4.png)

### Web app screenshot:

![image](https://user-images.githubusercontent.com/88608935/227511231-24853809-a19d-406b-a06e-f190d6f5be33.png)



**Use-case:** Such text classification models are used in News Apps or by reporter to classify news topics for better reach to right audience.

**Problem-statement:** Build news classification model using deep learning teechniques and deploy model for reporters to classify and label news articles.

## Objectives and tasks:
- load dataset from JSON files to pandas dataframe
- Text data analysis and visualization
- Data Cleaning using Regex
- Tokenization and vectorization
- Understanding word-embeddings and RNNs
- Model training using Bidirectional LSTMs

## Approches:
- Comprehensivaley analyzed dataset to find number of categories, distribution of categories, most frequent words and visualizations
- Cleaned data using Regex by removing, unwanted characters from text documents
- Tokenization and word indexing using Keras library of python
- Feature represenation using word embedding layer of keras
- Built two models using simpleRNN and bidirectional LSTMs

## Result:
1) In this notebook, I explored some of text data visualization techniques to derive insights out of text data and make use of them into model training.
2) I built first model using simpleRNN and embedding layer of keras where I found maximum of 49% accuracy on test data and also noticed foregtting features by model due to large sequences of inputs.
3) In our second model, we trained model using LSTMs and GRU for retaining information of longer sequences. I archived training accuracy of 85% but failed to archive Higher test accuracy due to large amount of text features. I could 'optimize' model as it improved accuracy of training data significantly but it could not 'generalize well' enough on unseen data.

## Learnings:
1) From this project, I learned one of complex of techniques in NLP which is word-embeddings and it is crucial to learn word-embeddings to work in NLP tasks efficiently.
2) Owing to large amount of word features and text data, model could not archive desired test accuracy but it can be solved by pre-trained models like BERT without worrying about test accuracy.
