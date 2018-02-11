## Quora Question Pairs
Kaggle competition 

## Files
main.py         -- Code to implement all the models

## Datasets
The datasets are available as train.csv.zip and test.csv.zip at:
https://www.kaggle.com/c/quora-question-pairs/data

## Dependencies
numpy, sklearn, pandas, nltk, csv, re

## How to run
python main.py <jaccard|cosine|tfidf|logistic|naivebayes|randomforest|voting>

## Individual Classifiers
* Jaccard Similarity
* Cosine Similarity
* Pearson Coefficient
* TF-IDF based Cosine Similarity

## Ensemble Classifiers
* Logistic Regression
* Naive Bayes Model
* Random Forest Model
* Probabilistic Voting Ensemble

Note: The voting ensemble takes a huge amount of time to train

## Results
log-loss value of: **0.40167** with Probabilistic Voting Ensemble. (Still improving it)

