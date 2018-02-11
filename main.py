import os, sys
import csv
import numpy as np
import re
from nltk.corpus import stopwords
from collections import Counter
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.naive_bayes import GaussianNB as gn
from sklearn.linear_model import LogisticRegression as lr
from sklearn.ensemble import VotingClassifier as vc
from sklearn.linear_model import SGDClassifier as sgd
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier as mlp
import pandas as pd

np.random.seed(123)

stp = set(stopwords.words('english'))

def create_counters(d1, d2):
    v1 = Counter()
    v2 = Counter()

    for item in d1:
        v1[item] += 1

    for item in d2:
        v2[item] += 1

    return v1, v2

def tokenize(d):
    clean = re.sub(r'[^\w]', ' ', d)
    tk = clean.lower().split()
    return [word for word in tk if word not in stp]

def jaccard_similarity(d1, d2):
    intersect = set(d1).intersection(set(d2))
    union = set(d1).union(set(d2))
    if len(union) == 0:
        return 0.0
    return float(len(intersect))/len(union)

def cosine_similarity(d1, d2):
    v1, v2 = create_counters(d1, d2)
    
    intersect = set(v1.keys()) & set(v2.keys())
    n = sum([v1[i] * v2[i] for i in intersect])
    denom1 = sum([pow(v1[i], 2) for i in v1.keys()])
    denom2 = sum([pow(v2[i], 2) for i in v2.keys()])

    if denom1 == 0 or denom2 == 0:
        return 0.0

    return float(n)/(math.sqrt(denom1) * math.sqrt(denom2))

def pearson_similarity(d1, d2):
    v1, v2 = create_counters(d1, d2)

    intersect = set(v1.keys()) & set(v2.keys())
    n = len(intersect)

    if n == 0:
        return 0.0
    sum1, sum2, sum1s, sum2s, product = 0, 0, 0, 0, 0

    for i in intersect:
        sum1 += v1[i]
        sum2 += v2[i]
        sum1s += pow(v1[i],2)
        sum2s += pow(v2[i],2)
        product += v1[i] * v2[i]

    num = abs(product - (sum1*sum2/n))
    denom = math.sqrt((sum1s - pow(sum1, 2)/n) * (sum2s - pow(sum2, 2)/n))

    if denom == 0:
        return 0.0
    return float(num)/denom

def tfidf(d1, d2):
    intersect = set(d1).intersection(set(d2))
    if len(intersect) == 0:
        return 0.0

    tid = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, 
            smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
    try:
        represent = tid.fit_transform([d1, d2]).toarray()
    except ValueError:
        return 0.0

    # Taking dot product
    product = 0
    sum1 = 0
    sum2 = 0
    for i in range(len(represent[0])):
        product += represent[0][i] * represent[1][i]
        sum1 += pow(represent[0][i], 2)
        sum2 += pow(represent[0][i], 2)

    denom = math.sqrt(sum1 * sum2)
    if denom == 0:
        return 0.0

    return float(product)/denom

def similarity(f, train):
    train_data = []
    with open(f, "rb") as csvfile:
        reader = csv.reader(csvfile)
        header = reader.next()
        for row in reader:
            if train:
                print "Parsing Training Question Pair: " + row[0]
                i1, i2 = 3, 4
            else:
                print "Parsing Test Question Pair: " + row[0]
                i1, i2 = 1, 2
            clean = re.sub(r'[^\w]', ' ', row[i1])
            tokenize1 = clean.lower().split()
            tokenize1 = [word for word in tokenize1 if word not in stp]

            clean = re.sub(r'[^\w]', ' ', row[i2])
            tokenize2 = clean.lower().split()
            tokenize2 = [word for word in tokenize2 if word not in stp]

            # Jaccard Similarity
            jaccard = jaccard_similarity(tokenize1, tokenize2)
            if jaccard == -1 and train:
                continue
            elif jaccard == -1:
                jaccard = 0.0
                cosine = 0.0
                tid = 0.0
                train_data.append([int(row[0]), jaccard, cosine, tid])
            else:
                # Cosine Similarity
                cosine = cosine_similarity(tokenize1, tokenize2)
                # Pearson Similarity -- this gives very bad results (don't know why)
                # Removing it improves the log-loss
                #pearson = pearson_similarity(tokenize1, tokenize2)

                # TF-IDF Cosine Similarity
                tid = tfidf(row[i1], row[i2]) 
                if train:
                    train_data.append([int(row[0]), int(row[5]), jaccard, cosine, tid])
                else:
                    train_data.append([int(row[0]), jaccard, cosine, tid])

        return train_data

mod = 0

if len(sys.argv) != 2:
    print "Usage: python main.py <jaccard|cosine|tfidf|logistic|naivebayes|randomforest|voting>"
    sys.exit(1)

if sys.argv[1] != 'jaccard' and sys.argv[1] != 'cosine' and sys.argv[1] != 'tfidf':
    mod = 1

if mod == 1:
    print "\nParsing training data"
    print "=" * 20
    train_data = similarity("./train.csv", True)
    df_train = pd.DataFrame(train_data, columns=["id", "V", "J", "C", "T"])

    print "\nBuilding a Modal"
    print "=" * 20
    if sys.argv[1] == 'logistic':
        modal = lr()
    
    elif sys.argv[1] == 'naivebayes':
        modal = gn()

    elif sys.argv[1] == 'randomforest':
        modal = rf()

    elif sys.argv[1] == 'voting':
        modal = vc(estimators = [
            ('lr', lr()), ('rf', rf()), ('gnb', gn()), 
            ('dt', tree.DecisionTreeClassifier()), ('sgd_log', sgd(loss="log")),
            ('sgd_hinge', sgd(loss='modified_huber'))
            ], voting='soft')
    else:
        print "\nUsage: python main.py <jaccard|cosine|tfidf|logistic|naivebayes|randomforest|voting>\n"
        sys.exit(1) 
 
    modal = modal.fit(df_train[df_train.columns[2:]], df_train['V'])

print "\nParsing test data"
print "=" * 20

test_data = similarity("./test.csv", False)
df_test = pd.DataFrame(test_data, columns=["id", "J", "C", "T"])

if mod == 0:
    if sys.argv[1] == 'jaccard':
        submission = pd.DataFrame(list(df_test['J']), columns=['is_duplicate'])
    elif sys.argv[1] == 'cosine':
        submission = pd.DataFrame(list(df_test['C']), columns=['is_duplicate'])
    elif sys.argv[1] == 'tfidf':
        submission = pd.DataFrame(list(df_test['T']), columns=['is_duplicate'])
    else: 
        print "Usage: python main.py <jaccard|cosine|tfidf|logistic|naivebayes|randomforest|voting>"
        sys.exit(1)

   
else:
    print "Making Predictions"
    print "=" * 20
    pred = np.array(modal.predict_proba(df_test[df_test.columns[1:]]))

    print "Building DataFrame"
    print "=" * 20
    
    submission = pd.DataFrame(list(pred[:,1]), columns=["is_duplicate"])
   
submission.insert(0, 'test_id', df_test['id'])
submission.head()
submission_name = "final_" + sys.argv[1] +".csv"
print "Writing to " + submission_name
print "=" * 20
submission.to_csv(submission_name, index=False)



