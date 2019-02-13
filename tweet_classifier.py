import numpy as np

def load_data():
    data = []
    data_labels = []
    with open("./pos_tweets.txt", encoding='utf-8') as f:
        for i in f: 
            data.append(i) 
            data_labels.append('pos')

    with open("./neg_tweets.txt", encoding='utf-8') as f:
        for i in f: 
            data.append(i)
            data_labels.append('neg')

    return data, data_labels

def transform_to_features(data):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(
        analyzer = 'word',
        lowercase = False,
    )
    features = vectorizer.fit_transform(
        data
    )
    features_nd = features.toarray()
    return features_nd

def train_then_build_model(data_labels, features_nd, data):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test  = train_test_split(
        features_nd, 
        data_labels,
        train_size=0.80, 
        random_state=1234)

    from sklearn.linear_model import LogisticRegression
    log_model = LogisticRegression()

    log_model = log_model.fit(X=X_train, y=y_train)
    y_pred = log_model.predict(X_test)

    
    # Define list objects for evaluating accuracy
    this_y_pred = []
    this_y_true = []
    
    for x in range(50):
        for index, my_array in enumerate(features_nd):
            if np.array_equal(X_test[x], my_array):
                # Printing first 10 predictions
                if x < 10:
                    print('::{0}::{1}'.format(y_pred[x], data[index]))
                this_y_pred.append(y_pred[x])
                this_y_true.append(data_labels[index])
    
    from sklearn.metrics import accuracy_score
    this_acc = accuracy_score(this_y_true, this_y_pred)
    print("Accuracy={}".format(this_acc))

def process():
    data, data_labels = load_data()
    features_nd = transform_to_features(data)
    train_then_build_model(data_labels, features_nd, data)

process()