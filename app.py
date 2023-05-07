import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define the UI using Streamlit
st.title('Machine Learning App')
file = st.file_uploader('Upload CSV or TXT file', type=['csv', 'txt'])
ml_type = st.selectbox('Select machine learning type', ['Supervised', 'Unsupervised'])
eval_methods = st.multiselect('Select evaluation methods', ['Accuracy', 'Precision', 'Recall', 'F1 Score'])

# Define helper functions
def preprocess_data(data):
    # Convert categorical variable "Education" to numerical
    le = LabelEncoder()
    data['Education'] = le.fit_transform(data['Education'])

    # Convert categorical variable "Gender" to numerical
    data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

    # Split data into features and labels
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Scale features using MinMaxScaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    return X, y

def perform_supervised_learning(X, y):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a model and make predictions
    from sklearn.svm import SVC
    clf = SVC(kernel='linear', C=1.0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Evaluate the model using selected evaluation methods
    results = []
    if 'Accuracy' in eval_methods:
        results.append(('Accuracy', accuracy_score(y_test, y_pred)))
    if 'Precision' in eval_methods:
        results.append(('Precision', precision_score(y_test, y_pred)))
    if 'Recall' in eval_methods:
        results.append(('Recall', recall_score(y_test, y_pred)))
    if 'F1 Score' in eval_methods:
        results.append(('F1 Score', f1_score(y_test, y_pred)))

    # Return the results
    return results

def perform_unsupervised_learning(X):
    # Perform clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)

    # Get cluster labels
    labels = kmeans.labels_

    # Evaluate the clusters using silhouette score
    from sklearn.metrics import silhouette_score
    score = silhouette_score(X, labels)

    # Return the results
    return [('Silhouette Score', score)]

# Preprocess the data and perform machine learning
if file is not None:
    data = pd.read_csv(file)

    # Preprocess the data
    X, y = preprocess_data(data)

    # Perform machine learning
    if ml_type == 'Supervised':
        results = perform_supervised_learning(X, y)
    else:
        results = perform_unsupervised_learning(X)

    # Display the results in a table
    st.write(pd.DataFrame(results, columns=['Evaluation Method', 'Result']))