import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_split(data):
    X = data[['math score', 'reading score', 'writing score']]
    y = data['race/ethnicity']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, le

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy}")

def main(filepath):
    data = load_data(filepath)
    X_train, X_test, y_train, y_test, le = preprocess_split(data)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    filepath = 'data/StudentsPerformance.csv'
    main(filepath)
