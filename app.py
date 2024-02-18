from flask import Flask, request, jsonify
import pandas as pd
from model import load_data, preprocess_split, train_model, evaluate_model

app = Flask(__name__)

data = load_data('data/StudentsPerformance.csv')
X_train, X_test, y_train, y_test, label_encoder = preprocess_split(data)

model = train_model(X_train, y_train)

@app.route('/')
def home():
    return "Student Performance Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    math_score = data['math_score']
    reading_score = data['reading_score']
    writing_score = data['writing_score']
    
    prediction = model.predict([[math_score, reading_score, writing_score]])
    predicted_race = label_encoder.inverse_transform(prediction)[0]
    
    return jsonify(predicted_race=predicted_race)

if __name__ == '__main__':
    app.run(debug=True)
