from flask import Flask, request, jsonify, render_template
import pandas as pd
from model import load_data, preprocess_split, train_model, evaluate_model

app = Flask(__name__)

data = load_data('data/StudentsPerformance.csv')
X_train, X_test, y_train, y_test, label_encoder = preprocess_split(data)

model = train_model(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        math_score = request.form['math_score']
        reading_score = request.form['reading_score']
        writing_score = request.form['writing_score']
        
        prediction = model.predict([[int(math_score), int(reading_score), int(writing_score)]])
        predicted_race = label_encoder.inverse_transform(prediction)[0]
        
        return f'Predicted Race/Ethnicity: {predicted_race}'

if __name__ == '__main__':
    app.run(debug=True)
