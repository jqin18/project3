from flask import Flask, request, render_template
import pandas as pd
import pickle
import prophet
# from prophet.serialize import model_to_json, model_from_json
import json
# from flask_cors import CORS, cross_origin

app = Flask(__name__)
model = pickle.load(open('forecast_model.pckl', 'rb'))  # loading the model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    """Grabs the input values and uses them to make prediction"""
    year = str(request.form['year'])
    future_date = pd.DataFrame({'ds':[year]})
    prediction = model.predict(future_date)
    data = prediction[["ds", "yhat"]]
    
    output = round(data["yhat"][0], 2)


    return render_template('index.html', prediction_text=f'It will be average {output} °C globally on {year}')

if __name__ == "__main__":
    app.run(debug=True)

