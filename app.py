from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def predict():
    if request.method=='GET':
        return render_template('index.html')
    else:
        # Get the input values from the form
        store = int(request.form['store'])
        dept = int(request.form['dept'])
        is_holiday = request.form['holiday']
        temperature = float(request.form['temp'])
        fuel_price = float(request.form['fuel'])
        markdown1 = float(request.form['md1'])
        markdown2 = float(request.form['md2'])
        markdown3 = float(request.form['md3'])
        markdown4 = float(request.form['md4'])
        markdown5 = float(request.form['md5'])
        cpi = float(request.form['cpi'])
        unemployment = float(request.form['unemp'])
        type = request.form['type']
        size = float(request.form['size'])
        date = request.form['date']

        
        # Create a dataframe with the input values
        input_data = pd.DataFrame({
            'Store': [store],
            'Dept': [dept],
            'IsHoliday': [is_holiday],
            'Temperature': [temperature],
            'Fuel_Price': [fuel_price],
            'MarkDown1': [markdown1],
            'MarkDown2': [markdown2],
            'MarkDown3': [markdown3],
            'MarkDown4': [markdown4],
            'MarkDown5': [markdown5],
            'CPI': [cpi],
            'Unemployment': [unemployment],
            'Type': [type],
            'Size': [size],
            'Date': [date]
        })
        
        # Get the department-wide sales prediction
        predict_pipeline=PredictPipeline()
        predictions = predict_pipeline.predict(input_data)
        # print(predictions)
        return render_template('index.html', results=predictions[0])

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
