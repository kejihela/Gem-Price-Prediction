import pandas as pd
import numpy as np
import os
from flask import Flask, request, render_template,jsonify
from pipelines.prediction_pipeline import ModelPredict, CustomData

app=Flask(__name__)

@app.route("/")
def home_page():
    return render_template("index.html")

@app.route("/predict",methods=["GET", "POST"])
def predict_datapoint():
    if request.method=="GET":
        return render_template("form.html")

    else:
        data = CustomData(
            carat = float(request.form.get('carat')),
            depth = float(request.form.get('depth')),
            table = float(request.form.get('table')),
            x= float(request.form.get('x')),
            y= float(request.form.get('y')),
            z= float(request.form.get('z')),
            cut= request.form.get('cut'),
            color= request.form.get('color'),
            clarity= request.form.get('clarity')
        )

        final_data = data.get_data_to_dataframe()
        prediction = ModelPredict()
        result = prediction.predictPipeline(final_data)
       
        return render_template('results.html', final_result = result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9090)
