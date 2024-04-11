from flask import Flask,request,render_template
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


app = Flask(__name__)

# route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method  == "GET":
        return render_template("home.html")
    else:
        print("Data reading from request .. ..")
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )

        prep_df = data.get_data_as_dataframe()
        # print("Here is the data : ",prep_df)

        print("Model predication started ...")
        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(prep_df)
        print("You can see result here :",result)
        return render_template('home.html',results=result[0])
    

if __name__ == "__main__":
    print("App running started...")
    app.run(debug=True)