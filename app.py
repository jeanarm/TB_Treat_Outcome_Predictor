from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.utils import load_object
application =Flask(__name__)
app = application

##Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
                  controlEnd= request.form.get('control_end'),
                  controlAfter2= request.form.get('control_after_2'),
                  controlAfter5= request.form.get('control_after_5'),
                  siteofdesease= request.form.get('site_of_disease'),
                  genexpertResult= request.form.get('genexpert_result'),
                  confirmationMethod= request.form.get('confirmation_method'),
                  smeaResult= request.form.get('smea_result'),
                  treatmenthistory = request.form.get('treatment_history'),
                  weightend=float(request.form.get('weight_end')), 
                  hospital= request.form.get('hospital'),

        )
        pred_df = data.get_data_as_data_frame()
        print("======================")
        print(pred_df)
        print("======================")
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        decoder_path= 'data/labelencoder.pkl'
        decoder = load_object(file_path=decoder_path)
        decoded_value = decoder.inverse_transform(results)
        return render_template('home.html', results = decoded_value[0])

if __name__ =="__main__":
    app.run("0.0.0.0", debug= True)




