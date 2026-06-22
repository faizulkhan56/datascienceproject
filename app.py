from flask import Flask, jsonify, render_template, request
import os
import numpy as np
import pandas as pd
import subprocess
import sys
from src.datascience.pipeline.prediction_pipeline import PredictionPipeline


app = Flask(__name__) # initializing a flask app
TRAINING_PROCESS = None
TRAINING_LOG_HANDLE = None


def _is_training_running():
    return TRAINING_PROCESS is not None and TRAINING_PROCESS.poll() is None


def _start_training_process():
    global TRAINING_PROCESS, TRAINING_LOG_HANDLE

    os.makedirs("logs", exist_ok=True)
    TRAINING_LOG_HANDLE = open("logs/training.log", "a", encoding="utf-8")
    TRAINING_PROCESS = subprocess.Popen(
        [sys.executable, "main.py"],
        stdout=TRAINING_LOG_HANDLE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return TRAINING_PROCESS

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    if _is_training_running():
        return (
            jsonify(
                {
                    "message": "Training is already running.",
                    "pid": TRAINING_PROCESS.pid,
                    "status_url": "/train/status",
                }
            ),
            202,
        )

    process = _start_training_process()
    return (
        jsonify(
            {
                "message": "Training started in background.",
                "pid": process.pid,
                "status_url": "/train/status",
                "log_file": "logs/training.log",
            }
        ),
        202,
    )


@app.route('/train/status', methods=['GET'])
def training_status():
    if _is_training_running():
        return jsonify({"status": "running", "pid": TRAINING_PROCESS.pid}), 200

    return jsonify({"status": "idle"}), 200


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            fixed_acidity =float(request.form['fixed_acidity'])
            volatile_acidity =float(request.form['volatile_acidity'])
            citric_acid =float(request.form['citric_acid'])
            residual_sugar =float(request.form['residual_sugar'])
            chlorides =float(request.form['chlorides'])
            free_sulfur_dioxide =float(request.form['free_sulfur_dioxide'])
            total_sulfur_dioxide =float(request.form['total_sulfur_dioxide'])
            density =float(request.form['density'])
            pH =float(request.form['pH'])
            sulphates =float(request.form['sulphates'])
            alcohol =float(request.form['alcohol'])
       
         
            data = [fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol]
            data = np.array(data).reshape(1, 11)
            
            obj = PredictionPipeline()
            predict = obj.predict(data)

            return render_template('results.html', prediction = str(predict))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')


if __name__ == "__main__":
	
	app.run(host="0.0.0.0", port = 8080)
