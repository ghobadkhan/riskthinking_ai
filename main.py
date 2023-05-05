import pandas as pd
from flask import Flask, request
from joblib import load
from sklearn.ensemble import RandomForestRegressor
from stage_1 import run_stage_1
from stage_2 import run_stage_2
from stage_3 import run_stage_3
from logging import getLogger, config
from dotenv import load_dotenv
from os import environ, makedirs
from yaml import safe_load

load_dotenv(".env")
makedirs("logs", exist_ok=True)
with open("logging.yml", "r") as conf:
    config.dictConfig(safe_load(conf))
logger = getLogger(__name__)

app = Flask(__name__)
logger.info("Loading the trained model from random_forest_trained.joblib")
rand_forest_reg_model: RandomForestRegressor = load("random_forest_trained.joblib")

@app.route("/run-pipeline")
def run_pipeline():
    # Running this pipeline with flask server requires a task scheduler like Celery to properly run on the server
    # Using Websocket or gRPC is recommended over REST API
    logger.debug("Received pipeline run request")

    env = environ["FLASK_ENV"]
    if env != 'prod':
        logger.warn("Environment is not set to 'production'. Operation aborted.")
        return "Running the entire pipeline is only allowed in the production mode!", 400
    
    logger.info("Running the stage 1")
    run_stage_1()
    logger.info("Running the stage 2")
    run_stage_2()
    logger.info("Running the stage 3")
    run_stage_3()

    logger.debug("Pipeline running finished")
    return "Pipeline running finished", 200


@app.route("/predict")
def prediction_test():
    vol_moving_avg = request.args.get("vol_moving_avg", default=None, type=float)
    adj_close_rolling_med = request.args.get("adj_close_rolling_med", default=None, type=float)
    if vol_moving_avg is None or adj_close_rolling_med is None:
        return {
            "response": "ERORR! You must provide both 'vol_moving_avg' and 'adj_close_rolling_med'"
        } , 404
    
    X = pd.DataFrame(data= [[vol_moving_avg,adj_close_rolling_med]], columns=["vol_moving_avg","adj_close_rolling_med"])
    logger.debug(f"Received prediction request with queries {X.to_dict('index')[0]}")
    y = rand_forest_reg_model.predict(X)
    logger.debug(f"Calculated Volume = {y}")
    return {
        "predictors": X.to_dict('index')[0],
        "response": {
            "Volume": y[0]
        }
    } , 200