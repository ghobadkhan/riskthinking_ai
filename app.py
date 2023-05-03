from flask import Flask, request
from joblib import load
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

app = Flask(__name__)
rand_forest_reg_model: RandomForestRegressor = load("random_forest_trained.joblib")

@app.route("/predict")
def prediction_test():
    # ?vol_moving_avg=12345&adj_close_rolling_med=25
    vol_moving_avg = request.args.get("vol_moving_avg", default=None, type=float)
    adj_close_rolling_med = request.args.get("adj_close_rolling_med", default=None, type=float)
    if vol_moving_avg is None or adj_close_rolling_med is None:
        return {
            "response": "ERORR! You must provide both 'vol_moving_avg' and 'adj_close_rolling_med'"
        } , 200
    
    X = pd.DataFrame(data= [[vol_moving_avg,adj_close_rolling_med]], columns=["vol_moving_avg","adj_close_rolling_med"])
    y = rand_forest_reg_model.predict(X)
    return {
        "predictors": X.to_dict('index')[0],
        "response": {
            "Volume": y[0]
        }
    } , 200