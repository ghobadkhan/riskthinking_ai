import pandas as pd
import dask.dataframe as dd
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
from datetime import datetime
from joblib import dump

"""
To train a random forest regressor model with highest score, it is necessary to optimize its hyper parameter
but the amount of data doesn't allow for this on the full data (at lest on my system). 
The easiest solution is to sample the existing data, perform the optimization on the sample
and then train the model on the complete dataset based on the best performing params.
"""

def prepare_data(symbol_n: int | None = None):
    """ This method extracts the columns "Date","Volume","vol_moving_avg", "adj_close_rolling_med" from
    the saved data and sets the "Date" as index.

    I wrote this method because most of the data preparation is repetitive. The only difference is that if
    we prepare the data for optimization, we will take a sample out of the whole data.
    To sample the data, We choose an arbitrary number of symbols to sample and then take out the entire dataset for those
    symbols. This might mitigate the risk of missing the market mega trends (e.g. economic down turns) in the trained
    model.
    """
    print("Fetching Data")
    data = dd.read_parquet("data/stage_2/stocks")

    if symbol_n:
        all_symbols = pd.read_csv("data/initial/stock_market_dataset/symbols_valid_meta.csv", usecols=["Symbol"])
        sample_symbols = all_symbols.sample(n=symbol_n)["Symbol"].to_list()
        data = data.loc[data["Symbol"].isin(sample_symbols)]

    data = data[["Date","Volume","vol_moving_avg", "adj_close_rolling_med"]].compute()
    data["Date"] = pd.to_datetime(data["Date"])
    data.set_index("Date",inplace=True)

    # Remove rows with NaN values
    data.dropna(inplace=True)
    # Select features and target
    features = ['vol_moving_avg', 'adj_close_rolling_med']
    target = 'Volume'

    X = data[features]
    y = data[target]
    return X,y


def optimize_hyper_params(X,y):
    """
    To optimize the hyper parameters I used the 'RandomizedSearchCV'
    (Cross validation for hyper parameter optimization is necessary)
    """
    
    common_params = {
        'n_estimators': [50,100,150], 
        'min_samples_leaf' : [1,2,3,4], 
        'max_depth':[5, 10, None]
    }
    print(f"Optimizing hyper parameters")
    clf = RandomizedSearchCV(estimator=RandomForestRegressor(n_jobs=-1, random_state=42), param_distributions=common_params, verbose=3, n_iter=10)
    clf.fit(X, y)

    best_params = clf.best_params_
    print(best_params)

    return best_params

def final_training(params:dict,X,y):
    """
    Although I used the hyper parameter optimization for the model, the results where under performing.
    For the final training I still used the train-test split to calculate mae and mse.
    """
    model = RandomForestRegressor(n_jobs=-1, random_state=42, verbose=1,**params)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Calculate the Mean Absolute Error and Mean Squared Error
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"mae: {mae} - mse: {mse}")

    return model

def run():
    X_sample, y_sample = prepare_data(symbol_n = 100)
    params = optimize_hyper_params(X_sample,y_sample)
    X,y = prepare_data()
    trained_model = final_training(params,X,y)
    dump(trained_model,"random_forest_trained.joblib")