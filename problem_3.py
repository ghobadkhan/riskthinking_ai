import pandas as pd
import dask.dataframe as dd
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
from datetime import datetime
from joblib import dump

def prepare_data(symbol_n: int | None = None):
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


if __name__ == "__main__":
    run()