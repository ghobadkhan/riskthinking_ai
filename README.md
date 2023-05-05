## Riskthinking.AI Assignment

This project is created to deliver the assignment of the RiskThinking.AI company as part of their employment process.

### STAGE 1
#### Loading the data from the source, type conversion and saving to disk

*File Name: Stage_1.py*

    One efficient way to read a large amount of data from disk is using dask dataframe.
    The logic is as follows:
    1- read the symbols metadata using normal pandas dataframe
    2- per each security type (etfs, stocks) run the following logic:
        a - read the related dataset into dask dataframe
        b - repartition the dataframe to more efficiently use the memory and io operations
        c - floor and change the type of 'Volume' column
        d - convert the added 'path' column to 'Symbol' using regex (taking only the file name in the path)
        e - merge the symbol metadata dataframe with the main one to add the corp name corresponding to each symbol
        f - save the results

### STAGE 2
#### Feature Engineering: Load the saved parquet data and calculate the rolling values

*File Name: Stage_2.py*

### STAGE 3
#### Train a machine learning model: Predict the volume based on the engineered features

*File Name: stage_3.py*

    To train a random forest regressor model with highest score, it is necessary to optimize its hyper parameter
    but the amount of data doesn't allow for this on the full data (at lest on my system). 
    The easiest solution is to sample the existing data, perform the optimization on the sample
    and then train the model on the complete dataset based on the best performing params.

### STAGE 4
#### Serve a simple application to predict the ``Volume`` base on the predictors: ```vol_moving_avg, adj_close_rolling_med```

*File Name: main.py*

    The main pipeline of all the first 3 stages is also included in this stage as a separate function run_pipeline()