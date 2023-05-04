import dask.dataframe as dd

def calculate_rolling(dask_df):
    # From the loaded dask dataframe (passed as param), get the full data of these 3 columns: "Symbol","Volume","Adj Close"
    # Since the info is already in daily freq and the order is preserved, no further sorting and filtering is needed
    samp = dask_df[["Symbol","Volume","Adj Close"]].compute()
    
    # Calculation of rolling values:
    # Note: If we calculate the rolling values without grouping, the initial 29 days values of each symbol is wrongly 
    # calculated from the last symbol.
    mov_avg_volume = samp[["Symbol","Volume"]].groupby("Symbol").rolling(30).mean()\
        .rename(columns={"Volume":"vol_moving_avg"})
    mov_median_adj_close = samp[["Symbol","Adj Close"]].groupby("Symbol").rolling(30).median()\
        .rename(columns={"Adj Close":"adj_close_rolling_med"})
    # Joining the newly calculated value to the present dataframe
    return dask_df.join(mov_avg_volume.join(mov_median_adj_close).reset_index(drop=True))

def run():
    for sec_type in ["stocks","etfs"]:
        # Read the saved parquet data into dask dataframe
        df = dd.read_parquet(f"data/stage_1/{sec_type}")
        # Calculation of rolling values and joining to the current dataframe
        df = calculate_rolling(df)
        # Saving the updated dataframe
        df.to_parquet(f"data/stage_2/{sec_type}",write_index=False)
    