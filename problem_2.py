import dask.dataframe as dd

def calculate_rolling(dask_df):
    samp = dask_df[["Symbol","Volume","Adj Close"]].compute()
    mov_avg_volume = samp[["Symbol","Volume"]].groupby("Symbol").rolling(30).mean().rename(columns={"Volume":"vol_moving_avg"})
    mov_median_adj_close = samp[["Symbol","Adj Close"]].groupby("Symbol").rolling(30).median().rename(columns={"Adj Close":"adj_close_rolling_med"})
    return dask_df.join(mov_avg_volume.join(mov_median_adj_close).reset_index(drop=True))

def run():
    for sec_type in ["stocks","etfs"]:
        df = dd.read_parquet(f"data/stage_1/{sec_type}")
        df = calculate_rolling(df)
        df.to_parquet(f"data/stage_2/{sec_type}",write_index=False)

if __name__ == "__main__":
    run()
    