import pandas as pd
from dotenv import load_dotenv
import dask.dataframe as dd
import zipfile
import re
import numpy as np
import os
load_dotenv(".env")
DOWNLOADED_FILE_NAME = "stock-market-dataset.zip"
UNPACKED_FOLDER_NAME = "stock_market_dataset"
reg = re.compile(r".*?(\w*#?)\.csv")


def download_and_unpack():
    """
    I use python-dotenv package to register and import kaggle creds from a .env file.
    The sample is included.
    """

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset="jacksoncrow/stock-market-dataset", path="data/initial")
    with zipfile.ZipFile("stock-market-dataset.zip", 'r') as zip_ref:
        zip_ref.extractall("data/initial/stock_market_dataset")

def write_securities():
    """
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
    """
    symbols_df = pd.read_csv("data/initial/stock_market_dataset/symbols_valid_meta.csv",usecols=["Symbol","Security Name"])
    for sec_type, n_partitions in [("stocks",100),("etfs",30)]:
        df = dd.read_csv(f"data/initial/stock_market_dataset/{sec_type}/*.csv",include_path_column=True,assume_missing=True)
        df = df.repartition(n_partitions)
        df['Volume'] = np.floor(pd.to_numeric(df['Volume'].fillna(-1), errors='coerce')).astype('int64')
        df["path"] = df["path"].map(lambda x: reg.findall(x)[0])
        df = df.rename(columns={"path":"Symbol"})
        df = df.merge(right=symbols_df, how="left",on="Symbol")
        df.to_parquet(f"data/stage_1/{sec_type}",write_index=False)

def run():
    os.makedirs("data/initial", exist_ok=True)
    download_and_unpack()
    write_securities()
    

if __name__ == "__main__":
    run()
