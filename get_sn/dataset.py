import pandas as pd
import csv

from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from get_sn.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = "/Users/hanalee/student_needs_repo/data/external/2024marketdata_ext.csv",
    output_path: Path = "/Users/hanalee/student_needs_repo/data/interim/2024marketdata_int.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------

#checks if the current script is being run directly as the main program, 
    # or if it's being imported as a module into another program
if __name__ == "__main__":
    # show the run time of command in terminal
    app()

market_2024 = pd.read_csv("../data/external/2024marketdata_ext.csv")

# desired column indices
df_index = [0, 1, 16, 18, 19, 24, 25, 26, 
            28, 31, 40, 50, 56, 57, 58]

# renaming selected columns
new_col = ['month', 'retail_total', 'furniture', 'appliances',
            'electronics', 'food_stores', 'grocery', 'supermarket',
            'hp_care', 'clothes', 'books', 'stationary',
            'food_service', 'restaurants', 'ls_eating']

# create dataframe with desired columns
market_df_uncleaned = pd.DataFrame()
for i in range(len(new_col)):
    market_df_uncleaned[new_col[i]] = market_2024.iloc[:, df_index[i]]

# export new, cleaned dataset to interim data folder
with open("../data/interim/2024marketdata_int.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(market_df_uncleaned)