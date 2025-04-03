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
    output_path: Path = "/Users/hanalee/student_needs_repo/data/interim/student_spending_int.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------

main()
#checks if the current script is being run directly as the main program, 
    # or if it's being imported as a module into another program
if __name__ == "__main__":
    # show the run time of command in terminal
    app()
