import argparse
import logging
from pathlib import Path

from src.text_processing import build_char_map
from src.utils import setup_logging
from src import config


def main(args):
    setup_logging(log_file=config.LOG_DIR / "prepare_data.log")
    logging.info("--- Starting Data Preparation ---")

    # --- 1. Создание карты символов ---
    logging.info("Building character map from training data...")
    char_map_path = Path(args.output_map if args.output_map else config.CHAR_MAP_FILE)
    train_csv_path = config.TRAIN_CSV_PATH
    blank_token = config.CTC_BLANK_TOKEN

    logging.info(f"Using training data: {train_csv_path}")
    logging.info(f"Blank token for CTC: '{blank_token}'")
    logging.info(f"Output character map file: {char_map_path}")

    char_map_path.parent.mkdir(parents=True, exist_ok=True)

    char_map_data = build_char_map(
        train_csv_path=train_csv_path,
        save_path=char_map_path,
        blank_token=blank_token
    )

    if char_map_data:
        logging.info("Character map created successfully.")
    else:
        logging.error("Failed to create character map. Check previous error messages.")
    logging.info("--- Data Preparation Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for Morse Code Recognition training.")
    parser.add_argument(
        "-o", "--output-map",
        type=str,
        default=None,
        help=f"Path to save the generated character map JSON file (default: {config.CHAR_MAP_FILE})"
    )

    args = parser.parse_args()
    main(args)
