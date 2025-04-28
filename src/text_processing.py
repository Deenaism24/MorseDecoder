import json
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union

from src import config


def build_char_map(
    train_csv_path: Path = config.TRAIN_CSV_PATH,
    save_path: Path = config.CHAR_MAP_FILE,
    blank_token: str = config.CTC_BLANK_TOKEN
) -> Optional[Dict]:
    try:
        print(f"Reading training data from: {train_csv_path}")
        df = pd.read_csv(train_csv_path)

        if 'message' not in df.columns:
            print(f"Error: 'message' column not found in {train_csv_path}")
            return None

        all_chars = set()
        for message in df['message'].astype(str):
            all_chars.update(list(message))

        if blank_token in all_chars:
            print(f"Warning: CTC Blank Token '{blank_token}' found in training data characters. "
                  "Consider changing config.CTC_BLANK_TOKEN.")
        unique_chars = sorted(list(all_chars))
        print(f"Found {len(unique_chars)} unique characters in training data.")

        char_to_int: Dict[str, int] = {blank_token: 0}
        int_to_char: Dict[int, str] = {0: blank_token}

        for i, char in enumerate(unique_chars):
            index = i + 1
            char_to_int[char] = index
            int_to_char[index] = char

        vocab_size = len(char_to_int)
        blank_id = char_to_int[blank_token]

        char_map_data = {
            "char_to_int": char_to_int,
            "int_to_char": int_to_char,
            "vocab_size": vocab_size,
            "blank_id": blank_id,
            "blank_token_char": blank_token
        }

        # Сохраняем в JSON
        print(f"Saving character map to: {save_path}")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(char_map_data, f, ensure_ascii=False, indent=4)

        print(f"Character map built successfully. Vocab size: {vocab_size} (including blank)")
        return char_map_data

    except FileNotFoundError:
        print(f"Error: Training CSV file not found at {train_csv_path}")
        return None
    except Exception as e:
        print(f"An error occurred during character map building: {e}")
        return None


def load_char_map(map_path: Path = config.CHAR_MAP_FILE) -> Optional[Dict]:
    if not map_path.exists():
        print(f"Error: Character map file not found at {map_path}.")
        print("Please run build_char_map first (e.g., via scripts/prepare_data.py or directly).")
        return None

    try:
        with open(map_path, 'r', encoding='utf-8') as f:
            char_map_data = json.load(f)

        char_map_data['int_to_char'] = {int(k): v for k, v in char_map_data['int_to_char'].items()}

        print(f"Character map loaded successfully from {map_path}.")
        print(f"Vocab size: {char_map_data.get('vocab_size', 'N/A')}, Blank ID: {char_map_data.get('blank_id', 'N/A')}")
        return char_map_data

    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {map_path}.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the character map: {e}")
        return None


def encode_text(text: str, char_to_int: Dict[str, int]) -> Optional[torch.LongTensor]:
    if not text:
        return None

    encoded = []
    for char in text:
        index = char_to_int.get(char)
        if index is not None:
            encoded.append(index)
        else:
            print(f"Warning: Character '{char}' not found in char_map. Skipping.")
            pass

    if not encoded:
        return None

    return torch.LongTensor(encoded)


def decode_indices(indices: Union[List[int], torch.Tensor], int_to_char: Dict[int, str]) -> str:
    if isinstance(indices, torch.Tensor):
        indices = indices.tolist()

    decoded_chars = []
    for index in indices:
        char = int_to_char.get(index)
        if char is not None:
            decoded_chars.append(char)
        else:
            decoded_chars.append('?')

    return "".join(decoded_chars)
