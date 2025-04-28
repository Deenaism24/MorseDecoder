import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Callable, Any

from src import config
from src.audio_processing import get_audio_features
from src.text_processing import load_char_map, encode_text


class MorseDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 audio_dir: Path,
                 char_map: Dict,
                 audio_transform_fn: Callable[[Path], Optional[torch.Tensor]],
                 text_transform_fn: Callable[[str, Dict[str, int]], Optional[torch.LongTensor]],
                 is_test: bool = False):
        super().__init__()
        self.df = df.copy()
        self.audio_dir = audio_dir
        self.char_map = char_map
        self.char_to_int = char_map['char_to_int']
        self.audio_transform_fn = audio_transform_fn
        self.text_transform_fn = text_transform_fn
        self.is_test = is_test

        if 'id' not in self.df.columns:
            raise ValueError("DataFrame must contain an 'id' column.")
        if not self.is_test and 'message' not in self.df.columns:
            raise ValueError("Training DataFrame must contain a 'message' column.")

        if not self.is_test:
            self.df['message'] = self.df['message'].fillna('')

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor], int, Any]]:
        item_info = self.df.iloc[idx]
        file_id_orig = item_info['id']
        file_id_path = str(item_info['id'])

        if not file_id_path.endswith('.opus'):
            file_id_path += '.opus'
        audio_path = self.audio_dir / file_id_path

        # 1. --Обработка аудио--
        try:
            features = self.audio_transform_fn(audio_path)
        except Exception as e:
            print(f"ERROR during audio transform for {audio_path}: {e}. Skipping item {idx}.")
            return None
        if features is None: return None
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            print(f"Warning: Skipping item {idx} (id: {file_id_orig}) due to unexpected feature dimension: {features.shape if isinstance(features, torch.Tensor) else type(features)}. Expected 2D tensor. Skipping.")
            return None
        f_length = features.shape[1]
        if f_length == 0:
            print(f"Warning: Skipping item {idx} (id: {file_id_orig}) due to zero feature length after processing. Skipping.")
            return None

        # 2. --Обработка текста--
        targets = None
        if not self.is_test:
            mes = item_info['message']
            try:
                targets = self.text_transform_fn(mes, self.char_to_int)
                if targets is None: return None
            except Exception as e:
                print(f"ERROR during text transform for item {idx} (id: {file_id_orig}): {e}. Skipping.")
                return None

        return features, targets, f_length, file_id_orig


def collate_fn(batch: List[Optional[Tuple[torch.Tensor, torch.Tensor, int, Any]]]) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    batch = [item for item in batch if item is not None]
    if not batch: return None

    features_lst = [item[0] for item in batch]
    targets_lst = [item[1] for item in batch if item[1] is not None]
    f_lengths_lst = [item[2] for item in batch]

    features_lst_transp = [f.transpose(0, 1) for f in features_lst]
    padded_features_transp = pad_sequence(features_lst_transp, batch_first=True, padding_value=0.0)
    padded_features = padded_features_transp.transpose(1, 2)
    f_lengths = torch.tensor(f_lengths_lst, dtype=torch.long)

    padded_targets = None
    t_lengths = None
    if targets_lst:
        blank_id = config.CTC_BLANK_TOKEN
        target_lengths_lst = [len(t) for t in targets_lst]
        if not isinstance(blank_id, (int, float)):
            raise TypeError(f"padding_value (blank_id from config) must be number, not {type(blank_id)}")
        padded_targets = pad_sequence(targets_lst, batch_first=True, padding_value=blank_id)
        t_lengths = torch.tensor(target_lengths_lst, dtype=torch.long)

    return padded_features, padded_targets, f_lengths, t_lengths


def create_dataloaders(
    train_csv_path: Path = config.TRAIN_CSV_PATH,
    test_csv_path: Path = config.TEST_CSV_PATH,
    audio_dir: Path = config.MORSE_AUDIO_DIR,
    char_map_path: Path = config.CHAR_MAP_FILE,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    val_split: float = config.VALIDATION_SPLIT,
    seed: int = config.SEED,
    pin_memory: bool = (config.DEVICE == "cuda")
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader], Optional[Dict]]:
    print("--- Creating DataLoaders ---")

    # 1. --Загрузка карты символов--
    char_map = load_char_map(char_map_path)
    if char_map is None:
        print("Error: Failed to load character map. Cannot create DataLoaders.")
        return None, None, None, None

    # 2. --Подготовка обучающей и валидационной выборок--
    train_loader, val_loader = None, None
    try:
        train_val_df = pd.read_csv(train_csv_path)
        if not train_val_df.empty:
            print(f"Loaded {len(train_val_df)} records from {train_csv_path}")
            # Разбиение на train и validation
            if val_split > 0:
                train_df, val_df = train_test_split(
                    train_val_df,
                    test_size=val_split,
                    random_state=seed,
                )
                print(f"Splitting into Train: {len(train_df)}, Validation: {len(val_df)}")
            else:
                train_df = train_val_df
                val_df = pd.DataFrame()
                print(f"Using full dataset for training ({len(train_df)}). No validation split.")

            train_dataset = MorseDataset(
                df=train_df,
                audio_dir=audio_dir,
                char_map=char_map,
                audio_transform_fn=get_audio_features,
                text_transform_fn=encode_text,
                is_test=False
            )
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=pin_memory,
                drop_last=False
            )
            print(f"Train DataLoader created. Batches per epoch: ~{len(train_loader)}")

            if not val_df.empty:
                val_dataset = MorseDataset(
                    df=val_df,
                    audio_dir=audio_dir,
                    char_map=char_map,
                    audio_transform_fn=get_audio_features,
                    text_transform_fn=encode_text,
                    is_test=False
                )
                val_loader = DataLoader(
                    dataset=val_dataset,
                    batch_size=batch_size * 2,
                    shuffle=False,
                    num_workers=num_workers,
                    collate_fn=collate_fn,
                    pin_memory=pin_memory,
                    drop_last=False
                )
                print(f"Validation DataLoader created. Batches per epoch: ~{len(val_loader)}")
            else:
                 print("No validation set created.")

        else:
            print(f"Warning: Training CSV ({train_csv_path}) is empty or not found.")

    except FileNotFoundError:
        print(f"Warning: Training CSV ({train_csv_path}) not found.")
    except Exception as e:
        print(f"Error creating train/val DataLoaders: {e}")

    # 3. --Подготовка тестовой выборки--
    test_loader = None
    try:
        test_df = pd.read_csv(test_csv_path)
        if not test_df.empty:
            print(f"Loaded {len(test_df)} records from {test_csv_path}")
            test_dataset = MorseDataset(
                df=test_df,
                audio_dir=audio_dir,
                char_map=char_map,
                audio_transform_fn=get_audio_features,
                text_transform_fn=encode_text,
                is_test=True
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=batch_size * 2,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=pin_memory,
                drop_last=False
            )
            print(f"Test DataLoader created. Batches: ~{len(test_loader)}")
        else:
             print(f"Warning: Test CSV ({test_csv_path}) is empty or not found.")

    except FileNotFoundError:
        print(f"Warning: Test CSV ({test_csv_path}) not found.")
    except Exception as e:
        print(f"Error creating test DataLoader: {e}")

    print("--- DataLoaders creation finished ---")
    return train_loader, val_loader, test_loader, char_map

