import random
import os
import numpy as np
import torch
import torch.nn as nn
import logging
import Levenshtein
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from src import config


def set_seed(seed: int = config.SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")


def setup_logging(
    log_file: Path = config.TRAINING_LOG_FILE,
    level: int = logging.INFO
) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    logger = logging.getLogger()
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    logging.info("Logging setup complete. Messages will be sent to console and file.")
    logging.info(f"Log file location: {log_file}")


def calculate_levenshtein(predictions: List[str], ground_truths: List[str]) -> Optional[float]:
    if not predictions or not ground_truths:
        logging.warning("Cannot calculate Levenshtein: prediction or ground truth list is empty.")
        return None
    if len(predictions) != len(ground_truths):
        logging.warning(f"Cannot calculate Levenshtein: prediction list size ({len(predictions)}) "
                        f"!= ground truth list size ({len(ground_truths)}).")
        return None

    total_dis = 0
    for pr, tr in zip(predictions, ground_truths):
        total_dis += Levenshtein.distance(pr, tr)

    mean_dis = total_dis / len(predictions)
    return mean_dis


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    score: float,
    checkpoint_dir: Path = config.MODEL_CHECKPOINT_DIR,
    filename_prefix: str = "checkpoint",
    is_best: bool = False
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'score': score,
    }
    if optimizer:
        state['optimizer_state_dict'] = optimizer.state_dict()

    filepath = checkpoint_dir / f"{filename_prefix}_epoch_{epoch+1}_score_{score:.4f}.pth"
    torch.save(state, filepath)
    logging.info(f"Checkpoint saved to {filepath}")

    if is_best:
        best_filepath = checkpoint_dir / config.BEST_MODEL_FILENAME
        shutil.copyfile(filepath, best_filepath)
        logging.info(f"New best model saved to {best_filepath} (Score: {score:.4f})")


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu")
) -> Tuple[int, float]:
    if not checkpoint_path.is_file():
        logging.error(f"Checkpoint file not found: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    logging.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Загружаем состояние модели
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        logging.info("Model state loaded successfully.")
    except KeyError:
        logging.error("Checkpoint does not contain 'model_state_dict'.")
        raise KeyError("Checkpoint does not contain 'model_state_dict'.")
    except RuntimeError as e:
        logging.error(f"Error loading model state_dict: {e}")
        logging.warning("This might happen if model architecture changed or checkpoint is incompatible.")
        raise e

    if optimizer and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            logging.info("Optimizer state loaded successfully.")
        except KeyError:
            logging.warning("Checkpoint does not contain 'optimizer_state_dict', optimizer state not loaded.")
        except Exception as e:
            logging.warning(f"Could not load optimizer state: {e}")

    start_epoch = checkpoint.get('epoch', 0)
    best_score = checkpoint.get('score', float('inf'))

    logging.info(f"Checkpoint loaded. Resuming from epoch {start_epoch}. Best score was {best_score:.4f}.")

    return start_epoch, best_score
