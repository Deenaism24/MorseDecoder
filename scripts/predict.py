import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import logging
import pandas as pd
from pathlib import Path
import time
from tqdm.auto import tqdm
from typing import List, Tuple, Any

from src import config
from src.dataset import MorseDataset
from src.model import MorseCRNN
from src.decoder import GreedyCTCDecoder, BeamSearchCTCDecoder
from src.text_processing import load_char_map
from src.audio_processing import get_audio_features
from src.text_processing import encode_text


def collate_fn_predict(batch: List[Tuple[torch.Tensor, Any, int, str]]):
    batch = [item for item in batch if item is not None and item[0] is not None]
    if not batch:
        return None, None, None

    features_list = [item[0] for item in batch] # [n_mels, time]
    feature_lengths_list = [item[2] for item in batch]
    ids_list = [item[3] for item in batch] # file_id

    features_list_transposed = [f.transpose(0, 1) for f in features_list]
    padded_features_transposed = nn.utils.rnn.pad_sequence(features_list_transposed, batch_first=True, padding_value=0.0)
    padded_features = padded_features_transposed.transpose(1, 2)

    feature_lengths = torch.tensor(feature_lengths_list, dtype=torch.long)

    return padded_features, feature_lengths, ids_list


def predict(args):
    # --- 1. Настройка ---
    start_time = time.time()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = torch.device(args.device if args.device else config.DEVICE)
    logging.info(f"Starting prediction run")
    logging.info(f"Using device: {device}")
    logging.info(f"Loading checkpoint from: {args.checkpoint}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        logging.error(f"Checkpoint file not found at {checkpoint_path}")
        return

    output_path = Path(args.output if args.output else config.SUBMISSION_DIR / config.SUBMISSION_FILENAME_MODEL)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output submission file: {output_path}")

    # --- 2. Загрузка карты символов ---
    logging.info("Loading character map...")
    char_map = load_char_map(config.CHAR_MAP_FILE)
    if char_map is None:
        logging.error("Failed to load character map. Exiting.")
        return
    num_classes = char_map['vocab_size']
    blank_id = char_map['blank_id']
    logging.info(f"Character map loaded. Vocab size: {num_classes}, Blank ID: {blank_id}")

    # --- 3. Инициализация модели ---
    logging.info("Initializing model...")
    model = MorseCRNN(
        num_classes=num_classes,
        n_mels=config.N_MELS,
        rnn_hidden_size=config.RNN_HIDDEN_SIZE,
        rnn_num_layers=config.RNN_NUM_LAYERS,
        rnn_bidirectional=config.RNN_BIDIRECTIONAL
        # Dropout не важен для eval
    ).to(device)
    logging.info(f"Model initialized: {model.__class__.__name__}")

    # --- 4. Загрузка весов модели ---
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        logging.info("Model weights loaded successfully.")
        best_score = checkpoint.get('score', 'N/A')
        logging.info(f"Checkpoint score (e.g., Levenshtein/loss): {best_score}")
    except Exception as e:
        logging.error(f"Failed to load model weights from {checkpoint_path}: {e}")
        return

    # --- 5. Подготовка тестовых данных ---
    logging.info("Loading test data...")
    try:
        test_df = pd.read_csv(config.TEST_CSV_PATH)
        if test_df.empty:
            logging.error(f"Test CSV file is empty: {config.TEST_CSV_PATH}")
            return
        logging.info(f"Found {len(test_df)} samples in test data.")

        test_dataset = MorseDataset(
            df=test_df,
            audio_dir=config.MORSE_AUDIO_DIR,
            char_map=char_map,
            audio_transform_fn=get_audio_features,
            text_transform_fn=encode_text,
            is_test=True
        )

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size if args.batch_size else config.INFERENCE_BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            collate_fn=collate_fn_predict,
            pin_memory=(device.type == 'cuda')
        )
        logging.info(f"Test DataLoader created. Batch size: {test_loader.batch_size}, Batches: {len(test_loader)}")

    except FileNotFoundError:
        logging.error(f"Test CSV file not found at {config.TEST_CSV_PATH}")
        return
    except Exception as e:
        logging.error(f"Error creating test DataLoader: {e}")
        return

    # --- 6. Инициализация декодера ---
    decoder_type = args.decoder.lower()
    if decoder_type == 'greedy':
        decoder = GreedyCTCDecoder(char_map)
    elif decoder_type == 'beam':
        beam_width = args.beam_width if args.beam_width else 10
        decoder = BeamSearchCTCDecoder(char_map, beam_width=beam_width)
        if decoder.decoder is None:
             logging.error("Beam search decoder selected, but 'ctcdecode' library is not available. Install it or use '--decoder greedy'. Exiting.")
             return
    else:
        logging.warning(f"Unknown decoder type '{args.decoder}'. Defaulting to 'greedy'.")
        decoder = GreedyCTCDecoder(char_map)
    logging.info(f"Using decoder: {decoder.__class__.__name__}" + (f" (Beam Width: {args.beam_width})" if decoder_type == 'beam' else ""))


    # --- 7. Цикл предсказания ---
    logging.info("Starting inference...")
    predictions = [] # Список для хранения словарей {'id': ..., 'message': ...}
    inference_start_time = time.time()

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Predicting")
        for batch in progress_bar:
            # collate_fn_predict возвращает (features, feature_lengths, ids_list)
            features, feature_lengths, ids_list = batch

            if features is None or ids_list is None:
                logging.warning("Skipping potentially empty batch.")
                continue

            features = features.to(device)
            output_log_probs = model(features)

            # Декодирование
            batch_preds_text = decoder.decode(output_log_probs.cpu())

            # Сохраняем результаты батча
            if len(ids_list) != len(batch_preds_text):
                 logging.error(f"Mismatch in batch size between IDs ({len(ids_list)}) and predictions ({len(batch_preds_text)}). Skipping batch.")
                 continue

            for file_id, message in zip(ids_list, batch_preds_text):
                predictions.append({"id": file_id, "message": message})

    inference_duration = time.time() - inference_start_time
    logging.info(f"Inference finished in {inference_duration:.2f} seconds.")

    if len(predictions) != len(test_df):
        logging.warning(f"Number of predictions ({len(predictions)}) does not match number of test samples ({len(test_df)}). Some samples might have been skipped due to errors.")

    # --- 8. Создание и сохранение файла submission.csv ---
    logging.info("Creating submission file...")
    submission_df = pd.DataFrame(predictions)

    submission_df['id_cat'] = pd.Categorical(submission_df['id'], categories=test_df['id'].tolist(), ordered=True)
    submission_df = submission_df.sort_values('id_cat')
    submission_df = submission_df.drop(columns=['id_cat'])

    # Проверяем колонки
    if list(submission_df.columns) != ['id', 'message']:
        logging.error(f"Submission DataFrame has incorrect columns: {submission_df.columns}. Expected ['id', 'message']")
        if set(submission_df.columns) == {'id', 'message'}:
            submission_df = submission_df[['id', 'message']]
        else:
            return # Не можем исправить

    try:
        submission_df.to_csv(output_path, index=False)
        logging.info(f"Submission file saved successfully to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save submission file: {e}")

    # --- 9. Завершение ---
    total_time = time.time() - start_time
    logging.info(f"Prediction script finished in {total_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions for Morse Code Recognition")
    parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint file (.pth)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help=f"Path to save the output submission CSV file (default: {config.SUBMISSION_DIR / config.SUBMISSION_FILENAME_MODEL})"
    )
    parser.add_argument(
        "-d", "--decoder",
        type=str,
        default="greedy",
        choices=["greedy", "beam"],
        help="Decoder type to use ('greedy' or 'beam')"
    )
    parser.add_argument(
        "-bw", "--beam_width",
        type=int,
        default=None,
        help="Beam width for Beam Search decoder (default: 10)"
    )
    parser.add_argument(
        "-bs", "--batch_size",
        type=int,
        default=None,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to run inference on ('cuda' or 'cpu')"
    )

    args = parser.parse_args()
    predict(args)
