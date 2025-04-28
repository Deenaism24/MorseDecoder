import torch
import torch.nn as nn
import argparse
import logging
import pandas as pd
from pathlib import Path
import time
from tqdm.auto import tqdm
import Levenshtein

from src import config
from src.dataset import create_dataloaders
from src.model import MorseCRNN
from src.decoder import GreedyCTCDecoder, BeamSearchCTCDecoder
from src.text_processing import load_char_map, decode_indices


def evaluate_model(args):
    # --- 1. Настройка ---
    start_time = time.time()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = torch.device(args.device if args.device else config.DEVICE)
    logging.info(f"Starting model evaluation")
    logging.info(f"Using device: {device}")
    logging.info(f"Loading checkpoint from: {args.checkpoint}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        logging.error(f"Checkpoint file not found at {checkpoint_path}")
        return

    output_results_path = None
    if args.output_results:
        output_results_path = Path(args.output_results)
        output_results_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Detailed evaluation results will be saved to: {output_results_path}")

    # --- 2. Загрузка карты символов ---
    logging.info("Loading character map...")
    char_map = load_char_map(config.CHAR_MAP_FILE)
    if char_map is None:
        logging.error("Failed to load character map. Exiting.")
        return
    num_classes = char_map['vocab_size']
    blank_id = char_map['blank_id']
    int_to_char = char_map['int_to_char']
    logging.info(f"Character map loaded. Vocab size: {num_classes}, Blank ID: {blank_id}")

    # --- 3. Инициализация модели ---
    logging.info("Initializing model...")
    model = MorseCRNN(
        num_classes=num_classes,
        n_mels=config.N_MELS,
        rnn_hidden_size=config.RNN_HIDDEN_SIZE,
        rnn_num_layers=config.RNN_NUM_LAYERS,
        rnn_bidirectional=config.RNN_BIDIRECTIONAL
    ).to(device)
    logging.info(f"Model initialized: {model.__class__.__name__}")

    # --- 4. Загрузка весов модели ---
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        logging.info("Model weights loaded successfully.")
        epoch = checkpoint.get('epoch', 'N/A')
        best_score = checkpoint.get('score', 'N/A')
        logging.info(f"Checkpoint info: Epoch {epoch}, Score {best_score}")

    except Exception as e:
        logging.error(f"Failed to load model weights from {checkpoint_path}: {e}")
        return

    # --- 5. Подготовка данных для оценки (валидационный набор) ---
    logging.info("Loading validation data...")
    _, val_loader, _, _ = create_dataloaders(
        train_csv_path=config.TRAIN_CSV_PATH,
        test_csv_path=config.TEST_CSV_PATH,
        audio_dir=config.MORSE_AUDIO_DIR,
        char_map_path=config.CHAR_MAP_FILE,
        batch_size=args.batch_size if args.batch_size else config.INFERENCE_BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        val_split=config.VALIDATION_SPLIT,
        seed=config.SEED
    )

    if val_loader is None:
        logging.error("Validation loader could not be created. Check val_split > 0 in config or dataset issues.")
        return
    logging.info(f"Validation DataLoader created. Batch size: {val_loader.batch_size}, Batches: {len(val_loader)}")

    # --- 6. Инициализация декодера ---
    decoder_type = args.decoder.lower()
    if decoder_type == 'greedy':
        decoder = GreedyCTCDecoder(char_map)
    elif decoder_type == 'beam':
        beam_width = args.beam_width if args.beam_width else 10
        decoder = BeamSearchCTCDecoder(char_map, beam_width=beam_width)
        if decoder.decoder is None:
            logging.error("Beam search decoder selected, but 'ctcdecode' library is not available. Exiting.")
            return
    else:
        logging.warning(f"Unknown decoder type '{args.decoder}'. Defaulting to 'greedy'.")
        decoder = GreedyCTCDecoder(char_map)
    logging.info(f"Using decoder: {decoder.__class__.__name__}" + (f" (Beam Width: {args.beam_width})" if decoder_type == 'beam' else ""))

    # --- 7. Инициализация функции потерь (опционально) ---
    criterion = None
    if args.calculate_loss:
        criterion = nn.CTCLoss(blank=blank_id, reduction='sum', zero_infinity=True)
        logging.info("Will calculate validation loss.")

    # --- 8. Цикл оценки ---
    logging.info("Starting evaluation loop...")
    evaluation_results = []
    total_loss = 0.0
    total_samples = 0
    total_levenshtein_dist = 0
    eval_start_time = time.time()

    val_dataset = val_loader.dataset # Получаем объект Dataset
    logging.info(f"Evaluating on {len(val_dataset)} validation samples.")

    with torch.no_grad():
        progress_bar = tqdm(range(len(val_dataset)), desc="Evaluating Samples")
        for idx in progress_bar:
            try:
                item_info = val_dataset.df.iloc[idx]
                file_id = item_info['id']
                data_item = val_dataset[idx]
            except IndexError:
                logging.warning(f"Index {idx} out of bounds for validation dataset df. Skipping.")
                continue
            except Exception as e:
                logging.warning(f"Error getting item {idx} from dataset: {e}. Skipping.")
                continue

            if data_item is None:
                logging.warning(f"Skipping sample {idx} (ID: {file_id}) due to error in dataset __getitem__.")
                continue

            features, targets, feature_length, _ = data_item
            features = features.unsqueeze(0).to(device)
            targets = targets.unsqueeze(0)
            feature_lengths = torch.tensor([feature_length], dtype=torch.long).to(device)
            target_lengths = torch.tensor([targets.size(1)], dtype=torch.long)

            # Прямой проход модели
            output_log_probs = model(features)

            # Декодирование предсказаний
            prediction_text = decoder.decode(output_log_probs.cpu())[0]

            # Декодирование ground truth
            targets_cpu = targets.cpu().tolist()[0]
            target_lengths_cpu = target_lengths.cpu().tolist()[0]
            true_indices = targets_cpu[:target_lengths_cpu]
            ground_truth_text = decode_indices(true_indices, int_to_char)

            # Расчет Левенштейна
            lev_distance = Levenshtein.distance(prediction_text, ground_truth_text)
            total_levenshtein_dist += lev_distance
            total_samples += 1

            # Расчет потерь
            if criterion:
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)
                input_lengths = model.get_output_seq_len(feature_lengths.to(model.fc.weight.device))
                loss = criterion(
                 output_log_probs,
                 targets,
                 input_lengths.to(output_log_probs.device),
                 target_lengths.to(output_log_probs.device)
                )
                total_loss += loss.item()

            if output_results_path:
                evaluation_results.append({
                    "id": file_id,
                    "prediction": prediction_text,
                    "ground_truth": ground_truth_text,
                    "levenshtein": lev_distance
                })

    eval_duration = time.time() - eval_start_time
    logging.info(f"Evaluation loop finished in {eval_duration:.2f} seconds.")

    # --- 9. Расчет и вывод итоговых метрик ---
    avg_levenshtein = total_levenshtein_dist / total_samples if total_samples > 0 else float('inf')
    logging.info(f"--- Evaluation Metrics ---")
    logging.info(f"Total Samples Evaluated: {total_samples}")
    logging.info(f"Average Levenshtein Distance: {avg_levenshtein:.4f}")

    if args.calculate_loss and total_samples > 0:
        avg_loss = total_loss / total_samples
        logging.info(f"Average Validation Loss: {avg_loss:.4f}")

    # --- 10. Сохранение детальных результатов ---
    if output_results_path:
        try:
            results_df = pd.DataFrame(evaluation_results)
            results_df.to_csv(output_results_path, index=False)
            logging.info(f"Detailed evaluation results saved to {output_results_path}")
        except Exception as e:
            logging.error(f"Failed to save detailed results: {e}")

    # --- 11. Завершение ---
    total_time = time.time() - start_time
    logging.info(f"Evaluation script finished in {total_time:.2f} seconds.")

    return avg_levenshtein


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Morse Code Recognition Model")
    parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint file (.pth)"
    )
    parser.add_argument(
        "--output-results",
        type=str,
        default=None,
        help="Optional path to save detailed evaluation results (prediction, ground truth, Levenshtein) to a CSV file."
    )
    parser.add_argument(
        "--calculate-loss",
        action="store_true",
        help="Calculate and report the validation loss (requires CTCLoss)."
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
        help="Batch size for evaluation (used for DataLoader creation)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to run evaluation on ('cuda' or 'cpu')"
    )

    args = parser.parse_args()
    evaluate_model(args)