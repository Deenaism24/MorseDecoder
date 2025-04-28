import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
import argparse
import logging
from pathlib import Path
import time

from src import config
from src.dataset import create_dataloaders
from src.model import MorseCRNN
from src.engine import train_epoch, evaluate_epoch
from src.decoder import GreedyCTCDecoder
from src.utils import (
    set_seed,
    setup_logging,
    save_checkpoint,
    load_checkpoint
)


def main(args):
    # --- 1. Настройка ---
    st_time = time.time()

    output_dir = config.OUTPUT_DIR
    if args.tag:
        output_dir = output_dir / args.tag
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / "training.log"
        checkpoint_dir = output_dir / "models_lstm"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    else:
        log_file = config.TRAINING_LOG_FILE
        checkpoint_dir = config.MODEL_CHECKPOINT_DIR
        log_file.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_file=log_file)
    set_seed(config.SEED)
    device = torch.device(config.DEVICE)
    logging.info(f"Starting training run with tag: {args.tag if args.tag else 'default'}")
    logging.info(f"Using device: {device}")
    logging.info(f"Output directory: {output_dir}")

    use_amp = (device.type == 'cuda')
    scaler = GradScaler(enabled=use_amp)
    logging.info(f"Using Automatic Mixed Precision (AMP): {use_amp}")

    # --- 2. Загрузка данных ---
    logging.info("Loading data...")
    train_loader, val_loader, _, char_map = create_dataloaders(
        train_csv_path=config.TRAIN_CSV_PATH,
        test_csv_path=config.TEST_CSV_PATH,
        audio_dir=config.MORSE_AUDIO_DIR,
        char_map_path=config.CHAR_MAP_FILE,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        val_split=config.VALIDATION_SPLIT,
        seed=config.SEED
    )

    if char_map is None or train_loader is None:
        logging.error("Failed to load character map or training data. Exiting.")
        return

    if val_loader is None:
        logging.warning("Validation loader not created (val_split might be 0). Validation metrics will not be available.")

    num_classes = char_map['vocab_size']
    blank_id = char_map['blank_id']
    logging.info(f"Data loaded. Vocab size: {num_classes}, Blank ID: {blank_id}")
    logging.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader) if val_loader else 'N/A'}")

    # --- 3. Инициализация модели ---
    logging.info("Initializing model...")
    model = MorseCRNN(
        num_classes=num_classes,
        n_mels=config.N_MELS,
        rnn_hidden_size=config.RNN_HIDDEN_SIZE,
        rnn_num_layers=config.RNN_NUM_LAYERS,
        rnn_bidirectional=config.RNN_BIDIRECTIONAL,
        rnn_dropout=config.RNN_DROPOUT
    ).to(device)
    logging.info(f"Model initialized: {model.__class__.__name__}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total Parameters: {total_params:,}")
    logging.info(f"Trainable Parameters: {trainable_params:,}")

    # --- 4. Функция потерь и Оптимизатор ---
    criterion = nn.CTCLoss(blank=blank_id, reduction='mean', zero_infinity=True)
    logging.info(f"Loss function: CTCLoss (blank={blank_id}, zero_infinity=True)")

    if config.OPTIMIZER.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0)
    elif config.OPTIMIZER.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    else:
        logging.warning(f"Unsupported optimizer '{config.OPTIMIZER}', defaulting to AdamW.")
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    logging.info(f"Optimizer: {optimizer.__class__.__name__} (LR={config.LEARNING_RATE}, WD={config.WEIGHT_DECAY if config.OPTIMIZER.lower() == 'adamw' else 'N/A'})")

    # --- 5. Планировщик скорости обучения ---
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.2,
        patience=5,
        verbose=True,
        min_lr=1e-7
    )
    logging.info(f"Scheduler: ReduceLROnPlateau (mode='min', factor={scheduler.factor}, patience={scheduler.patience})")

    # --- 6. Декодер для оценки ---
    decoder = GreedyCTCDecoder(char_map)
    logging.info(f"Decoder for evaluation: {decoder.__class__.__name__}")

    # --- 7. Загрузка чекпоинта ---
    st_epoch = 0
    best_metric = float('inf')

    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.is_file():
            try:
                st_epoch, best_metric = load_checkpoint(
                    checkpoint_path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    device=device
                )
                logging.info(f"Resuming training from epoch {st_epoch}. Best previous metric: {best_metric:.4f}")
            except Exception as e:
                logging.error(f"Failed to load checkpoint from {args.resume}: {e}")
                logging.warning("Starting training from scratch.")
        else:
            logging.warning(f"Resume checkpoint not found at {args.resume}. Starting training from scratch.")

    # --- 8. Цикл обучения ---
    logging.info("Starting training loop...")
    for epoch in range(st_epoch, config.EPOCHS):
        epoch_st_time = time.time()
        logging.info(f"--- Epoch {epoch+1}/{config.EPOCHS} ---")

        # Обучение
        avg_train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scheduler=None,
            grad_clip_thresh=config.GRAD_CLIP_THRESH,
            scaler=scaler if use_amp else None
        )
        logging.info(f"Epoch {epoch+1} Train Avg Loss: {avg_train_loss:.4f}")

        if val_loader:
            avg_val_loss, val_levenshtein = evaluate_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                decoder=decoder.decode,
                device=device,
                char_map=char_map
            )
            logging.info(f"Epoch {epoch+1} Validation Avg Loss: {avg_val_loss if avg_val_loss is not None else 'N/A'}, Avg Levenshtein: {val_levenshtein:.4f}")
        else:
            val_levenshtein = avg_train_loss
            logging.warning("No validation set. Using train loss for scheduler and checkpointing.")

        scheduler.step(val_levenshtein)

        # Сохранение чекпоинта
        if val_levenshtein < best_metric:
            best_metric = val_levenshtein
            logging.info(f"*** New best metric found: {best_metric:.4f} ***")

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            score=val_levenshtein,
            checkpoint_dir=checkpoint_dir,
            filename_prefix=f"morse_crnn{'_' + args.tag if args.tag else ''}",
            is_best=val_levenshtein < best_metric
        )

        epoch_dur = time.time() - epoch_st_time
        logging.info(f"Epoch {epoch+1} finished in {epoch_dur:.2f} seconds.")

    # --- 9. Завершение ---
    total_training_time = time.time() - st_time
    logging.info("--- Training Finished ---")
    logging.info(f"Total training time: {total_training_time:.2f} seconds ({total_training_time/3600:.2f} hours)")
    logging.info(f"Best validation Levenshtein score achieved: {best_metric:.4f}")
    logging.info(f"Model checkpoints saved in: {checkpoint_dir}")
    logging.info(f"Logs saved in: {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Morse Code Recognition CRNN Model")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from (e.g., output/models_lstm/best_model.pth)"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag for the training run (creates a subdirectory in output)"
    )
    args = parser.parse_args()
    main(args)
