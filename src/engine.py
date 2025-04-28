import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import Levenshtein
from typing import Optional, Tuple, Callable, Dict, Any, List

from src import config
from src.text_processing import decode_indices

DecoderFunctionType = Callable[[torch.Tensor], List[str]]


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.CTCLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    grad_clip_thresh: Optional[float] = config.GRAD_CLIP_THRESH,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    progress_bar = tqdm(enumerate(dataloader), total=num_batches, desc=f"Epoch {epoch+1} Train")

    for i, batch in progress_bar:
        if batch is None:
            print(f"Warning: Skipping empty batch {i+1}/{num_batches}")
            continue

        features, targets, f_lengths, t_lengths = batch
        features = features.to(device)
        targets = targets.to(device)
        f_lengths = f_lengths.to(device)
        t_lengths = t_lengths.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            out_log_probs = model(features)

            in_lengths = model.get_out_seq_len(f_lengths.to(model.fc.weight.device))

            loss = criterion(
                out_log_probs,
                targets.to(out_log_probs.device),
                in_lengths.to(out_log_probs.device),
                t_lengths.to(out_log_probs.device)
            )

        if torch.isinf(loss) or torch.isnan(loss):
            print(f"Warning: Skipping batch {i+1} due to inf/nan loss: {loss.item()}")
            continue

        if scaler:
            scaler.scale(loss).backward()
            if grad_clip_thresh:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_thresh:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
            optimizer.step()

        if scheduler and isinstance(scheduler, (torch.optim.lr_scheduler.OneCycleLR, torch.optim.lr_scheduler.CyclicLR)):
            scheduler.step()

        cur_loss = loss.item()
        total_loss += cur_loss

        progress_bar.set_postfix(loss=f"{cur_loss:.4f}", avg_loss=f"{total_loss/(i+1):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    if scheduler and not isinstance(scheduler, (torch.optim.lr_scheduler.ReduceLROnPlateau, torch.optim.lr_scheduler.OneCycleLR, torch.optim.lr_scheduler.CyclicLR)):
        scheduler.step()

    return avg_loss


def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Optional[nn.CTCLoss],
    decoder: DecoderFunctionType,
    device: torch.device,
    char_map: Dict[str, Any]
) -> Tuple[Optional[float], float]:
    model.eval()
    total_loss = 0.0
    predictions = []
    ground_truths = []
    int_to_char = char_map.get('int_to_char')
    if not int_to_char:
        raise ValueError("Character map must contain 'int_to_char' mapping.")

    num_batches = len(dataloader)
    progress_bar = tqdm(dataloader, total=num_batches, desc="Evaluate")

    with torch.no_grad():
        for batch in progress_bar:
            if batch is None:
                print(f"Warning: Skipping empty evaluation batch")
                continue

            features, targets, feature_lengths, target_lengths = batch
            features = features.to(device)
            feature_lengths = feature_lengths.to(device)

            output_log_probs = model(features)

            if criterion and targets is not None and target_lengths is not None:
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)

                in_lengths = model.get_out_seq_len(feature_lengths.to(model.fc.weight.device))

                loss = criterion(
                    output_log_probs,
                    targets.to(output_log_probs.device),
                    in_lengths.to(output_log_probs.device),
                    target_lengths.to(output_log_probs.device)
                )
                total_loss += loss.item()

            # Декодирование предсказаний модели
            batch_predictions = decoder(output_log_probs.cpu())
            predictions.extend(batch_predictions)

            # Декодирование реальных меток
            if targets is not None and target_lengths is not None:
                targets_cpu = targets.cpu().tolist()
                target_lengths_cpu = target_lengths.cpu().tolist()
                for i in range(len(targets_cpu)):
                    true_indices = targets_cpu[i][:target_lengths_cpu[i]]
                    ground_truth_text = decode_indices(true_indices, int_to_char)
                    ground_truths.append(ground_truth_text)

    total_levenshtein = 0
    if ground_truths and len(predictions) == len(ground_truths):
        for pred, truth in zip(predictions, ground_truths):
            total_levenshtein += Levenshtein.distance(pred, truth)
        num_comparisons = len(predictions)
        avg_levenshtein = total_levenshtein / num_comparisons if num_comparisons > 0 else 0.0
    else:
        # Невозможно посчитать Левенштейн, если нет ground truth или размеры не совпадают
        avg_levenshtein = float('inf')
        if not ground_truths and predictions:
            print("Evaluation finished. Ground truth not available for Levenshtein calculation.")
        elif len(predictions) != len(ground_truths):
            print(f"Warning: Mismatch in number of predictions ({len(predictions)}) and ground truths ({len(ground_truths)}). Cannot calculate Levenshtein accurately.")

    avg_loss = total_loss / num_batches if criterion and num_batches > 0 else None

    print(f"Evaluation Results: Avg Loss: {avg_loss if avg_loss is not None else 'N/A'}, Avg Levenshtein: {avg_levenshtein:.4f}")
    return avg_loss, avg_levenshtein
