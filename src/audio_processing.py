import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from typing import Tuple, Optional

from src import config

MEL_SPECTROGRAM_TRANSFORM = None
RESAMPLER = {}  # Словарь для ресемплеров под разные частоты


def _init_transforms():
    global MEL_SPECTROGRAM_TRANSFORM
    if MEL_SPECTROGRAM_TRANSFORM is None:
        MEL_SPECTROGRAM_TRANSFORM = T.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            win_length=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            f_min=config.F_MIN,
            f_max=config.F_MAX,
        )


def _get_resampler(original_freq: int, new_freq: int) -> T.Resample:
    global RESAMPLER
    if (original_freq, new_freq) not in RESAMPLER:
        print(f"Creating resampler from {original_freq} Hz to {new_freq} Hz")
        RESAMPLER[(original_freq, new_freq)] = T.Resample(orig_freq=original_freq, new_freq=new_freq)
    return RESAMPLER[(original_freq, new_freq)]


def load_audio(
    file_path: Path,
    target_sample_rate: int = config.SAMPLE_RATE
) -> Tuple[torch.Tensor, int]:
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        waveform, original_sample_rate = torchaudio.load(file_path)
    except Exception as e:
        raise Exception(f"Error loading audio file {file_path}: {e}")

    # 1. Преобразование в моно
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # 2. Ресемплинг
    if original_sample_rate != target_sample_rate:
        resampler = _get_resampler(original_sample_rate, target_sample_rate)
        waveform = resampler(waveform)
        cur_sample_rate = target_sample_rate
    else:
        cur_sample_rate = original_sample_rate

    return waveform, cur_sample_rate


def calculate_mel_spectrogram(
    waveform: torch.Tensor,
    apply_log: bool = True
) -> torch.Tensor:
    _init_transforms()

    mel_spec = MEL_SPECTROGRAM_TRANSFORM(waveform)

    mel_spec = mel_spec.squeeze(0)

    if apply_log:
        amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80)
        mel_spec = amplitude_to_db(mel_spec)

    return mel_spec


def normalize_spectrogram(spectrogram: torch.Tensor) -> torch.Tensor:
    mean = spectrogram.mean(dim=0, keepdim=True)  # Среднее по частотам для каждого временного шага
    std = spectrogram.std(dim=0, keepdim=True)  # Стандартное отклонение по частотам
    eps = 1e-6

    return (spectrogram - mean) / (std + eps)


def get_audio_features(
    file_path: Path,
    target_sample_rate: int = config.SAMPLE_RATE,
    apply_log: bool = True,
    apply_normalization: bool = True
) -> Optional[torch.Tensor]:
    try:
        waveform, sr = load_audio(file_path, target_sample_rate)

        if waveform.numel() == 0:
            print(f"Warning: Waveform is empty for file {file_path}. Skipping.")
            return None

        mel_spectrogram = calculate_mel_spectrogram(waveform, apply_log=apply_log)

        if apply_normalization:
            features = normalize_spectrogram(mel_spectrogram)
        else:
            features = mel_spectrogram

        if torch.isnan(features).any() or torch.isinf(features).any():
            print(f"Warning: Features contain NaN/Inf for file {file_path}. Skipping.")
            return None

        return features

    except FileNotFoundError:
        print(f"Error: File not found {file_path}. Skipping.")
        return None
    except Exception as e:
        print(f"Error processing file {file_path}: {e}. Skipping.")
        return None
