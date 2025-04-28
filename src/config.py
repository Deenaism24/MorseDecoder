import torch
from pathlib import Path

# --- Основные Пути ---
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Пути к данным ---
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"  # Для словаря символов
MORSE_AUDIO_DIR = RAW_DATA_DIR / "morse_dataset"  # Распакованная папка с аудио
TRAIN_CSV_PATH = RAW_DATA_DIR / "train.csv"
TEST_CSV_PATH = RAW_DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = RAW_DATA_DIR / "sample_submission.csv"
LOGBOOK_PATH = RAW_DATA_DIR / "logbook.txt"

# --- Пути для вывода результатов ---
OUTPUT_DIR = BASE_DIR / "output"
MODEL_CHECKPOINT_DIR = OUTPUT_DIR / "models_lstm"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"
LOG_DIR = OUTPUT_DIR / "logs"

# --- Создание директорий, если они не существуют ---
MODEL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Имена файлов для вывода ---
CHAR_MAP_FILE = PROCESSED_DATA_DIR / "char_map.json"
BEST_MODEL_FILENAME = "best_model.pth"
SUBMISSION_FILENAME_MODEL = "submission_model_only.csv"
SUBMISSION_FILENAME_FINAL = "submission_final.csv"
TRAINING_LOG_FILE = LOG_DIR / "training.log"

# --- Параметры Аудио Обработки ---
SAMPLE_RATE = 8000  # Частота дискретизации (Гц)

# Параметры для Mel-Спектрограммы
N_MELS = 128        # Количество Mel-фильтров
N_FFT = 512         # Размер окна БПФ (Fast Fourier Transform)
HOP_LENGTH = 64    # Шаг окна (смещение между кадрами)
F_MIN = 0           # Минимальная частота для Mel-фильтров
F_MAX = SAMPLE_RATE / 2  # Максимальная частота (Найквиста)

# --- Параметры Обработки Текста ---
CTC_BLANK_TOKEN = 0

# --- Параметры RNN части ---
RNN_HIDDEN_SIZE = 256  # Размер скрытого состояния RNN
RNN_NUM_LAYERS = 2    # Количество слоев RNN
RNN_BIDIRECTIONAL = True  # Использовать двунаправленный RNN
RNN_DROPOUT = 0.1     # Dropout в RNN слоях для регуляризации

# --- Параметры Обучения ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 31           # Количество эпох обучения
BATCH_SIZE = 32       # Размер батча
LEARNING_RATE = 1e-4  # Скорость обучения
WEIGHT_DECAY = 1e-5   # L2 регуляризация
OPTIMIZER = "AdamW"   # Тип оптимизатора
GRAD_CLIP_THRESH = 1.0  # Порог для обрезки градиентов
NUM_WORKERS = 4       # Количество потоков для загрузки данных
SEED = 42             # Зерно для генераторов случайных чисел для воспроизводимости
VALIDATION_SPLIT = 0.1  # Доля данных для валидации

# --- Параметры Оценки и Предсказания ---
INFERENCE_BATCH_SIZE = BATCH_SIZE * 2
DECODING_METHOD = "greedy"  # Метод декодирования 'greedy' или 'beam'

NUM_BONUS_FILES = 17
MANUAL_DECODINGS_PATH = PROCESSED_DATA_DIR / "manual_decodings.json"  # Пример

# --- Вывод конфигурации при запуске ---
print(f"--- Configuration ---")
print(f"Project Base Dir: {BASE_DIR}")
print(f"Using Device: {DEVICE}")
print(f"Morse Audio Dir: {MORSE_AUDIO_DIR}")
print(f"Output Dir: {OUTPUT_DIR}")
print(f"Sample Rate: {SAMPLE_RATE}, N_Mels: {N_MELS}")
print(f"Batch Size: {BATCH_SIZE}, Learning Rate: {LEARNING_RATE}")
print(f"Epochs: {EPOCHS}")
print(f"--------------------")