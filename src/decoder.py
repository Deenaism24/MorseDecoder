import torch
from typing import List, Dict, Any, Optional


class GreedyCTCDecoder:
    def __init__(self, char_map: Dict[str, Any]):
        if 'int_to_char' not in char_map or 'blank_id' not in char_map:
            raise ValueError("char_map must contain 'int_to_char' and 'blank_id'")

        self.int_to_char = char_map['int_to_char']
        self.blank_id = char_map['blank_id']
        print(f"Initialized GreedyCTCDecoder with Blank ID: {self.blank_id}")

    def decode(self, log_probs: torch.Tensor) -> List[str]:
        # 1. --Наиболее вероятные индексы--
        # argmax возвращает индексы максимальных значений по указанной оси
        best_path = torch.argmax(log_probs, dim=2)

        decoded_batch = []
        batch_size = best_path.size(1)

        # 2. --Итерируем по каждому элементу батча--
        for i in range(batch_size):
            sequence_indices = best_path[:, i] # [seq_len]

            # 3. --Удаляем дубликаты и blank токены--
            decoded_indices = []
            last_ind = None
            for i in sequence_indices:
                ind = i.item()
                if ind != self.blank_id and ind != last_ind:
                    decoded_indices.append(ind)
                last_ind = ind

            # 4. --Преобразуем индексы в символы--
            decoded_string = "".join([self.int_to_char.get(idx, '?') for idx in decoded_indices])
            decoded_batch.append(decoded_string)

        return decoded_batch


class BeamSearchCTCDecoder:
    def __init__(self,
                 char_map: Dict[str, Any],
                 beam_width: int = 10,
                 model_path: Optional[str] = None,  # Путь к языковой модели KenLM
                 alpha: float = 0.5,  # Вес языковой модели
                 beta: float = 1.5,  # Штраф за длину слова
                 cutoff_top_n: int = 40,
                 cutoff_prob: float = 1.0):
        try:
            from ctcdecode import CTCBeamDecoder
        except ImportError:
            print("*"*50)
            print("WARNING: ctcdecode library not found.")
            print("BeamSearchCTCDecoder will not be available.")
            print("Install it via: pip install https://github.com/parlance/ctcdecode/archive/master.zip")
            print("(May require additional dependencies like pybind11, build tools, KenLM)")
            print("*"*50)
            self.decoder = None
            return

        self.int_to_char = char_map['int_to_char']
        self.blank_id = char_map['blank_id']
        labels = [self.int_to_char.get(i, '') for i in range(len(self.int_to_char))]

        print(f"Initializing CTCBeamDecoder with Blank ID: {self.blank_id}, Beam Width: {beam_width}")
        print(f"Labels for decoder: {''.join(labels)}")

        self.decoder = CTCBeamDecoder(
            labels=labels,
            model_path=model_path,
            alpha=alpha,
            beta=beta,
            cutoff_top_n=cutoff_top_n,
            cutoff_prob=cutoff_prob,
            beam_width=beam_width,
            num_processes=4,  # Количество процессов для распараллеливания
            blank_id=self.blank_id,
            log_probs_input=True
        )

    def decode(self, log_probs: torch.Tensor) -> List[str]:
        if self.decoder is None:
            print("ERROR: ctcdecode not available. Cannot perform Beam Search.")
            batch_size = log_probs.size(1)
            return ["<BeamSearchUnavailable>"] * batch_size

        probs = torch.exp(log_probs).permute(1, 0, 2)

        batch_size, seq_len, _ = probs.shape
        s_lengths = torch.full((batch_size,), seq_len, dtype=torch.int)

        beam_res, beam_scores, timesteps, out_lens = self.decoder.decode(probs, s_lengths)

        decoded_batch = []
        for i in range(batch_size):
            best_beam_ind = beam_res[i, 0, :out_lens[i, 0]]
            decoded_string = "".join([self.int_to_char.get(idx.item(), '?') for idx in best_beam_ind])
            decoded_batch.append(decoded_string)

        return decoded_batch
