import argparse
import math
import os
import glob
import whisperx
import torch
import soundfile as sf
import scipy.signal as sps
import numpy as np
from g2p_en import G2p
import nltk

nltk.download('averaged_perceptron_tagger_eng')

TARGET_SR = 16000
AUDIO_EXTS = ('.wav', '.mp3', '.m4a', '.flac', '.ogg')

def load_audio(filepath: str, target_sr: int = TARGET_SR) -> np.ndarray:
    data, original_samplerate = sf.read(filepath, dtype='float32', always_2d=True)
    data = data.mean(axis=1)  # whisper takes 1-D mono; a multi-channel file is downmixed
    if original_samplerate != target_sr:
        # polyphase rather than sps.resample, whose FFT treats the signal as periodic
        # and rings where the first and last samples disagree
        step = math.gcd(original_samplerate, target_sr)
        data = sps.resample_poly(data, target_sr // step, original_samplerate // step)
    return data.astype(np.float32)

class WhisperAnnotator:
    def __init__(self, language_code: str = 'en', model_dir: str = './whisper_weights', model_name: str = 'large-v2'):
        self.model_name = model_name
        self.model_dir = model_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.compute_type = "float16" if self.device == 'cuda' else "float32"
        self.language_code = language_code

        self.asr_model = whisperx.load_model(
            model_name, self.device, language=language_code, compute_type=self.compute_type, download_root=model_dir
        )
        self.align_model, self.align_metadata = whisperx.load_align_model(language_code=language_code, device=self.device)
        self.g2p = G2p()

    def annotate(self, filepath: str) -> dict:
        data = load_audio(filepath)
        result = self.asr_model.transcribe(data)
        result = whisperx.align(result["segments"], self.align_model, self.align_metadata, data, self.device, return_char_alignments=True)
        return result

    def save_word_timings(self, result: dict, output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            for word_info in result["word_segments"]:
                word = word_info["word"].strip(".,'").lower()
                start = word_info["start"]
                end = word_info["end"]
                f.write(f"{start:.6f}\t{end:.6f}\t{word}\n")

# --- file selection and processing ---
def expand_inputs(inputs):
    """Files pass through; directories expand to the audio files directly inside them."""
    file_paths = []
    for path in inputs:
        if os.path.isdir(path):
            matches = [p for p in sorted(glob.glob(os.path.join(path, "*")))
                       if p.lower().endswith(AUDIO_EXTS)]
            if not matches:
                print(f"No audio files found in {path}")
            file_paths.extend(matches)
        else:
            file_paths.append(path)
    return file_paths


def pick_files_interactively():
    """Tk file picker. Only reachable when no paths were passed on the command line."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.update()                       # macOS needs a turn of the event loop before the panel
    file_paths = filedialog.askopenfilenames(
        title="Select audio files",
        filetypes=[("WAV files", "*.wav"),
                   ("Audio Files", " ".join(f"*{e}" for e in AUDIO_EXTS)),
                   ("All Files", "*.*")],
    )
    root.destroy()
    return list(file_paths)


def select_and_process_files(inputs=()):
    if inputs:
        file_paths = expand_inputs(inputs)
    else:
        print("No paths given; opening the file picker. Pass paths as arguments to skip it.")
        file_paths = pick_files_interactively()

    missing = [p for p in file_paths if not os.path.isfile(p)]
    if missing:
        raise SystemExit("These paths are not files:\n" + '\n'.join(missing))

    if not file_paths:
        print("No files selected.")
        return

    print(f"Selected {len(file_paths)} files for processing:\n" + '\n'.join(file_paths))
    annotator = WhisperAnnotator()
    
    for filepath in file_paths:
        print(f"Processing {filepath} ...")
        result = annotator.annotate(filepath)
        output_txt = os.path.splitext(filepath)[0] + "_labels_whisperx.txt"
        
        # with open(filepath, "r", encoding="utf-8") as f:
        #     lines = f.readlines()
        # with open(output_txt, "w", encoding="utf-8") as f_out:
        #     for line in lines:
        #         word, start, end = line.strip().split('\t')
        #         word = word.strip(',').lower()
        #         # Write back the processed word
        #         f_out.write(f"{start}\t{end}\t{word}\n")
        
        annotator.save_word_timings(result, output_txt)
        print(f"Saved timings to {output_txt}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Annotate audio files with WhisperX word timings.")
    ap.add_argument("inputs", nargs="*",
                    help="audio files and/or directories of audio files; "
                         "omit to open a file picker")
    select_and_process_files(ap.parse_args().inputs)
