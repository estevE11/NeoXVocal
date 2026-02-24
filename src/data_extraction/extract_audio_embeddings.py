import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor


def extract_embeddings(audio_path, model, processor, device):
    speech_array, sampling_rate = torchaudio.load(audio_path)
    if speech_array.shape[0] > 1:
        speech_array = torch.mean(speech_array, dim=0)
    else:
        speech_array = speech_array.squeeze(0)

    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        speech_array = resampler(speech_array)
        sampling_rate = 16000

    speech = speech_array.numpy() 

    inputs = processor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state
    return embeddings.mean(dim=1).squeeze().cpu().numpy()


def process_audio_files(data_path, output_csv, model_name='facebook/wav2vec2-base-960h'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings = []
    for root, _, files in os.walk(data_path):
        for file in tqdm(files, desc=f'Processing {root}'):
            if not file.lower().endswith('.wav'):
                continue
            audio_path = os.path.join(root, file)
            patient_id = os.path.splitext(file)[0]
            try:
                embedding = extract_embeddings(audio_path, model, processor, device)
                all_embeddings.append({'patient_id': patient_id, 'embedding': embedding})
            except Exception as e:
                print(f'Error processing {audio_path}: {e}')

    if len(all_embeddings) == 0:
        print('No embeddings were extracted. Please check for errors.')
        return

    df = pd.DataFrame(all_embeddings)
    embedding_cols = pd.DataFrame(df['embedding'].tolist())
    embedding_cols['patient_id'] = df['patient_id']
    embedding_cols.to_csv(output_csv, index=False)


def main():
    parser = argparse.ArgumentParser(description="Extract Wav2Vec 2.0 embeddings from audio files.")
    parser.add_argument('data_path', help='Path to the directory containing .wav files.')
    parser.add_argument('--output_csv', default='audio_embeddings.csv', help='Output CSV file name.')
    parser.add_argument('--model_name', default='facebook/wav2vec2-base-960h', help='HuggingFace model id for Wav2Vec2.')
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f'Error: The directory {args.data_path} does not exist.')
        sys.exit(1)

    if os.path.exists(args.output_csv) and os.path.getsize(args.output_csv) > 0:
        print(f'Skipping: output already exists at {args.output_csv}')
        return

    process_audio_files(args.data_path, output_csv=args.output_csv, model_name=args.model_name)


if __name__ == '__main__':
    main()

