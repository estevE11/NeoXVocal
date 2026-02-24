import argparse
import os
import sys

import whisper


def transcribe_audio_files(data_path: str, output_dir: str | None = None, model_name: str = 'base') -> None:
    model = whisper.load_model(model_name)
    output_base = output_dir or os.path.join(os.path.dirname(data_path), 'extracted_data')

    for root, _, files in os.walk(data_path):
        for file in files:
            if not file.lower().endswith('.wav'):
                continue
            audio_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, data_path)
            out_dir = os.path.join(output_base, relative_path)
            os.makedirs(out_dir, exist_ok=True)
            output_file = os.path.join(out_dir, os.path.splitext(file)[0] + '.txt')
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                continue
            result = model.transcribe(audio_path)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result['text'])


def main():
    parser = argparse.ArgumentParser(description='Transcribe .wav files to text using Whisper.')
    parser.add_argument('data_path', help='Path to the directory containing .wav files.')
    parser.add_argument('--output_dir', default=None, help='Base output directory for .txt files.')
    parser.add_argument('--model', default='base', help='Whisper model name (e.g. tiny, base, small, medium, large).')
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f'Error: The directory {args.data_path} does not exist.')
        sys.exit(1)

    transcribe_audio_files(args.data_path, output_dir=args.output_dir, model_name=args.model)


if __name__ == '__main__':
    main()

