#!/usr/bin/env python3
import argparse
import base64
import heapq
import os
import re
import struct
import sys
from dataclasses import dataclass
from datetime import timedelta

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline

Vec = list[float]

DEFAULT_MODEL_NAME = 'large-v2'
DEFAULT_DATA_FOLDER = os.path.expanduser('~/.local/share/pod-transcribe/')
NAME_PATTERNS = [re.compile(pattern, flags=re.IGNORECASE) for pattern in [r'my name is (\w+)', r"i'm (\w+)"]]


def patch_env():
    if 'cudnn' not in os.getenv('LD_LIBRARY_PATH', ''):
        import nvidia

        path = os.path.join(os.path.dirname(str(nvidia.__file__)), 'cudnn', 'lib', '')
        os.environ['LD_LIBRARY_PATH'] = path
        print('patching env')
        os.execve(__file__, sys.argv, env=os.environ)


def encode_vec(vec: Vec) -> str:
    return base64.b64encode(struct.pack(f'{len(vec)}f', *vec)).decode('utf-8')


def decode_vec(s: str) -> Vec:
    return list(sum(struct.iter_unpack('f', base64.b64decode(s.encode('utf-8'))), tuple()))


def distance(vec_a: Vec, vec_b: Vec) -> float:
    return sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)) ** 0.5


def format_ts(ts: int) -> str:
    return str(timedelta(seconds=round(ts)))


def yn_check(prompt: str, default_yes: bool = True) -> bool:
    prompt += ' (Y/n)' if default_yes else ' (y/N)'
    while True:
        response = input(prompt).lower()
        if response in 'yn':
            return default_yes if not response else response == 'y'
        print('invalid input', response)


def get_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class Transcript:
    segments: list
    speaker_embeddings: dict[str, Vec]

    @staticmethod
    def read(filename: str, hf_token: str, model_name: str, print_progress: bool = False) -> 'Transcript':
        if not hf_token:
            with open('.hf_token') as fp:
                hf_token = fp.read().strip()
        audio = whisperx.load_audio(filename)
        model = whisperx.load_model(model_name, device=get_device())
        if print_progress:
            print('\nTranscribing..')
        transcription = model.transcribe(audio, batch_size=4, language='en', print_progress=print_progress)
        if print_progress:
            print('\nAligning..')
        align_model, metadata = whisperx.load_align_model(language_code=transcription['language'], device=get_device())
        transcription = whisperx.align(
            transcription['segments'],
            model=align_model,
            align_model_metadata=metadata,
            audio=audio,
            device=get_device(),
            return_char_alignments=False,
            print_progress=print_progress,
        )
        if print_progress:
            print('\nDiarizing..')
        diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=get_device())
        diarized_segments, embeddings = diarize_model(audio, return_embeddings=True)
        transcription = whisperx.assign_word_speakers(diarized_segments, transcription, embeddings)
        return Transcript(transcription['segments'], transcription['speaker_embeddings'])

    def find_names_in_transcript(
        self, speaker_names: dict[str, str], confirm: bool = False
    ) -> tuple[dict[str, str], dict[str, str]]:
        new_names = {}
        renames = {}
        for segment in self.segments:
            speaker = segment.get('speaker')
            if not speaker or speaker in new_names:
                continue
            for pattern in NAME_PATTERNS:
                if (match := pattern.search(segment['text'])) and match.group(1)[0].isupper():
                    name = match.group(1)
                    if confirm:
                        query = '\nIn the segment "{}" at {}, is "{}" the speaker\'s name?'.format(
                            segment['text'], format_ts(segment['start']), name
                        )
                        if not yn_check(query):
                            continue
                        if speaker in speaker_names:
                            matched_name = speaker_names[speaker]
                            if matched_name != name and yn_check(f'Replace "{name}" with "{matched_name}"?'):
                                renames[name] = speaker_names[speaker]
                                continue
                            if not yn_check(f'Update speaker from "{matched_name}"?', default_yes=False):
                                continue
                        if new_name := input(f'Input the correct spelling (or blank to keep "{name}"): ').strip():
                            renames[name] = new_name
                            name = new_name
                    else:
                        print(f'Using name "{name}" from line "{match.group(1)}"')
                        matched_name = speaker_names.get(speaker, name)
                        if matched_name != name:
                            renames[name] = speaker_names[speaker]
                    new_names[speaker] = name
                    speaker_names[speaker] = name
                    break
        return new_names, renames


class NamedEmbeddings(list[tuple[str, Vec]]):
    def closest(self, embedding: Vec, from_index: int = 0) -> tuple[str, float]:
        if len(self) == 0:
            return '', -1
        scores = ((name, distance(embedding, other)) for name, other in self[from_index:])
        return min(scores, key=lambda p: p[1])

    def match_all(self, transcript: Transcript, threshold: float = 1.0) -> dict[str, str]:
        speaker_names = {}
        block_start = -1
        for speaker, embedding in transcript.speaker_embeddings.items():
            name, score = self.closest(embedding)
            if name and score < threshold:
                if 0 < score:
                    index = self.add_embedding(name, embedding)
                    block_start = index if block_start < 0 else block_start
                speaker_names[speaker] = name
        while block_start != -1:
            from_index = block_start
            block_start = -1
            for speaker, embedding in transcript.speaker_embeddings.items():
                if speaker in speaker_names:
                    continue
                name, score = self.closest(embedding, from_index=from_index)
                if name and score < threshold:
                    if 0 < score:
                        index = self.add_embedding(name, embedding)
                        block_start = index if block_start < 0 else block_start
                        speaker_names[speaker] = name
        return speaker_names

    def add_embedding(self, name: str, embedding: Vec) -> int:
        self.append((name, embedding))
        return len(self) - 1

    @staticmethod
    def load(data_dir: str) -> 'NamedEmbeddings':
        embeddings_path = os.path.join(data_dir, 'named_embeddings.conf')
        embeddings = []
        if os.path.exists(embeddings_path):
            with open(embeddings_path) as fp:
                embeddings = [(name, decode_vec(vec)) for name, vec in map(str.split, fp.readlines())]
        return NamedEmbeddings(embeddings)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        embeddings_path = os.path.join(path, 'named_embeddings.conf')
        with open(embeddings_path, 'w') as fp:
            fp.write('\n'.join(f'{name} {encode_vec(vec)}' for name, vec in self))


def get_hf_token(data_folder: str) -> str:
    hf_path = os.path.join(data_folder, 'hf_token')
    while not os.path.exists(hf_path):
        hf_token = input('Enter hf token:\n').strip()
        if hf_token:
            os.makedirs(data_folder, exist_ok=True)
            with open(hf_path, 'w') as fp:
                fp.write(hf_token)
    with open(hf_path) as fp:
        return fp.read()


def ensure_unique(filepath: str) -> str:
    a, b = os.path.splitext(filepath)
    i = 0
    while os.path.exists(filepath):
        filepath = f'{a}.{i}{b}'
        i += 1
    return filepath


def apply_renames(text: str, renames: dict[str, str]) -> str:
    for k, v in renames.items():
        text = text.replace(k, v)
    return text


def main():
    parser = argparse.ArgumentParser(
        'Simple audio file transcription and diarization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('file', help='Input file')
    parser.add_argument('out', help='Output filename')
    parser.add_argument('--model', default=DEFAULT_MODEL_NAME, help='Model to use for ASR')
    parser.add_argument('--format', default='{s} {n}: {m}', help='Output format string')
    parser.add_argument('--data-folder', default=DEFAULT_DATA_FOLDER, help='Path for stored files')
    parser.add_argument('--show-progress', action='store_true')
    parser.add_argument('--patch', action='store_true', help='Patch LD_LIBRARY_PATH to point to cudnn')

    args = parser.parse_args()
    if args.patch:
        patch_env()
    args.out = ensure_unique(args.out or (os.path.splitext(args.file)[0] + '.txt'))
    args.format = args.format.strip() + '\n'
    embeddings = NamedEmbeddings.load(args.data_folder)
    start_len = len(embeddings)
    transcript = Transcript.read(
        filename=args.file,
        hf_token=get_hf_token(args.data_folder),
        model_name=args.model,
        print_progress=args.show_progress,
    )

    speaker_names = embeddings.match_all(transcript)
    new_names, renames = transcript.find_names_in_transcript(speaker_names, confirm=True)
    for speaker, name in new_names.items():
        embeddings.add_embedding(name, transcript.speaker_embeddings[speaker])
    for speaker in set(transcript.speaker_embeddings) - set(speaker_names):
        lines = (s for s in transcript.segments if s.get('speaker') == speaker)
        segments = heapq.nlargest(3, lines, key=lambda s: len(s['text']))
        if not segments:
            continue
        formatted = '\n'.join(f'{format_ts(s["start"])}: {s["text"]}' for s in segments)
        name = input(f'What name should be used for this speaker:\n\n{formatted}\n\n')
        speaker_names[speaker] = name
        embeddings.add_embedding(name, transcript.speaker_embeddings[speaker])

    with open(os.path.expanduser(args.out), 'w') as fp:
        for segment in transcript.segments:
            if 'speaker' not in segment:
                continue
            format_args = dict(
                s=format_ts(segment['start']),
                e=format_ts(segment['end']),
                n=speaker_names[segment['speaker']],
                m=apply_renames(segment['text'].strip(), renames),
            )
            fp.write(args.format.format(**format_args))

    print(f'Output written to {args.out}')
    if start_len != len(embeddings) and yn_check('Save new embeddings?'):
        embeddings.save(args.data_folder)


if __name__ == '__main__':
    main()
