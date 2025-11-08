## pod-transcribe

Generates basic multi-speaker transcripts from audio files using [whisperx](https://github.com/m-bain/whisperX). Optionally stores labelled speaker embeddings to improve matching on future files. Requires a read permissioned Hugging Face token for diarization.

### Usage

```
transcribe recording.mp3 transcript.txt --show-progress
```
