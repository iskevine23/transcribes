---
title: 'Ijambo - Speech & Translation'
emoji: 🎙️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false


---

# Ijambo: Speech & Translation

Record or upload speech to transcribe it in Kinyarwanda, English, French, or
Swahili. Optionally translate the transcript into any of those four languages.
Transcript text, subtitle (`.srt`), and translated text can be downloaded.

## Run locally

Install the packages in `requirements.txt` and the system package in
`packages.txt`, then run:

```sh
python app.py
```

The first request downloads the selected speech-recognition and translation
models. Set `HF_TOKEN` if access to the Kinyarwanda speech model requires it.

## Model notes

The Kinyarwanda speech-recognition model is a third-party Whisper checkpoint
fine-tuned on Common Voice data; results can vary with noise and speaking style.
Translation uses `facebook/nllb-200-distilled-600M`. Its model card identifies
the license as CC-BY-NC and says it is intended for research rather than
production use. Machine translations are not certified and should be reviewed,
especially for important or domain-specific content."# transcribes" 
