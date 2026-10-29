"""Speech transcription and translation web app (Whisper + NLLB + Gradio).

Works in three places with the same file:
  - your laptop / Colab:   python app.py   (set SHARE=1 for a public link)
  - Hugging Face ZeroGPU:  just upload it (the @spaces.GPU decorator is used)
"""
import os
import re
import tempfile
import threading

import gradio as gr
import librosa
import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

try:
    import spaces  # only present on Hugging Face Spaces

    gpu = spaces.GPU(duration=120)
except ImportError:

    def gpu(fn):
        return fn


ON_SPACE = bool(os.environ.get("SPACE_ID"))
SR = 16000
MAX_SEG = 28  # seconds; Whisper handles up to 30 s per window
MAX_AUDIO_S = 180  # cap on Spaces so one request stays inside the GPU time limit

# label -> (model id, language code passed to Whisper)
# The Kinyarwanda model's card says to use language="sw".
CHOICES = {
    "Kinyarwanda": ("mbazaNLP/Whisper-Small-Kinyarwanda", "sw"),
    "English": ("openai/whisper-small", "en"),
    "French": ("openai/whisper-small", "fr"),
    "Swahili": ("openai/whisper-small", "sw"),
}
TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"
NLLB_CODES = {
    "Kinyarwanda": "kin_Latn",
    "English": "eng_Latn",
    "French": "fra_Latn",
    "Swahili": "swh_Latn",
}

_pipes = {}
_translation_tokenizer = None
_translation_model = None
_translation_lock = threading.Lock()
DEVICE = 0 if torch.cuda.is_available() else -1
MODEL_DEVICE = torch.device("cuda:0" if DEVICE >= 0 else "cpu")


def get_pipe(model_id):
    if model_id not in _pipes:
        _pipes[model_id] = pipeline(
            "automatic-speech-recognition", model=model_id, device=DEVICE
        )
    return _pipes[model_id]


if ON_SPACE:  # load models once at startup so requests don't wait on downloads
    for _mid in sorted({m for m, _ in CHOICES.values()}):
        try:
            get_pipe(_mid)
        except Exception as _e:  # e.g. gated model without access/token yet
            print(f"WARNING: could not preload {_mid}: {_e}")


def split_audio(y, top_db=30):
    """Cut on silences, merge into segments <= MAX_SEG seconds.
    Returns a list of (start_sample, end_sample)."""
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) == 0:
        return []
    max_len = MAX_SEG * SR
    segs = []
    cur_s, cur_e = int(intervals[0][0]), int(intervals[0][1])
    for s, e in intervals[1:]:
        s, e = int(s), int(e)
        if e - cur_s <= max_len:
            cur_e = e
        else:
            segs.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    segs.append((cur_s, cur_e))
    out = []
    for s, e in segs:  # hard-split anything still too long
        while e - s > max_len:
            out.append((s, s + max_len))
            s += max_len
        out.append((s, e))
    return out


def srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def run_asr(pipe, audio, lang):
    try:
        res = pipe(
            {"raw": audio, "sampling_rate": SR},
            generate_kwargs={"language": lang, "task": "transcribe"},
        )
    except Exception:
        # Some fine-tuned checkpoints lack the language table: force the
        # decoder prompt directly instead.
        pipe.model.config.forced_decoder_ids = (
            pipe.tokenizer.get_decoder_prompt_ids(language=lang, task="transcribe")
        )
        res = pipe({"raw": audio, "sampling_rate": SR})
    return res["text"].strip()


def split_translation_text(text, tokenizer, max_tokens=450):
    """Split long transcripts without silently truncating translation input."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current = ""

    def token_count(value):
        return len(tokenizer(value, add_special_tokens=True)["input_ids"])

    for sentence in sentences:
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip()
        if token_count(candidate) <= max_tokens:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        for word in sentence.split():
            candidate = f"{current} {word}".strip()
            if token_count(candidate) > max_tokens:
                if not current:
                    raise ValueError("A word exceeds the translation model's input limit.")
                chunks.append(current)
                current = word
                if token_count(current) > max_tokens:
                    raise ValueError("A word exceeds the translation model's input limit.")
            else:
                current = candidate

    if current:
        chunks.append(current)
    return chunks


def translate_text(text, source, target):
    global _translation_tokenizer, _translation_model

    with _translation_lock:
        if _translation_tokenizer is None or _translation_model is None:
            _translation_tokenizer = AutoTokenizer.from_pretrained(
                TRANSLATION_MODEL, src_lang=NLLB_CODES[source]
            )
            _translation_model = AutoModelForSeq2SeqLM.from_pretrained(
                TRANSLATION_MODEL
            ).to(MODEL_DEVICE)
            _translation_model.eval()

        tokenizer = _translation_tokenizer
        model = _translation_model
        tokenizer.src_lang = NLLB_CODES[source]
        target_id = tokenizer.convert_tokens_to_ids(NLLB_CODES[target])
        translated = []

        for chunk in split_translation_text(text, tokenizer):
            inputs = tokenizer(chunk, return_tensors="pt")
            inputs = {key: value.to(MODEL_DEVICE) for key, value in inputs.items()}
            with torch.inference_mode():
                output = model.generate(
                    **inputs, forced_bos_token_id=target_id, max_length=512
                )
            translated.append(
                tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            )

        return " ".join(part.strip() for part in translated if part.strip())


@gpu
def translate_segments(texts, source, target):
    return [translate_text(text, source, target) for text in texts]


@gpu
def transcribe_segments(y, segs, model_id, lang):
    """The only part that needs the GPU. Returns [(start, end, text), ...]."""
    pipe = get_pipe(model_id)
    out = []
    for s, e in segs:
        text = run_asr(pipe, y[s:e].astype(np.float32), lang)
        if text:
            out.append((s, e, text))
    return out


def transcribe(audio_path, choice, target):
    if audio_path is None:
        return "Please record or upload an audio file first.", "", None, None, None

    model_id, lang = CHOICES[choice]
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    if len(y) < SR * 0.3:
        return "Audio is too short.", "", None, None, None
    if ON_SPACE and len(y) > MAX_AUDIO_S * SR:
        return (
            f"Audio is longer than {MAX_AUDIO_S // 60} minutes. "
            "Please upload a shorter clip (this demo's GPU time is limited).",
            "",
            None,
            None,
            None,
        )

    segs = split_audio(y)
    if not segs:
        return "No speech detected.", "", None, None, None

    try:
        results = transcribe_segments(y, segs, model_id, lang)
    except Exception as e:
        return (
            f"Could not run the {choice} model: {e}\n"
            "If this is the Kinyarwanda model, access to it (and an HF_TOKEN) may be missing.",
            "",
            None,
            None,
            None,
        )
    if not results:
        return "No speech detected.", "", None, None, None

    full = " ".join(t for _, _, t in results)
    srt = [
        f"{i}\n{srt_time(s / SR)} --> {srt_time(e / SR)}\n{t}\n"
        for i, (s, e, t) in enumerate(results, 1)
    ]
    d = tempfile.mkdtemp()
    txt_path = os.path.join(d, "transcript.txt")
    srt_path = os.path.join(d, "transcript.srt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full)
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt))

    translation = ""
    translated_path = None
    if target != "No translation":
        try:
            if target == choice:
                translation = full
            else:
                translation = " ".join(
                    translate_segments(
                        [text for _, _, text in results], choice, target
                    )
                )
            translated_path = os.path.join(d, "translation.txt")
            with open(translated_path, "w", encoding="utf-8") as f:
                f.write(translation)
        except Exception as e:
            translation = f"Translation failed: {e}"

    return full, translation, txt_path, srt_path, translated_path


with gr.Blocks(title="Ijambo - Speech & Translation") as demo:
    gr.Markdown(
        "# Ijambo\n"
        "Transcribe speech and translate between Kinyarwanda, English, French, and Swahili. "
        "Record or upload audio, choose its spoken language, and optionally select a translation target. "
        "Kinyarwanda uses a Whisper-small model fine-tuned by a third party on "
        "Common Voice data, so accuracy drops on noisy audio and on speech unlike "
        "its training data. Subtitle timestamps are per speech segment, not per word. "
        "Translations are machine-generated and may need review."
    )
    with gr.Row():
        audio = gr.Audio(
            sources=["upload", "microphone"], type="filepath", label="Audio"
        )
        with gr.Column():
            lang = gr.Dropdown(
                list(CHOICES), value="Kinyarwanda", label="Spoken language"
            )
            target = gr.Dropdown(
                ["No translation", *CHOICES],
                value="No translation",
                label="Translate to",
            )
            btn = gr.Button("Transcribe audio", variant="primary")
    out = gr.Textbox(label="Transcript", lines=10, show_copy_button=True)
    translated = gr.Textbox(label="Translation", lines=10, show_copy_button=True)
    with gr.Row():
        f_txt = gr.File(label="Download .txt")
        f_srt = gr.File(label="Download .srt")
        f_translation = gr.File(label="Download translation")
    btn.click(
        transcribe,
        [audio, lang, target],
        [out, translated, f_txt, f_srt, f_translation],
    )

if __name__ == "__main__":
    demo.queue().launch(share=os.environ.get("SHARE") == "1")