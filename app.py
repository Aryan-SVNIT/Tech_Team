"""Local Whisper speech-to-text web application."""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch
import whisper
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
try:
    MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "100"))
except ValueError as exc:
    raise ValueError("MAX_UPLOAD_MB must be a positive whole number.") from exc
if MAX_UPLOAD_MB < 1:
    raise ValueError("MAX_UPLOAD_MB must be at least 1.")
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
# Leave room for multipart boundaries and form fields while keeping the audio
# file itself capped at exactly MAX_UPLOAD_BYTES below.
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 1024 * 1024
UPLOAD_TOO_LARGE_MESSAGE = (
    f"That upload is over the {MAX_UPLOAD_MB} MB limit. Choose a smaller audio file "
    "or increase MAX_UPLOAD_MB before starting the app."
)
MIN_AUDIO_SECONDS = 0.5
# Treat only near-digital-zero recordings as silent so quiet speech remains valid.
SILENCE_PEAK_THRESHOLD = 1e-5
ALLOWED_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".ogg",
    ".wav",
    ".webm",
}
LANGUAGES = {"auto": None, "en": "en", "hi": "hi", "gu": "gu"}


def _model_settings() -> tuple[str, str]:
    model_name = os.environ.get("WHISPER_MODEL", "small").strip()
    if model_name not in whisper.available_models():
        allowed = ", ".join(whisper.available_models())
        raise ValueError(f"WHISPER_MODEL must be one of: {allowed}")

    requested_device = os.environ.get("WHISPER_DEVICE", "auto").strip().lower()
    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested_device in {"cpu", "cuda"}:
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise ValueError("WHISPER_DEVICE=cuda was set, but CUDA is not available.")
        device = requested_device
    else:
        raise ValueError("WHISPER_DEVICE must be auto, cpu, or cuda.")

    return model_name, device


def _load_model() -> tuple[whisper.Whisper, str, str]:
    model_name, device = _model_settings()
    cuda_available = torch.cuda.is_available()
    device_detail = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(
        f"Loading the local Whisper '{model_name}' model; device={device} ({device_detail}), "
        f"cuda_available={cuda_available}, torch_cuda_build={torch.version.cuda}. "
        "The first run downloads its checkpoint if it is not cached yet.",
        flush=True,
    )
    try:
        # This module-level call loads the model once for this Flask process.
        model = whisper.load_model(model_name, device=device)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Whisper model '{model_name}'. Check the network on the "
            "first run, or confirm the model checkpoint is present in the Whisper cache."
        ) from exc
    return model, model_name, device


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES
app.config["JSON_SORT_KEYS"] = False

# Load once as the application module starts. The lock avoids overlapping work on
# the shared model when Flask handles multiple transcription requests at once.
MODEL, MODEL_NAME, DEVICE = _load_model()
MODEL_LOCK = threading.Lock()


@app.get("/")
def index():
    return render_template(
        "index.html",
        model_name=MODEL_NAME,
        max_upload_mb=MAX_UPLOAD_MB,
        max_upload_bytes=MAX_UPLOAD_BYTES,
    )


@app.errorhandler(RequestEntityTooLarge)
def file_too_large(_error):
    return jsonify(error=UPLOAD_TOO_LARGE_MESSAGE), 413


@app.post("/transcribe")
def transcribe():
    uploaded_file = request.files.get("file")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify(error="Choose an audio file or record audio before transcribing."), 400

    safe_original_name = secure_filename(uploaded_file.filename)
    extension = Path(safe_original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        extensions = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return jsonify(error=f"Unsupported audio format. Accepted formats: {extensions}."), 400

    # Do not silently fall back to detection if a frontend request loses the
    # selected language field; auto-detection must be an explicit UI choice.
    language_choice = (request.form.get("language") or "").strip().lower()
    if language_choice not in LANGUAGES:
        return jsonify(error="Choose automatic detection, English, Hindi, or Gujarati."), 400

    upload_dir = BASE_DIR / ".uploads"
    temporary_path: Path | None = None
    try:
        upload_dir.mkdir(parents=True, exist_ok=True)
        # NamedTemporaryFile creates an unpredictable name, and the accepted
        # extension lets FFmpeg identify the container after the file is closed.
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix="audio_", suffix=extension, dir=upload_dir, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            uploaded_file.save(temporary_file)

        uploaded_size = temporary_path.stat().st_size
        if uploaded_size > MAX_UPLOAD_BYTES:
            return jsonify(error=UPLOAD_TOO_LARGE_MESSAGE), 413
        if uploaded_size == 0:
            return jsonify(error="The selected audio file is empty."), 400

        selected_language = LANGUAGES[language_choice]
        try:
            # Use Whisper's own FFmpeg decoder once, then pass its unmodified
            # 16 kHz mono waveform into transcribe for duration checks and ASR.
            decoded_audio = whisper.audio.load_audio(str(temporary_path))
        except FileNotFoundError:
            app.logger.exception(
                "FFmpeg was not found (format=%s, selected_language=%s, duration_seconds=unknown)",
                extension.lstrip("."),
                language_choice,
            )
            return jsonify(
                error="FFmpeg is required to read audio. Install FFmpeg and make sure it is available on PATH."
            ), 503
        except RuntimeError as exc:
            if str(exc).startswith("Failed to load audio"):
                details = str(exc).replace(str(temporary_path), "<temporary-audio>")
                app.logger.warning(
                    "FFmpeg audio decode failed (format=%s, selected_language=%s, "
                    "duration_seconds=unknown, details=%s)",
                    extension.lstrip("."),
                    language_choice,
                    details[-600:],
                )
                return jsonify(
                    error="Whisper could not read this audio. Check that the file is valid and not damaged."
                ), 422
            app.logger.exception(
                "Audio decoding failed (format=%s, selected_language=%s, duration_seconds=unknown)",
                extension.lstrip("."),
                language_choice,
            )
            return jsonify(error="Audio decoding failed. Check that this is a valid, playable audio file."), 422
        except Exception:
            app.logger.exception(
                "Unexpected audio decode error (format=%s, selected_language=%s, duration_seconds=unknown)",
                extension.lstrip("."),
                language_choice,
            )
            return jsonify(error="Audio decoding failed. Check that this is a valid, playable audio file."), 422

        if decoded_audio.size == 0 or not np.isfinite(decoded_audio).all():
            app.logger.warning(
                "Decoded audio is empty or invalid (format=%s, selected_language=%s, duration_seconds=0)",
                extension.lstrip("."),
                language_choice,
            )
            return jsonify(error="The audio decoded as empty or invalid. Try a different recording or file."), 422

        audio_duration = decoded_audio.size / whisper.audio.SAMPLE_RATE
        peak_level = float(np.max(np.abs(decoded_audio)))
        mime_type = (uploaded_file.mimetype or "unknown").split(";", 1)[0].lower()
        app.logger.info(
            "Audio decoded (duration_seconds=%.3f, decoded_samples=%d, decoded_sample_rate_hz=%d, "
            "selected_language=%s, format=%s, "
            "mime_type=%s, peak_level=%.8f, model=%s, device=%s)",
            audio_duration,
            decoded_audio.size,
            whisper.audio.SAMPLE_RATE,
            language_choice,
            extension.lstrip("."),
            mime_type,
            peak_level,
            MODEL_NAME,
            DEVICE,
        )

        if audio_duration < MIN_AUDIO_SECONDS:
            app.logger.warning(
                "Audio too short (duration_seconds=%.3f, selected_language=%s)",
                audio_duration,
                language_choice,
            )
            return jsonify(
                error=f"This audio is only {audio_duration:.2f} seconds long. Record or choose at least 0.5 seconds of audio."
            ), 422
        if peak_level <= SILENCE_PEAK_THRESHOLD:
            app.logger.warning(
                "Silent audio (duration_seconds=%.3f, selected_language=%s, peak_level=%.8f)",
                audio_duration,
                language_choice,
                peak_level,
            )
            return jsonify(
                error="No sound was detected. Check the microphone input or choose an audio file containing speech."
            ), 422

        transcription_started = time.perf_counter()
        try:
            with MODEL_LOCK:
                # Explicit transcription preserves the audio's spoken language.
                result = MODEL.transcribe(
                    decoded_audio,
                    task="transcribe",
                    language=selected_language,
                    fp16=(DEVICE == "cuda"),
                )
        except Exception:
            processing_seconds = time.perf_counter() - transcription_started
            app.logger.exception(
                "Whisper transcription failed (duration_seconds=%.3f, selected_language=%s, "
                "model=%s, device=%s, processing_seconds=%.3f, format=%s)",
                audio_duration,
                language_choice,
                MODEL_NAME,
                DEVICE,
                processing_seconds,
                extension.lstrip("."),
            )
            return jsonify(error="Transcription failed while running the local Whisper model."), 500
        processing_seconds = time.perf_counter() - transcription_started

        transcript = (result.get("text") or "").strip()
        detected_language = result.get("language")
        segments = result.get("segments") or []
        app.logger.info(
            "Transcription finished (duration_seconds=%.3f, selected_language=%s, "
            "detected_language=%s, model=%s, device=%s, processing_seconds=%.3f, "
            "realtime_factor=%.3f, transcript_chars=%d)",
            audio_duration,
            language_choice,
            detected_language or "unknown",
            MODEL_NAME,
            DEVICE,
            processing_seconds,
            processing_seconds / audio_duration,
            len(transcript),
        )
        no_speech_segments = [
            segment for segment in segments
            if float(segment.get("no_speech_prob", 0.0)) >= 0.9
        ]
        if not transcript or (segments and len(no_speech_segments) == len(segments)):
            app.logger.warning(
                "No speech detected (duration_seconds=%.3f, selected_language=%s, detected_language=%s)",
                audio_duration,
                language_choice,
                detected_language or "unknown",
            )
            return jsonify(
                error="Whisper did not detect clear speech. Check the recording volume and try again."
            ), 422
        return jsonify(
            transcript=transcript,
            selected_language=language_choice,
            detected_language=detected_language,
            message="Transcription complete.",
        )
    except OSError:
        app.logger.exception("Could not save or read the uploaded audio")
        return jsonify(error="The audio could not be saved for processing. Check available disk space and try again."), 500
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                app.logger.warning("Could not remove temporary audio file %s", temporary_path)


if __name__ == "__main__":
    # Keep the development reloader off so it cannot start a second model copy.
    port = int(os.environ.get("PORT", "5000"))
    logging.basicConfig(level=logging.INFO)
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
