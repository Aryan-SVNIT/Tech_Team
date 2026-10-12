# Whisper Desk

A local speech-to-text web app built with Flask and OpenAI Whisper. Audio is transcribed by the Whisper model running on your computer; the app does not use the OpenAI API and needs no API key.

## Requirements

- Python and the existing `.venv` in this folder
- FFmpeg available on `PATH` (Whisper uses it to decode audio)
- A network connection the first time a model is started, unless its checkpoint is already cached

The current workspace already has Flask, `openai-whisper`, Torch, and FFmpeg available. If you need to install or refresh the Python dependencies, run this from `whisper-project`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

From the `whisper-project` folder, run:

```powershell
.\.venv\Scripts\python.exe app.py
```

Then open <http://127.0.0.1:5000> in your browser. Keep the terminal window open while using the app.

On its first run, Whisper downloads the selected checkpoint to its local cache (normally `%USERPROFILE%\.cache\whisper`). Later starts reuse that file. Transcription itself runs locally. Microphone recording requires browser permission and works on localhost in current browsers that support `MediaRecorder`.

## Model and device settings

The default model is `small`. Set `WHISPER_MODEL` before starting Flask to compare with `base` or `medium`:

```powershell
$env:WHISPER_MODEL = "base"
.\.venv\Scripts\python.exe app.py
```

`WHISPER_MODEL` accepts any official model name supported by the installed Whisper package. `WHISPER_DEVICE` can be `auto` (default), `cpu`, or `cuda`; `auto` uses CUDA when available and otherwise CPU. For example:

```powershell
$env:WHISPER_MODEL = "base"
$env:WHISPER_DEVICE = "cpu"
.\.venv\Scripts\python.exe app.py
```

### Compare `base` and `small`

Use the same short audio clip, the same language selection, and the same upload or recording path for both runs. Stop the server with `Ctrl+C` between runs because the model is loaded once at startup. Compare each transcript with what was actually spoken; model choice can affect results, but accuracy is not guaranteed.

If `small` is too slow on your computer, stop the server and start it with `base`:

```powershell
$env:WHISPER_MODEL = "base"
.\.venv\Scripts\python.exe app.py
```

Use `medium` the same way by setting `$env:WHISPER_MODEL = "medium"` before launch. Model weights are loaded once when the application starts.

To return to the default `small` model, set `$env:WHISPER_MODEL = "small"` before starting the server.

## Use

1. Upload an MP3, WAV, M4A, WebM, MP4, MPEG, OGG, FLAC, AAC, or MPGA file, or start a microphone recording.
2. Leave language on **Auto-detect**, or choose English, Hindi, or Gujarati.
3. Select **Transcribe audio** for an uploaded file, or **Stop & transcribe** when a recording is done.
4. Read the result in the page or save it as a UTF-8 `transcript.txt` file.

Uploads are limited to 100 MB by default. Set `MAX_UPLOAD_MB` before starting Flask to change that limit; the browser uses the same configured value:

```powershell
$env:MAX_UPLOAD_MB = "200"
.\.venv\Scripts\python.exe app.py
```

There is no audio-duration cutoff. Whisper processes the full decoded recording in its internal windows; FFmpeg decodes uploads to mono 16 kHz audio first. Long files take more memory and time on the local computer. Temporary audio files are stored with randomly generated names during processing and removed when the request finishes. Songs and singing can remain less accurate than clear speech, and transcription accuracy is not guaranteed.
