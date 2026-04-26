# transcribe-modal-gpu

GPU-side Modal app. Deploys a Modal App named `transcribe-modal-gpu` with a single exposed function:

- `transcribe_and_diarization_audio_function` — pure Python Modal function invoked from other Modal projects (or `modal run`) via `modal.Function.from_name(...).call()` / `.spawn(...)`. There is no HTTP entry point; long-running work is expected to use Modal’s function timeouts, not a request gateway.

The handler dispatches internally to either Whisper transcription (for single chunks) or pyannote speaker diarization based on the request payload shape / `request_type` field. See `src/config/modal_gpu_config.py` for the dispatcher.

## Layout

```
src/
  config/
    modal_shared.py          # Modal App, image, volume, model preloading
    modal_gpu_config.py      # GPU Modal function definition
  services/
    whisper_service.py       # Whisper runtime
    speaker_diarization_service.py  # pyannote runtime
    transcription_endpoint_service.py  # chunk handler (WhisperService cache etc.)
  utils/
    file_utils.py
```

No orchestration code lives here. The client-side splitting/merging/concurrency lives in the CPU project.

## Prerequisites

- A Modal account and `modal token new` already run.
- A Modal Secret named `huggingface-secret` exposing `HF_TOKEN` (or `HUGGING_FACE_TOKEN`). Without this the image still builds, but speaker diarization is disabled at runtime.
- Environment variables (can be set via `config.env` in project root, see `config.env.example`):
  - `MODAL_APP_NAME` (default `transcribe-modal-gpu`)
  - `MODAL_GPU_TYPE` (default `T4`)
  - `MODAL_CPU`, `MODAL_MEMORY`, `MODAL_GPU_TIMEOUT` etc.

## Install & deploy

```bash
cd transcribe-modal-gpu
uv sync

# Sanity check (imports only; does not contact Modal):
uv run python -c "import src.config.modal_gpu_config as m; print('ok', m.app.name)"

# Deploy to Modal
uv run modal deploy -m src.config.modal_gpu_config
```

After deploy, the Modal dashboard lists `transcribe_and_diarization_audio_function` (no separate web URL).

## Local test of the pure function (one-shot invocation)

```bash
uv run modal run src.config.modal_gpu_config::transcribe_and_diarization_audio_function \
  --request-data '{"request_type": "diarization", "audio_file_data": "<base64>", "audio_file_name": "test.mp3"}'
```
