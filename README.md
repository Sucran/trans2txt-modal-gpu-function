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
- Environment variables (can be set via `config.env` in project root, see `config.env.example`):
  - `MODAL_APP_NAME` (default `transcribe-modal-gpu`)
  - `MODAL_GPU_TYPE` (default `T4`)
  - `MODAL_CPU`, `MODAL_MEMORY`, `MODAL_GPU_TIMEOUT` etc.
  - `PYANNOTE_SEGMENTATION_BATCH_SIZE` / `PYANNOTE_EMBEDDING_BATCH_SIZE` (default `128` / `128` in the service; community-1 config default is `32` / `32`)
  - `PYANNOTE_USE_MEMORY_INPUT` / `PYANNOTE_PROGRESS_HOOK` (default `true` / `true`)

## Hugging Face token (deploy-only, never user-facing)

End users do **not** create or manage any Hugging Face Modal secret. The GPU image bakes pyannote weights at build time and runs fully offline (`HF_HUB_OFFLINE=1`); `secrets = []` on the runtime function, so nothing HF-related shows up in the user's Modal workspace `Secrets` list.

There are two deploy paths and the token is supplied differently in each:

- **Production (via [`transcribe-modal-bridge`](../../transcribe-modal-bridge))** — the bridge server holds `HF_TOKEN` in its own environment. `POST /modal/deploy-gpu` injects it into the `modal deploy` subprocess and rejects the request with `503 no_hf_token` if it is not configured. This is the only path end users go through, and they never see the token.
- **Operator local deploy** — when running `modal deploy` yourself (without the bridge), `export HF_TOKEN=hf_...` in your shell first. The image build will fail with a clear message if the token is missing. The token is bound to a single deploy via `modal.Secret.from_dict` for `run_function`; it is **not** registered as a named workspace secret.

### Migrating from the old `huggingface-secret` pattern

Earlier versions of this project required users to run `modal secret create huggingface-secret HF_TOKEN=...` in their workspace. That requirement is removed. Operators upgrading existing deployments can optionally `modal secret delete huggingface-secret` after a successful redeploy — the GPU app no longer references it. End users were never expected to perform that delete and should not be asked to.

## Install & deploy

```bash
cd transcribe-modal-gpu
uv sync

# Sanity check (imports only; does not contact Modal):
uv run python -c "import src.config.modal_gpu_config as m; print('ok', m.app.name)"

# Deploy to Modal (operator path; bridge path injects HF_TOKEN automatically)
export HF_TOKEN=hf_your_token_here
uv run modal deploy -m src.config.modal_gpu_config
```

After deploy, the Modal dashboard lists `transcribe_and_diarization_audio_function` (no separate web URL). The **Secrets** tab in the user's workspace will not contain any HF entry — that is the intended state.

## Local test of the pure function (one-shot invocation)

```bash
uv run modal run src.config.modal_gpu_config::transcribe_and_diarization_audio_function \
  --request-data '{"request_type": "diarization", "audio_file_data": "<base64>", "audio_file_name": "test.mp3"}'
```
