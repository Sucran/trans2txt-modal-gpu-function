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
  - `ASR_BACKEND` is only the fallback for older callers that do not pass
    `asr_backend`. Product routing should be done per request by the CPU
    function, based on the user's plan, with `asr_backend=whisper` or
    `asr_backend=qwen3_asr`.
  - `QWEN_ASR_MODEL_ID` / `QWEN_FORCED_ALIGNER_MODEL_ID` for the Qwen3-ASR
    transformers backend. The default ASR model is `Qwen/Qwen3-ASR-1.7B` for
    accuracy; `Qwen/Qwen3-ASR-0.6B` is the faster throughput-oriented option.
    ForcedAligner is mandatory on the Qwen path so SRT timestamps keep the same
    contract as Whisper segments. T4 deployments use FP16 automatically; newer
    GPUs use BF16 when supported.
  - `QWEN_ALLOWED_ASR_MODEL_IDS` optionally allowlists request-level
    `asr_model_id` overrides. Blank means only the configured
    `QWEN_ASR_MODEL_ID` is accepted.
  - `QWEN_ASR_CONTEXT` optionally provides Qwen3-ASR contextual hints for proper
    nouns, acronyms, and domain terms. Prefer request-level `qwen_context` when
    the product layer knows episode/title-specific vocabulary.
  - `QWEN_SUBTITLE_MAX_SECONDS` / `QWEN_SUBTITLE_MAX_CHARS` /
    `QWEN_SUBTITLE_GAP_SECONDS` tune Qwen forced-align timestamp aggregation
    into subtitle-sized segments (defaults `1.6` / `12` / `0.35`, tuned to a
    Whisper-like subtitle cadence rather than raw word/character timestamps).
    Qwen subtitles project punctuation from ASR `result.text` back onto
    ForcedAligner timestamps before splitting, so boundaries prefer `。！？；，`
    over hard character limits.
  - `PYANNOTE_SEGMENTATION_BATCH_SIZE` / `PYANNOTE_EMBEDDING_BATCH_SIZE` (default `128` / `128` in the service; community-1 config default is `32` / `32`)
  - `PYANNOTE_USE_MEMORY_INPUT` / `PYANNOTE_PROGRESS_HOOK` (default `true` / `true`)

## Hugging Face token (deploy-only, never user-facing)

End users do **not** create or manage any Hugging Face Modal secret. The GPU image bakes pyannote, Qwen3-ASR, and Qwen3-ForcedAligner weights at build time and runs fully offline (`HF_HUB_OFFLINE=1`); `secrets = []` on the runtime function, so nothing HF-related shows up in the user's Modal workspace `Secrets` list.

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

For an isolated Qwen3-ASR GPU direct-test deployment, keep the original app
untouched and use a distinct Modal app name:

```bash
export HF_TOKEN=hf_your_token_here
export MODAL_APP_NAME=transcribe-modal-gpu-qwen3-dev
export MODAL_GPU_TYPE=T4
# Keep the fallback as Whisper; CPU decides Whisper vs Qwen per request by plan.
export ASR_BACKEND=whisper
export QWEN_ASR_MODEL_ID=Qwen/Qwen3-ASR-1.7B
export QWEN_ALLOWED_ASR_MODEL_IDS=""
export QWEN_FORCED_ALIGNER_MODEL_ID=Qwen/Qwen3-ForcedAligner-0.6B
export QWEN_ASR_CONTEXT=""
export QWEN_ALIGNER_MAX_SEGMENT_SECONDS=60
export QWEN_SUBTITLE_MAX_SECONDS=1.6
export QWEN_SUBTITLE_MAX_CHARS=12
export QWEN_SUBTITLE_GAP_SECONDS=0.35
uv run modal deploy -m src.config.modal_gpu_config --name transcribe-modal-gpu-qwen3-dev
```

After deploy, the Modal dashboard lists `transcribe_and_diarization_audio_function` (no separate web URL). The **Secrets** tab in the user's workspace will not contain any HF entry — that is the intended state.

## Local test of the pure function (one-shot invocation)

```bash
uv run modal run src.config.modal_gpu_config::transcribe_and_diarization_audio_function \
  --request-data '{"request_type": "diarization", "audio_file_data": "<base64>", "audio_file_name": "test.mp3"}'
```

## Qwen3-ASR direct comparison smoke test

After deploying `transcribe-modal-gpu-qwen3-dev`, run the comparison script
against the original Whisper app and the isolated Qwen app:

```bash
uv run python scripts/qwen3_gpu_direct_test.py
```

The script downloads the configured Apple Podcast sample, cuts 90s and 300s
clips with ffmpeg, calls `transcribe_and_diarization_audio_function` on both
apps, and writes JSON/text/SRT artifacts under
`/tmp/qwen3-asr-smoke/apple-1000767368971/`.
