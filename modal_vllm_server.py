"""Standalone Modal deployment: OpenAI-compatible vLLM server for CiteClaw.

This file is INTENTIONALLY ISOLATED from the main CiteClaw package. It has a
single dependency — ``modal`` — which CiteClaw itself does not require. The main
project remains runnable with only ``pip install -e ".[dev]"`` and never
imports anything from this file.

============================================================================
Quickstart
============================================================================

1. One-time setup (on your local machine):

       pip install modal
       modal setup                 # authenticate with Modal

2. (Optional) set the model / GPU via environment variables. Defaults are
   shown. All values are read at deploy time and baked into the app:

       export CITECLAW_VLLM_MODEL="Qwen/Qwen3.5-122B-A10B-FP8"
       export CITECLAW_VLLM_GPU="H200"           # or B200, H100, A100-80GB, ...
       export CITECLAW_VLLM_GPU_COUNT=1           # tensor parallel degree
       export CITECLAW_VLLM_API_KEY="citeclaw-local-key"
       export CITECLAW_VLLM_MAX_MODEL_LEN=16384
       export CITECLAW_VLLM_SCALEDOWN=300         # idle seconds before shutdown
       export CITECLAW_VLLM_REASONING_PARSER=""  # gemma4 / qwen3 / deepseek_r1 / ...
       export CITECLAW_VLLM_EXTRA_ARGS=""        # extra `vllm serve` flags
       export CITECLAW_VLLM_APP_NAME="citeclaw-vllm-test"  # Modal app name

   To deploy MULTIPLE distinct vLLM endpoints (e.g. one Qwen + one Gemma),
   set CITECLAW_VLLM_APP_NAME to a unique value per deploy. Each app gets
   its own URL; the CiteClaw config's `models:` registry can route YAML
   aliases to each one independently. Example::

       # Qwen3.5-122B endpoint
       CITECLAW_VLLM_APP_NAME=citeclaw-vllm-qwen \
       CITECLAW_VLLM_MODEL=Qwen/Qwen3.5-122B-A10B-FP8 \
       CITECLAW_VLLM_REASONING_PARSER=qwen3 \
       modal deploy modal_vllm_server.py

       # Gemma 4 31B endpoint
       CITECLAW_VLLM_APP_NAME=citeclaw-vllm-gemma \
       CITECLAW_VLLM_MODEL=google/gemma-4-31B-it \
       CITECLAW_VLLM_REASONING_PARSER=gemma4 \
       CITECLAW_VLLM_EXTRA_ARGS="--tool-call-parser gemma4 --gpu-memory-utilization 0.92" \
       modal deploy modal_vllm_server.py

3. Deploy (one-shot, long-lived):

       modal deploy modal_vllm_server.py

   Or develop iteratively with auto-reload:

       modal serve modal_vllm_server.py

   Modal will print a public URL like
   ``https://<you>--citeclaw-vllm-serve.modal.run``.

4. Point CiteClaw at the endpoint (config.yaml):

       screening_model:  "Qwen/Qwen3.5-122B-A10B-FP8"
       llm_base_url:     "https://<you>--citeclaw-vllm-serve.modal.run/v1"
       llm_api_key:      "citeclaw-local-key"

   Then run CiteClaw normally:

       citeclaw -c config.yaml

5. When finished:

       modal app stop citeclaw-vllm

   Or simply let the container auto-scale down after the configured idle
   window (``CITECLAW_VLLM_SCALEDOWN`` seconds).

============================================================================
Cold start & caching
============================================================================

- The first cold start on a new model will download the weights from the
  HuggingFace Hub into a persistent Modal Volume (mounted at the HF cache
  directory). For a 122B model this takes ~5–15 min depending on HF bandwidth.
- Subsequent cold starts reuse the cached weights and only pay the vLLM
  initialization cost (~1–3 min for a 122B model).
- ``hf_transfer`` is enabled for faster downloads.

============================================================================
Security
============================================================================

The endpoint is publicly reachable by default but requires a bearer token
(``CITECLAW_VLLM_API_KEY``). Treat this key like any other secret. For private
deployments, use Modal's auth-token web endpoint feature or an IP allowlist.
"""

from __future__ import annotations

import os

import modal

# ---------------------------------------------------------------------------
# Configuration (read from environment at deploy time)
# ---------------------------------------------------------------------------

MODEL_NAME: str = os.environ.get("CITECLAW_VLLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
GPU_TYPE: str = os.environ.get("CITECLAW_VLLM_GPU", "H200")
GPU_COUNT: int = int(os.environ.get("CITECLAW_VLLM_GPU_COUNT", "1"))
API_KEY: str = os.environ.get("CITECLAW_VLLM_API_KEY", "citeclaw-local-key")
MAX_MODEL_LEN: int = int(os.environ.get("CITECLAW_VLLM_MAX_MODEL_LEN", "16384"))
SCALEDOWN: int = int(os.environ.get("CITECLAW_VLLM_SCALEDOWN", "300"))
# Optional extra `vllm serve` flags as a single shell-quoted string. Used for
# things vLLM exposes per-model — e.g. `--tool-call-parser gemma4`,
# `--kv-cache-dtype fp8`, `--gpu-memory-utilization 0.92`, etc. Empty by
# default; the deploy script can pin model-specific knobs without us having
# to add a typed env var for every flag vLLM ships.
EXTRA_VLLM_ARGS: str = os.environ.get("CITECLAW_VLLM_EXTRA_ARGS", "")

# Name of a Modal secret to attach to the function. Required for gated
# models on HuggingFace (Gemma, Llama, etc.) where the secret should
# expose ``HF_TOKEN``. Empty (the default) attaches no secret, which
# preserves prior behavior for public models like Qwen3.5.
HF_SECRET_NAME: str = os.environ.get("CITECLAW_VLLM_HF_SECRET", "")

# Optional pre-built Docker image reference. When set, skip the
# ``nvidia/cuda + pip install vllm`` build path and pull this image
# directly. Empty (default) uses the build-from-source path. Note: most
# prebuilt vLLM images set ENTRYPOINT=["vllm"], which conflicts with
# Modal's container entrypoint mechanism, so this option is reserved
# for special cases that explicitly handle the entrypoint reset.
IMAGE_REF: str = os.environ.get("CITECLAW_VLLM_IMAGE_REF", "")

# Force-install a specific transformers version AFTER vllm. Used when a
# new model architecture (e.g. Gemma 4 31B's ``model_type=gemma4``) needs
# a transformers release more recent than the one vLLM's wheel pins.
# Pip's --force-reinstall override is required because vLLM 0.19.0
# pins ``transformers<5`` in its wheel metadata even though the runtime
# is compatible with transformers 5.x. Empty (default) uses whatever
# vLLM's wheel resolves on its own.
#
# Examples:
#   "transformers==5.5.0"  — pin a specific release
#   "git+https://github.com/huggingface/transformers.git"  — main branch
FORCE_TRANSFORMERS: str = os.environ.get("CITECLAW_VLLM_FORCE_TRANSFORMERS", "")
# vLLM version.
#
# 0.19.0 is the first stable release with Qwen3.5-122B-A10B support (added in
# the 0.17.x line alongside qwen3_5_moe architecture and the GDN attention
# backend). vLLM 0.11.0 — the previous pin — does NOT recognize Qwen3.5 and
# will refuse to load the model with a KeyError on the architecture.
VLLM_VERSION: str = os.environ.get("CITECLAW_VLLM_VERSION", "0.19.0")

# Reasoning parser — let vLLM extract <think>...</think> blocks and expose
# them as `message.reasoning_content` + `usage.completion_tokens_details.reasoning_tokens`.
# Common values: "qwen3", "deepseek_r1", "granite", "gpt_oss". Set to empty
# string to disable parsing (raw thinking tags leak into the response — CiteClaw
# strips them client-side as a safety net).
#
# NOTE: only set this for actual reasoning models (Qwen3 thinking variants,
# DeepSeek-R1, etc). Qwen2.5-Instruct is NOT a reasoning model — it doesn't
# emit <think>...</think> blocks, and forcing the qwen3 parser on it can
# cause vLLM to error at startup. Default is empty (no parser).
REASONING_PARSER: str = os.environ.get("CITECLAW_VLLM_REASONING_PARSER", "")

# Modal app name. Configurable via env var so the same file can deploy
# multiple distinct vLLM endpoints in the same Modal account — e.g.
# ``citeclaw-vllm-qwen`` for Qwen3.5-122B and ``citeclaw-vllm-gemma`` for
# Gemma 4 31B — and the CiteClaw config can route different YAML aliases
# to each. The default ``citeclaw-vllm-test`` preserves prior behavior
# for users who only need one endpoint.
APP_NAME: str = os.environ.get("CITECLAW_VLLM_APP_NAME", "citeclaw-vllm-test")
PORT = 8000

# ---------------------------------------------------------------------------
# Image + volume
# ---------------------------------------------------------------------------

app = modal.App(APP_NAME)

# Persistent HF cache so we don't re-download 100GB+ weights on every cold start.
hf_cache_vol = modal.Volume.from_name("citeclaw-hf-cache", create_if_missing=True)

# Build the base image. Two paths:
#
# 1. ``IMAGE_REF`` set → pull a pre-built image directly. The vLLM team
#    publishes ``vllm/vllm-openai:<tag>`` images that already contain a
#    working vllm + transformers + torch combo for specific models. Use
#    this for Gemma 4 (``vllm/vllm-openai:gemma4``) and any other model
#    whose dependency story is too messy to reproduce via pip.
#
# 2. ``IMAGE_REF`` empty (default) → build from nvidia/cuda + pip install
#    a pinned vLLM version. Used by Qwen3.5-122B and the legacy path.
if IMAGE_REF:
    # Reserved for special cases — most prebuilt vllm images set
    # ENTRYPOINT=["vllm"] which conflicts with Modal's container
    # entrypoint mechanism. Use FORCE_TRANSFORMERS instead for the
    # common Gemma 4 case.
    _base_image = modal.Image.from_registry(
        IMAGE_REF,
        setup_dockerfile_commands=[
            "RUN ln -sf $(command -v python3 || command -v python3.12) "
            "/usr/local/bin/python || true",
        ],
    )
else:
    _base_image = (
        # CUDA devel image gives us `nvcc` under /usr/local/cuda, which FlashInfer
        # needs to JIT-compile its top-k sampling kernel at first use. The plain
        # `debian_slim` image only ships CUDA runtime libraries (via pip wheels),
        # not the toolkit, so flashinfer bails with
        #   RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
        # Matching CUDA 12.8 aligns with the torch 2.8 wheel vLLM 0.11.0 depends on.
        modal.Image.from_registry(
            "nvidia/cuda:12.8.0-devel-ubuntu22.04",
            add_python="3.12",
        )
        .apt_install("git")
        .pip_install(
            f"vllm=={VLLM_VERSION}",
            # NOTE: vLLM 0.19 pins its own compatible transformers range in its
            # wheel requirements — we no longer pin it here. (The old 4.56.x pin
            # was a workaround for a vLLM 0.11 × transformers 4.57 tokenizer bug.)
            "huggingface_hub[hf_transfer]",
            "flashinfer-python",
        )
    )
    if FORCE_TRANSFORMERS:
        # Force-upgrade transformers (and its transitive deps) AFTER vLLM.
        # vLLM 0.19.0's wheel pins ``transformers<5`` but the runtime is
        # compatible with 5.x — the pin only matters at install-time.
        #
        # We do NOT use ``--no-deps``: a newer transformers needs a newer
        # huggingface_hub (one with ``is_offline_mode`` etc), so pip needs
        # to pull in the deps. Pip will warn about the vLLM metadata pin
        # conflict but proceeds; the runtime works.
        _base_image = _base_image.run_commands(
            f"pip install --upgrade --force-reinstall {FORCE_TRANSFORMERS}",
        )

image = (
    _base_image
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # NOTE: VLLM_USE_V1 was required by vLLM 0.11 (which asserted
            # it at startup). vLLM ≥0.13 removed the flag entirely and
            # logs a warning if it's set ("Unknown vLLM environment
            # variable detected: VLLM_USE_V1"), so we no longer pass it.
            # Avoid NCCL hangs on some multi-GPU configs
            "NCCL_CUMEM_ENABLE": "0",
            # Help flashinfer's JIT find nvcc without having to probe.
            "CUDA_HOME": "/usr/local/cuda",
            #
            # --- Config propagation ---
            # The module-level `os.environ.get("CITECLAW_VLLM_...", ...)`
            # calls at the top of this file run in TWO places:
            #   1. Locally, when `modal deploy` executes the file (sees
            #      whatever you `export`-ed in your shell).
            #   2. Remotely, when the container starts and re-imports
            #      the module (no shell env vars → falls back to
            #      defaults).
            # Without propagation, step 2 silently ignored your local
            # overrides (e.g. you set CITECLAW_VLLM_MODEL=Qwen3.5-122B but
            # the container kept launching Qwen2.5-0.5B). By baking the
            # resolved values into the image's environment below, the
            # remote re-import sees the same values you set locally.
            "CITECLAW_VLLM_MODEL": MODEL_NAME,
            "CITECLAW_VLLM_GPU": GPU_TYPE,
            "CITECLAW_VLLM_GPU_COUNT": str(GPU_COUNT),
            "CITECLAW_VLLM_API_KEY": API_KEY,
            "CITECLAW_VLLM_MAX_MODEL_LEN": str(MAX_MODEL_LEN),
            "CITECLAW_VLLM_SCALEDOWN": str(SCALEDOWN),
            "CITECLAW_VLLM_VERSION": VLLM_VERSION,
            "CITECLAW_VLLM_REASONING_PARSER": REASONING_PARSER,
            "CITECLAW_VLLM_APP_NAME": APP_NAME,
            "CITECLAW_VLLM_EXTRA_ARGS": EXTRA_VLLM_ARGS,
            # CRITICAL: must be propagated so the remote module re-import
            # sees the same value as the local deploy. Otherwise the
            # conditional ``_FUNCTION_SECRETS`` evaluates to ``[]`` in the
            # container while the deploy submitted 3 object ids (image,
            # volume, secret), and Modal raises
            # ``Function has 2 dependencies but container got 3 object ids``.
            "CITECLAW_VLLM_HF_SECRET": HF_SECRET_NAME,
            "CITECLAW_VLLM_IMAGE_REF": IMAGE_REF,
            "CITECLAW_VLLM_FORCE_TRANSFORMERS": FORCE_TRANSFORMERS,
        }
    )
)


# ---------------------------------------------------------------------------
# Web-server function — launches vLLM's OpenAI-compatible API server and
# exposes it on a public Modal URL.
# ---------------------------------------------------------------------------

_GPU_SPEC = f"{GPU_TYPE}:{GPU_COUNT}" if GPU_COUNT > 1 else GPU_TYPE

# Optional HuggingFace secret. Modal looks the secret up at deploy time;
# if HF_SECRET_NAME is empty we attach nothing (so the default Qwen path
# keeps working without the user needing any secret). For gated models
# like Gemma 4 the deploy script sets CITECLAW_VLLM_HF_SECRET=huggingface
# and the secret must contain HF_TOKEN.
_FUNCTION_SECRETS: list[modal.Secret] = (
    [modal.Secret.from_name(HF_SECRET_NAME)] if HF_SECRET_NAME else []
)


@app.function(
    image=image,
    gpu=_GPU_SPEC,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    secrets=_FUNCTION_SECRETS,
    # Keep the container warm briefly to absorb bursty CiteClaw batches:
    scaledown_window=SCALEDOWN,
    # vLLM can serve many requests concurrently on a single replica:
    max_containers=1,
    timeout=60 * 60,  # 1 hour safety cap per container lifetime
)
@modal.concurrent(max_inputs=256)
# startup_timeout must cover:
#   1. HuggingFace download of weights (~12 min for Qwen3.5-122B ≈ 127GB, only
#      on the FIRST cold start before the volume is populated)
#   2. Loading sharded weights into GPU memory (~30s for TP=2 H200)
#   3. KV cache profile + FlashInfer kernel warmup (1-3 min on large MoE)
# 15 min was too tight — the first cold start timed out at 14:51 (port 8000
# still wasn't up, Modal killed the container mid-warmup). 45 min gives
# comfortable headroom for first-download cold starts and is harmless for
# subsequent warm-cache cold starts (Modal only waits until the port is up).
@modal.web_server(port=PORT, startup_timeout=60 * 45)
def serve() -> None:
    """Launch ``vllm serve`` as a background process on port 8000.

    The ``@modal.web_server`` decorator tunnels public HTTPS traffic to this
    port. Requests arrive as standard OpenAI chat completions calls — the
    CiteClaw client just needs ``base_url=<the-modal-url>/v1``.
    """
    import subprocess

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--api-key",
        API_KEY,
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--trust-remote-code",
        # NOTE: vLLM 0.11 had a FlashInfer-on-Blackwell bug in
        # `v1/attention/backends/flashinfer.py` during "decode, FULL" CUDA
        # graph capture (`assert decode_wrapper._sm_scale == self.scale`),
        # which forced us to pass `--enforce-eager` as a workaround.
        # vLLM 0.19 has since fixed this, so we let CUDA graphs + torch.compile
        # run — this is worth roughly a 2× decode throughput improvement on
        # big MoE models like Qwen3.5-122B-A10B. If you see FlashInfer
        # assertion errors during graph capture again, re-enable eager mode
        # by passing `--enforce-eager` here.
        #
        # Enable prefix caching: the graph-annotation workload sends the
        # same system prompt and instruction for every paper, so the
        # first ~200 tokens of every request are identical. vLLM caches
        # those KV blocks across requests, cutting prefill cost ~80%.
        # (vLLM normally disables prefix caching for Qwen3.5 MoE by
        # default — we override that here since the upstream concerns
        # don't apply to our per-paper inference pattern.)
        "--enable-prefix-caching",
    ]
    if GPU_COUNT > 1:
        cmd += ["--tensor-parallel-size", str(GPU_COUNT)]
    if REASONING_PARSER:
        # vLLM ≥0.10 enables reasoning parsing implicitly when a parser name
        # is supplied. (Older vLLM required the now-removed --enable-reasoning
        # flag; passing it here causes "unrecognized arguments" on 0.10+.)
        # The parser makes vLLM expose thinking content as a separate field
        # so CiteClaw can count reasoning tokens in its budget tracker.
        cmd += ["--reasoning-parser", REASONING_PARSER]
    if EXTRA_VLLM_ARGS:
        # Allow passing arbitrary `vllm serve` flags from the deploy
        # environment without having to add a typed env var for every
        # one. Use ``shlex.split`` so quoted multi-word values survive.
        import shlex
        cmd += shlex.split(EXTRA_VLLM_ARGS)

    print(f"[citeclaw-vllm] Launching vLLM: app={APP_NAME} model={MODEL_NAME} gpu={_GPU_SPEC} max_len={MAX_MODEL_LEN}")
    print(f"[citeclaw-vllm] Command: {' '.join(cmd)}")

    # Sanity check: can we even import vllm? (fail fast with a clear error
    # instead of waiting 15 min for Modal's startup_timeout to fire.)
    try:
        import vllm  # noqa: F401

        print(f"[citeclaw-vllm] vllm import OK (version={vllm.__version__})")
    except Exception as exc:
        print(f"[citeclaw-vllm] FATAL: `import vllm` failed: {exc!r}")
        print("[citeclaw-vllm] → likely vLLM/transformers version mismatch. "
              "Set CITECLAW_VLLM_VERSION to a newer release and redeploy.")
        raise

    # Launch in background — web_server waits for the port to be open.
    subprocess.Popen(cmd)


# ---------------------------------------------------------------------------
# Local smoke-test entrypoint: `modal run modal_vllm_server.py`
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main() -> None:
    """Print the deployed endpoint URL and a minimal usage example."""
    url = serve.get_web_url() if hasattr(serve, "get_web_url") else "(run `modal deploy` first)"
    print("=" * 72)
    print(f"CiteClaw vLLM server — Modal app: {APP_NAME}")
    print("=" * 72)
    print(f"  Model:         {MODEL_NAME}")
    print(f"  GPU:           {_GPU_SPEC}")
    print(f"  API key:       {API_KEY}")
    print(f"  Max context:   {MAX_MODEL_LEN}")
    print(f"  Scaledown:     {SCALEDOWN}s idle")
    print()
    print(f"  Endpoint URL:  {url}")
    print()
    print("Add to your CiteClaw config.yaml:")
    print()
    print(f'  screening_model: "{MODEL_NAME}"')
    print(f'  llm_base_url:    "{url}/v1"')
    print(f'  llm_api_key:     "{API_KEY}"')
    print()
