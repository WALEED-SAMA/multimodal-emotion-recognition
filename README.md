# Multimodal Emotion Recognition

![demo](assets/demo.gif)

Real-time emotion recognition that combines facial expression and speech emotion into a single label per person, running on a laptop webcam + microphone.

Facial expression alone misses a lot. People fake smiles, look away from the camera, or just stay neutral while feeling something else. Speech alone has the same problem (silence is uninformative, sarcasm reads as cheerful, monotone delivery washes everything out). Running both at the same time and combining the outputs gives a more useful prediction than either modality on its own.

## What it does

- Opens the default webcam and microphone.
- Detects faces every frame and runs an 8-class facial emotion classifier (Anger / Contempt / Disgust / Fear / Happiness / Neutral / Sadness / Surprise).
- Runs a speech emotion classifier on a sliding 2-second window of microphone audio in a background thread.
- Combines the two probability distributions with a configurable weight and displays one emotion label per face.
- Optional: per-frame CSV timeline, annotated video output, verbose overlay with the per-modality breakdown.

Runs at ~15+ FPS on an Apple M3 CPU, no GPU required.

## Architecture

```
┌─────────────────┐                  ┌──────────────────┐
│  Webcam (cv2)   │                  │  Mic (sounddevice)│
│    30 FPS       │                  │  16 kHz, mono     │
└────────┬────────┘                  └─────────┬─────────┘
         │ BGR frame                            │ float32 samples
         ▼                                      ▼
┌─────────────────┐                  ┌──────────────────┐
│ MediaPipe       │                  │ Ring buffer       │
│ Face Detector   │                  │ (last 2 sec)      │
└────────┬────────┘                  └─────────┬─────────┘
         │ face crops                          │ 2-sec window every ~1s
         ▼                                      ▼
┌─────────────────┐                  ┌──────────────────┐
│ Facial emotion  │                  │ HuBERT speech     │
│ classifier      │                  │ emotion model     │
│ ONNX, ~30 ms    │                  │ (~80 ms warm)     │
└────────┬────────┘                  └─────────┬─────────┘
         │  P_visual[8]                         │ P_audio[k]
         │  (every frame)                       │ (every ~1s, async)
         │                                      │
         └─────────────┬──────────────────────┬─┘
                       ▼                      ▼
                ┌──────────────────────────────┐
                │ Class mapping +               │
                │ Late fusion                   │
                │ α·P_v + (1-α)·P_a             │
                │ (audio decays if stale > 3s)  │
                └──────────────┬───────────────┘
                               ▼
                ┌──────────────────────────────┐
                │ Single label overlay          │
                │ "Happiness 71%"               │
                └──────────────────────────────┘
```

### Threading model

The speech model is about 5× slower than the facial classifier per inference, so it runs in a dedicated daemon thread. The video stream stays at full FPS regardless of how long the next audio inference takes. The audio thread updates a lock-guarded shared dict with the latest probabilities, and the main thread reads from it every frame.

### Stale-audio decay

If the latest audio prediction is older than `--stale-s` seconds (default 3), the fusion weight α is set to 1.0 and the label falls back to vision only. This handles the case where the person stops talking. Without it, silence would drag the fused label toward whatever the audio model last predicted before the speaker went quiet.

## Installation

```bash
git clone https://github.com/WALEED-SAMA/multimodal-emotion-recognition
cd multimodal-emotion-recognition

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Python version note:** TensorFlow, `mediapipe`, and `onnxruntime` ship wheels for CPython 3.10–3.12 on macOS arm64. Newer Python (3.13+) falls back to source builds and won't work. Pin to 3.12.

**macOS permissions:** grant camera **and** microphone access to your terminal app in System Settings → Privacy & Security. The first run will trigger the prompts. After granting, quit the terminal with ⌘Q and relaunch (macOS only re-reads permissions at process start).

## Quick start

```bash
# Live webcam + mic with fused label per face
python src/main.py

# Show per-modality breakdown (visual / audio / α) below each face
python src/main.py --verbose

# Vision only (skip the audio thread)
python src/main.py --no-audio

# Headless run with a 10-second timeline written to CSV
python src/main.py --no-display --duration 10 --csv timeline.csv
```

Press `q` in the window to quit.

## CLI options

| Flag | Default | Notes |
|---|---|---|
| `--alpha` | `0.6` | Visual weight in the fusion. `1.0` = visual only, `0.0` = audio only. |
| `--audio-window` | `2.0` | Seconds of audio fed to the speech model per inference. |
| `--audio-hop` | `1.0` | How often the audio thread re-runs the speech model. |
| `--stale-s` | `3.0` | Drop the audio contribution after this many seconds of no fresh prediction. |
| `--ser-model` | `superb/hubert-base-superb-er` | Any HuggingFace `AutoModelForAudioClassification` checkpoint. |
| `--label-order` | unset | Comma-separated class labels for SER models with anonymous `LABEL_n` outputs. |
| `--visual-model` | `enet_b0_8_best_vgaf` | Facial emotion model. Use `enet_b2_8` for slightly higher accuracy at ~2× the latency. |
| `--engine` | `onnx` | `onnx` (faster on CPU) or `torch`. |
| `--device` | `cpu` | `cuda` if available; leave as `cpu` on Apple Silicon. |
| `--min-face` | `40` | Minimum face size in pixels. |
| `--no-audio` | off | Disable the speech thread entirely. |
| `--no-display` | off | Headless mode (useful with `--csv`). |
| `--duration` | `0` | Auto-stop after N seconds. `0` runs until `q`. |
| `--verbose` | off | Show per-modality breakdown below each face. |

## How the fusion works

For each frame:

1. **Visual.** MediaPipe detects faces, the face crops are batched into the EmotiEffLib ONNX recognizer, producing per-face 8-class probabilities `P_v` over `[Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise]`.
2. **Audio.** The speech thread holds the latest 2-second mic window and runs HuBERT-base-SUPERB-ER on it about once per second. The output (`neu / hap / ang / sad`) is remapped to the EmotiEffLib class order. Non-overlapping classes contribute 0 and the vector is renormalized.
3. **Fusion.** `P_fused = α · P_v + (1 - α) · P_a`. If audio is older than `--stale-s`, α is forced to 1.0.
4. **Display.** `argmax(P_fused)` is the label drawn on the face. The confidence is the corresponding probability.

For multi-face scenes the same `P_a` is applied to every face. Audio is a room-level signal; speaker diarization is out of scope.

## Choice of speech emotion model

Most published HuggingFace speech emotion checkpoints fail one of three sanity checks: missing classifier weights, anonymous labels with no documented order, or weights stored in a precision the model code doesn't expect. The default `superb/hubert-base-superb-er` is the one checkpoint we tested that loads cleanly with verified labels and produces non-uniform predictions on diverse inputs.

The tradeoff is a 4-class vocabulary (`neu / hap / ang / sad`). Audio therefore contributes nothing to `Contempt / Disgust / Fear / Surprise`, which stay vision-only. Voice tone isn't a strong signal for those four classes anyway.

Larger 8-class models can be swapped in via `--ser-model`. If the checkpoint exposes anonymous `LABEL_n` classes, supply the ordering with `--label-order`.

## Performance

Measured on Apple M3, CPU only, default settings, 1080p webcam:

| Metric | Value |
|---|---|
| End-to-end FPS (visual stream) | 15–20 |
| Visual inference latency (ONNX, enet_b0) | ~30 ms warm |
| Audio inference latency (HuBERT-base) | ~80 ms warm |
| Audio update cadence | ~1 Hz |
| Visual model size on disk | 16 MB |
| Audio model size on disk | ~95 MB |
| First-run download (audio model) | ~95 MB |

The audio thread runs concurrently with the video thread. The main thread reads the latest audio probabilities whenever they're available.

## Failure modes

- **`OpenCV: not authorized to capture video`**: camera permission denied for the terminal app. Grant it under System Settings → Privacy & Security → Camera, then ⌘Q and relaunch.
- **`PortAudioError` or "no input device"**: microphone permission denied. Same fix, under Microphone.
- **Window takes a long time to open on first run**: the speech model is downloading from HuggingFace (~95 MB). Subsequent runs use the cached copy.
- **FPS drops with `--audio-window 4.0` or other long windows**: speech inference becomes expensive enough to compete for the CPU. Lower the window or increase `--audio-hop`.
- **Audio label is always `neutral`**: either the mic is muted, the speaker is whispering below the 0.005 amplitude threshold, or the room is quiet. Speak normally and watch the audio-status line in the corner.

## Related work

This project builds on the following:

- **EmotiEffLib** (Andrey V. Savchenko et al.): the facial emotion classifier. The default visual model is `enet_b0_8_best_vgaf`, trained on AffectNet. Used through `pip install emotiefflib`. Source: [github.com/sb-ai-lab/EmotiEffLib](https://github.com/sb-ai-lab/EmotiEffLib).
  - Savchenko, A. V. *Facial Expression Recognition with Adaptive Frame Rate based on Multiple Testing Correction.* ICML 2023.
  - Savchenko, A. V., Savchenko, L. V., Makarov, I. *Classifying emotions and engagement in online learning based on a single facial expression recognition neural network.* IEEE Transactions on Affective Computing, 2022.
- **HuBERT-base SUPERB-ER**: the speech emotion classifier, trained on IEMOCAP and benchmarked under the SUPERB protocol.
  - Hsu, W.-N., et al. *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units.* IEEE/ACM TASLP 2021.
  - Yang, S.-w., et al. *SUPERB: Speech Processing Universal PERformance Benchmark.* Interspeech 2021.
- **MediaPipe Face Detector** (Google): BlazeFace short-range model used for face detection.
- **Hugging Face Transformers**: model loading and inference for the speech emotion model.
- **ONNX Runtime**: fast CPU inference for the facial emotion model.
- **OpenCV**: video capture and on-frame rendering.
- **sounddevice**: microphone capture via PortAudio.

## License

Apache License 2.0. See [LICENSE](LICENSE).
