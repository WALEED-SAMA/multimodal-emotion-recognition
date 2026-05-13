"""Real-time multimodal emotion recognition.

Fuses per-frame facial expression probabilities with rolling speech emotion
probabilities and displays a single fused emotion label per detected face.

Webcam by default. Pass --no-audio for vision only, --no-display for headless
runs, --csv for a per-frame timeline.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------
EMOTIEFFLIB_CLASSES = [
    "Anger", "Contempt", "Disgust", "Fear",
    "Happiness", "Neutral", "Sadness", "Surprise",
]
EMOTIEFFLIB_IDX = {c: i for i, c in enumerate(EMOTIEFFLIB_CLASSES)}

# Audio label -> visual class. Handles the common SER naming conventions:
#   short forms produced by the SUPERB benchmark (neu/hap/ang/sad)
#   long forms used by RAVDESS-style 8-class models
AUDIO_LABEL_TO_EMOTIEFFLIB = {
    "neu": "Neutral",
    "hap": "Happiness",
    "ang": "Anger",
    "sad": "Sadness",
    "angry": "Anger",
    "anger": "Anger",
    "calm": "Neutral",
    "disgust": "Disgust",
    "fear": "Fear",
    "fearful": "Fear",
    "happy": "Happiness",
    "happiness": "Happiness",
    "neutral": "Neutral",
    "sadness": "Sadness",
    "surprise": "Surprise",
    "surprised": "Surprise",
}

# Applied when a checkpoint exposes anonymous LABEL_n classes. Override via
# --label-order when the model isn't RAVDESS-ordered.
_DEFAULT_ANON_RAVDESS_LABELS = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprised",
]


# ---------------------------------------------------------------------------
# Face detection (MediaPipe Tasks API)
# ---------------------------------------------------------------------------
_MP_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)


def _ensure_mp_face_model() -> str:
    """Download the BlazeFace tflite asset on first use; cache under ~/.cache/."""
    cache_dir = os.path.expanduser("~/.cache/multimodal-emotion")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "blaze_face_short_range.tflite")
    if not os.path.isfile(path):
        print(f"[face] downloading detector to {path}")
        urllib.request.urlretrieve(_MP_FACE_MODEL_URL, path)
    return path


def load_face_detector(min_face: int):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    options = mp_vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=_ensure_mp_face_model()),
        min_detection_confidence=0.5,
    )
    detector = mp_vision.FaceDetector.create_from_options(options)

    def detect(frame_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        res = detector.detect(mp_image)
        if not res.detections:
            return []
        boxes = []
        for det in res.detections:
            bb = det.bounding_box
            x = max(0, int(bb.origin_x))
            y = max(0, int(bb.origin_y))
            bw = int(bb.width)
            bh = int(bb.height)
            if bw < min_face or bh < min_face:
                continue
            boxes.append((x, y, bw, bh))
        return boxes

    return detect


def expand_box(x: int, y: int, w: int, h: int, frame_w: int, frame_h: int,
               pct: float = 0.10) -> Tuple[int, int, int, int]:
    dx = int(w * pct)
    dy = int(h * pct)
    nx = max(0, x - dx)
    ny = max(0, y - dy)
    nw = min(frame_w - nx, w + 2 * dx)
    nh = min(frame_h - ny, h + 2 * dy)
    return nx, ny, nw, nh


# ---------------------------------------------------------------------------
# Visual emotion recognizer
# ---------------------------------------------------------------------------
def load_visual_recognizer(engine: str, model_name: str, device: str):
    from emotiefflib.facial_analysis import EmotiEffLibRecognizer
    return EmotiEffLibRecognizer(engine=engine, model_name=model_name, device=device)


def visual_probs(fer, crops: List[np.ndarray]) -> List[np.ndarray]:
    """Run the visual recognizer on a batch of crops, return per-crop 8-class softmax probs."""
    emotions, scores = fer.predict_emotions(crops, logits=False)
    out: List[np.ndarray] = []
    for s in scores:
        s = np.asarray(s, dtype=np.float32)
        if fer.is_mtl:
            s = s[:-2]
        if abs(float(s.sum()) - 1.0) > 0.05:
            e = np.exp(s - np.max(s))
            s = e / e.sum()
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# Speech emotion thread
# ---------------------------------------------------------------------------
def remap_audio_probs(audio_probs: np.ndarray, id2label: Dict[int, str]) -> np.ndarray:
    """Reorder the SER output into EmotiEffLib's class order.

    Classes that the audio model doesn't speak to (e.g. Contempt for SUPERB or
    RAVDESS models) stay zero, and the result is renormalized.
    """
    out = np.zeros(len(EMOTIEFFLIB_CLASSES), dtype=np.float32)
    for audio_idx, audio_label in id2label.items():
        target = AUDIO_LABEL_TO_EMOTIEFFLIB.get(str(audio_label).lower())
        if target is None:
            continue
        out[EMOTIEFFLIB_IDX[target]] += audio_probs[audio_idx]
    s = float(out.sum())
    if s > 0:
        out /= s
    return out


class SpeechEmotionThread(threading.Thread):
    """Pulls audio from the mic in a background thread and runs an SER model on it.

    Shared state: `latest` is updated under `lock` with the most recent probs
    (already remapped to EmotiEffLib's class order) and a wall-clock timestamp.
    """

    def __init__(self, model_name: str, samplerate: int, window_s: float, hop_s: float,
                 label_order: Optional[List[str]] = None):
        super().__init__(daemon=True)
        self.model_name = model_name
        self.samplerate = samplerate
        self.window_n = int(samplerate * window_s)
        self.hop_n = int(samplerate * hop_s)
        self.label_order = label_order
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest: Dict = {"probs": None, "timestamp": 0.0, "raw_label": None}
        self.ready_event = threading.Event()
        self._ring = np.zeros(self.window_n * 2, dtype=np.float32)
        self._ring_lock = threading.Lock()
        self._write_pos = 0

    def stop(self) -> None:
        self.stop_event.set()

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            print(f"[audio] sounddevice status: {status}", file=sys.stderr)
        chunk = indata[:, 0].astype(np.float32, copy=False)
        n = len(chunk)
        with self._ring_lock:
            end = self._write_pos + n
            if end <= len(self._ring):
                self._ring[self._write_pos:end] = chunk
            else:
                first = len(self._ring) - self._write_pos
                self._ring[self._write_pos:] = chunk[:first]
                self._ring[:n - first] = chunk[first:]
            self._write_pos = (self._write_pos + n) % len(self._ring)

    def _latest_window(self) -> np.ndarray:
        with self._ring_lock:
            buf = np.concatenate([self._ring[self._write_pos:], self._ring[:self._write_pos]])
        return buf[-self.window_n:].copy()

    def run(self) -> None:
        import sounddevice as sd
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        print(f"[audio] loading SER model: {self.model_name}")
        extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        model = AutoModelForAudioClassification.from_pretrained(self.model_name)
        # Some checkpoints ship weights as fp16; conv layers expect fp32.
        model = model.float()
        model.eval()

        id2label = {int(k): v for k, v in model.config.id2label.items()}
        if self.label_order is not None:
            id2label = {i: name for i, name in enumerate(self.label_order)}
            print(f"[audio] applying user label-order override: {id2label}")
        elif all(str(v).upper().startswith("LABEL_") for v in id2label.values()):
            if len(id2label) == 8:
                id2label = {i: c for i, c in enumerate(_DEFAULT_ANON_RAVDESS_LABELS)}
                print(
                    f"[audio] anonymous labels detected (8 classes); assuming RAVDESS "
                    f"order: {id2label}. Pass --label-order to override."
                )
            else:
                print(
                    f"[audio] WARNING: anonymous labels with {len(id2label)} classes and "
                    f"no --label-order override; audio fusion will be unreliable."
                )
        print(f"[audio] SER classes: {[id2label[i] for i in sorted(id2label)]}")

        try:
            stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=1,
                dtype="float32",
                blocksize=0,
                callback=self._audio_callback,
            )
        except Exception as e:
            print(
                f"[audio] ERROR opening microphone: {e}\n"
                "On macOS, grant microphone permission to your terminal app: "
                "System Settings -> Privacy & Security -> Microphone, quit the "
                "terminal with Cmd-Q, relaunch.",
                file=sys.stderr,
            )
            self.ready_event.set()
            return

        self.ready_event.set()
        with stream:
            time.sleep(self.window_n / self.samplerate)
            while not self.stop_event.is_set():
                window = self._latest_window()
                if float(np.max(np.abs(window))) < 0.005:
                    time.sleep(self.hop_n / self.samplerate)
                    continue
                inputs = extractor(
                    window, sampling_rate=self.samplerate, return_tensors="pt", padding=True
                )
                with torch.no_grad():
                    logits = model(**inputs).logits[0]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                raw_idx = int(np.argmax(probs))
                remapped = remap_audio_probs(probs, id2label)
                with self.lock:
                    self.latest = {
                        "probs": remapped,
                        "timestamp": time.time(),
                        "raw_label": id2label[raw_idx],
                    }
                time.sleep(self.hop_n / self.samplerate)

    def snapshot(self) -> Tuple[Optional[np.ndarray], float, Optional[str]]:
        with self.lock:
            return self.latest["probs"], self.latest["timestamp"], self.latest["raw_label"]


# ---------------------------------------------------------------------------
# Fusion + display
# ---------------------------------------------------------------------------
def fuse(p_visual: np.ndarray, p_audio: Optional[np.ndarray], audio_age_s: float,
         alpha: float, stale_s: float) -> Tuple[np.ndarray, float]:
    if p_audio is None or audio_age_s > stale_s:
        return p_visual, 1.0
    return alpha * p_visual + (1.0 - alpha) * p_audio, alpha


def argmax_to_label(probs: np.ndarray) -> Tuple[str, float]:
    i = int(np.argmax(probs))
    return EMOTIEFFLIB_CLASSES[i], float(probs[i])


def draw_overlay(frame_bgr: np.ndarray,
                 boxes: List[Tuple[int, int, int, int]],
                 labels: List[Tuple[str, float]],
                 fps: float,
                 audio_status: str,
                 verbose_lines: Optional[List[List[str]]] = None) -> None:
    for i, ((x, y, w, h), (label, conf)) in enumerate(zip(boxes, labels)):
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{label} {conf * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame_bgr, (x, y - th - 10), (x + tw + 6, y), (0, 255, 0), -1)
        cv2.putText(
            frame_bgr, text, (x + 3, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA,
        )
        if verbose_lines and i < len(verbose_lines):
            for j, line in enumerate(verbose_lines[i]):
                cv2.putText(
                    frame_bgr, line, (x, y + h + 18 + 18 * j),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    frame_bgr, line, (x, y + h + 18 + 18 * j),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
                )

    cv2.putText(
        frame_bgr, f"{fps:5.1f} FPS  |  audio: {audio_status}", (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time multimodal emotion recognition (face + speech)."
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--visual-model", default="enet_b0_8_best_vgaf",
                        help="EmotiEffLib model name (default: enet_b0_8_best_vgaf).")
    parser.add_argument("--engine", default="onnx", choices=["onnx", "torch"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-face", type=int, default=40)
    parser.add_argument("--no-audio", action="store_true",
                        help="Disable speech, fall back to visual-only.")
    parser.add_argument(
        "--ser-model",
        default="superb/hubert-base-superb-er",
        help="HuggingFace speech-emotion model. Default is verified 4-class "
             "(neu/hap/ang/sad). 8-class options exist but most published "
             "checkpoints have broken classifier heads or anonymous labels.",
    )
    parser.add_argument(
        "--label-order",
        help="Comma-separated class labels for SER models that expose anonymous "
             "LABEL_n classes (e.g. --label-order angry,happy,sad,neutral).",
    )
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="Visual weight in fusion. 1.0 = visual only.")
    parser.add_argument("--audio-window", type=float, default=2.0)
    parser.add_argument("--audio-hop", type=float, default=1.0)
    parser.add_argument("--stale-s", type=float, default=3.0,
                        help="Ignore audio predictions older than this (seconds).")
    parser.add_argument("--csv", help="Per-frame CSV output.")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Stop after N seconds (0 = run until 'q').")
    parser.add_argument("--verbose", action="store_true",
                        help="Draw per-modality breakdown below each face.")
    args = parser.parse_args()

    print(f"[run] loading visual model: {args.engine} / {args.visual_model}")
    fer = load_visual_recognizer(args.engine, args.visual_model, args.device)
    detect = load_face_detector(args.min_face)
    print("[run] face detector: mediapipe (Tasks API)")

    audio_thread: Optional[SpeechEmotionThread] = None
    if not args.no_audio:
        label_order = None
        if args.label_order:
            label_order = [s.strip() for s in args.label_order.split(",") if s.strip()]
        audio_thread = SpeechEmotionThread(
            model_name=args.ser_model,
            samplerate=16000,
            window_s=args.audio_window,
            hop_s=args.audio_hop,
            label_order=label_order,
        )
        audio_thread.start()
        print("[run] waiting for speech model to load (cold cache may take a few minutes)...")
        audio_thread.ready_event.wait(timeout=300.0)
        if audio_thread.ready_event.is_set():
            print("[run] speech model ready (or failed cleanly — see audio messages above).")
        else:
            print("[run] WARNING: speech model did not become ready within 300s; continuing visual-only.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(
            "ERROR: cannot open webcam. Grant camera permission to your terminal app "
            "(System Settings -> Privacy & Security -> Camera) and relaunch.",
            file=sys.stderr,
        )
        if audio_thread:
            audio_thread.stop()
        raise SystemExit(1)

    csv_fp = None
    csv_writer = None
    if args.csv:
        csv_fp = open(args.csv, "w", newline="")
        cols = (["frame", "timestamp_s", "face_idx", "fused_emotion", "fused_conf",
                 "visual_emotion", "visual_conf", "audio_emotion", "audio_conf",
                 "alpha_effective", "audio_age_s"]
                + [f"fused_{c}" for c in EMOTIEFFLIB_CLASSES])
        csv_writer = csv.writer(csv_fp)
        csv_writer.writerow(cols)

    t_start = time.time()
    frame_idx = 0
    t_window_start = time.time()
    t_window_frames = 0
    fps = 0.0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            fh, fw = frame_rgb.shape[:2]

            boxes = detect(frame_rgb)
            labels: List[Tuple[str, float]] = []
            verbose_lines: List[List[str]] = []

            audio_status = "off" if args.no_audio else "warming"
            audio_age = float("inf")
            p_audio: Optional[np.ndarray] = None
            audio_raw_label: Optional[str] = None
            if audio_thread is not None:
                p_audio, ts, audio_raw_label = audio_thread.snapshot()
                audio_age = time.time() - ts if ts > 0 else float("inf")
                if p_audio is None:
                    audio_status = "warming"
                elif audio_age > args.stale_s:
                    audio_status = f"stale ({audio_age:.1f}s)"
                else:
                    audio_status = f"{audio_raw_label} ({audio_age:.1f}s)"

            if boxes:
                crops: List[np.ndarray] = []
                for (x, y, w, h) in boxes:
                    ex, ey, ew, eh = expand_box(x, y, w, h, fw, fh)
                    crops.append(frame_rgb[ey:ey + eh, ex:ex + ew, :])
                v_probs_list = visual_probs(fer, crops)
                for i, p_v in enumerate(v_probs_list):
                    p_fused, alpha_eff = fuse(p_v, p_audio, audio_age, args.alpha, args.stale_s)
                    fused_label, fused_conf = argmax_to_label(p_fused)
                    labels.append((fused_label, fused_conf))

                    if args.verbose:
                        vl, vc = argmax_to_label(p_v)
                        if p_audio is None or audio_age > args.stale_s:
                            line2 = "audio: --"
                        else:
                            al, ac = argmax_to_label(p_audio)
                            line2 = f"audio: {al} {ac * 100:.0f}%"
                        verbose_lines.append([
                            f"visual: {vl} {vc * 100:.0f}%",
                            line2,
                            f"a={alpha_eff:.2f}",
                        ])

                    if csv_writer is not None:
                        ts_now = time.time() - t_start
                        vl, vc = argmax_to_label(p_v)
                        if p_audio is None or audio_age > args.stale_s:
                            al, ac_str = "", ""
                        else:
                            al, ac = argmax_to_label(p_audio)
                            ac_str = f"{ac:.4f}"
                        row = [
                            frame_idx, f"{ts_now:.3f}", i, fused_label, f"{fused_conf:.4f}",
                            vl, f"{vc:.4f}", al, ac_str,
                            f"{alpha_eff:.2f}",
                            f"{audio_age:.3f}" if audio_age != float("inf") else "",
                        ] + [f"{p_fused[j]:.4f}" for j in range(len(EMOTIEFFLIB_CLASSES))]
                        csv_writer.writerow(row)

            t_window_frames += 1
            if time.time() - t_window_start >= 0.5:
                fps = t_window_frames / (time.time() - t_window_start)
                t_window_start = time.time()
                t_window_frames = 0

            if not args.no_display:
                draw_overlay(frame_bgr, boxes, labels, fps, audio_status,
                             verbose_lines if args.verbose else None)
                cv2.imshow("Multimodal Emotion Recognition", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_idx += 1

            if args.duration and (time.time() - t_start) >= args.duration:
                break
    finally:
        cap.release()
        if csv_fp is not None:
            csv_fp.close()
        if audio_thread is not None:
            audio_thread.stop()
        if not args.no_display:
            cv2.destroyAllWindows()

    print(f"[run] processed {frame_idx} frames, final FPS ~{fps:.1f}")
    if args.csv:
        print(f"[run] timeline written to {args.csv}")


if __name__ == "__main__":
    main()
