"""
Phototeam Buddy Web — Streamlit Edition  v0.7
=============================================
Modes   : Process a Video / Batch Process Photos / Generate Sibling
Scoring : MediaPipe blendshapes, expression-first, native-OCR penalty
Crop    : Face-anchored 3:2 smart crop (no blur / no padding)
Sibling : Reads IPTC + EXIF from source Getty, injects into output
"""

import io
import os
import sys
import tempfile
import random
import logging

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import streamlit as st

log = logging.getLogger("phototeam")

# ═══════════════════════════════════════════════════════════════════════════
# Constants & paths
# ═══════════════════════════════════════════════════════════════════════════

VERSION = "0.7"


def resource_path(relative: str) -> str:
    """Works in both dev and PyInstaller --onefile contexts."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative)


APP_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = resource_path("face_landmarker.task")
LOGO_PATH  = resource_path("WebMD_logo.png")

TARGET_W, TARGET_H = 1800, 1200
TARGET_RATIO       = TARGET_W / TARGET_H
NUM_CANDIDATES     = 30
NUM_FRAMES         = 6

METADATA_KEYWORDS  = "WebMD; health; thumbnail; medical; video"
METADATA_COPYRIGHT = "WebMD"
METADATA_DESC      = "WebMD Video Thumbnail"


# ═══════════════════════════════════════════════════════════════════════════
# MediaPipe — cached so the model loads once per server session
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_landmarker() -> vision.FaceLandmarker:
    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        num_faces=4,
        min_face_detection_confidence=0.4,
        output_face_blendshapes=True,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def _detect_faces(frame: np.ndarray):
    h, w = frame.shape[:2]
    scale = min(640 / w, 1.0)
    small = cv2.resize(frame, (int(w * scale), int(h * scale)))
    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return get_landmarker().detect(mp_img)


def _largest_face(result):
    best_idx, best_area, best_lm = -1, 0.0, None
    for fi, face_lm in enumerate(result.face_landmarks):
        xs = [lm.x for lm in face_lm]
        ys = [lm.y for lm in face_lm]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area > best_area:
            best_idx, best_area, best_lm = fi, area, face_lm
    return best_idx, best_area, best_lm


def _face_bbox_pixels(landmarks, img_w: int, img_h: int):
    xs = [lm.x * img_w for lm in landmarks]
    ys = [lm.y * img_h for lm in landmarks]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)


# ═══════════════════════════════════════════════════════════════════════════
# Platform-aware text density
# ═══════════════════════════════════════════════════════════════════════════

def _text_density(frame: np.ndarray) -> float:
    h, w = frame.shape[:2]
    scale = min(640 / w, 1.0)
    small = cv2.resize(frame, (int(w * scale), int(h * scale)))
    if sys.platform == "darwin":
        try:
            return _text_density_macos(small)
        except Exception:
            pass
    elif sys.platform == "win32":
        try:
            return _text_density_windows(small)
        except Exception:
            pass
    return _text_density_cv_fallback(small)


def _text_density_macos(small: np.ndarray) -> float:
    import Vision as VN
    from Foundation import NSData
    _, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
    ns_data = NSData.dataWithBytes_length_(jpg.tobytes(), len(jpg))
    handler = VN.VNImageRequestHandler.alloc().initWithData_options_(ns_data, None)
    request = VN.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(1)
    success, _ = handler.performRequests_error_([request], None)
    if not success:
        return 0.0
    results = request.results()
    if not results:
        return 0.0
    total = sum(
        obs.boundingBox().size.width * obs.boundingBox().size.height
        for obs in results
    )
    return min(total * 3.0, 1.0)


def _text_density_windows(small: np.ndarray) -> float:
    import asyncio
    from winrt.windows.media.ocr import OcrEngine
    from winrt.windows.globalization import Language
    from winrt.windows.graphics.imaging import (
        SoftwareBitmap, BitmapPixelFormat, BitmapAlphaMode,
    )

    async def _run():
        lang   = Language("en-US")
        engine = OcrEngine.try_create_from_language(lang)
        if engine is None:
            return 0.0
        h, w = small.shape[:2]
        bgra = cv2.cvtColor(small, cv2.COLOR_BGR2BGRA)
        bmp  = SoftwareBitmap.create_copy_from_buffer(
            bytes(bgra), BitmapPixelFormat.BGRA8,
            w, h, BitmapAlphaMode.PREMULTIPLIED,
        )
        result = await engine.recognize_async(bmp)
        if not result or not result.text:
            return 0.0
        return min(len(result.text.strip()) / 200.0, 1.0)

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run())
    finally:
        loop.close()


def _text_density_cv_fallback(small: np.ndarray) -> float:
    h, w  = small.shape[:2]
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 130)
    h_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, h_kern)
    n_lbl, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    hit = 0
    for i in range(1, n_lbl):
        x  = stats[i, cv2.CC_STAT_LEFT]
        y  = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if not (1.8 < bw / max(bh, 1) < 25 and 6 < bh < 70):
            continue
        in_zone = (
            (y + bh) / h > 0.68
            or y / h < 0.18
            or ((x / w < 0.28 or (x + bw) / w > 0.72)
                and (y / h < 0.28 or (y + bh) / h > 0.72))
        )
        if in_zone:
            hit += bw * bh
    return min(hit / (h * w), 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Frame extraction
# ═══════════════════════════════════════════════════════════════════════════

def _sample_positions(total: int, num: int, seed: int) -> list:
    start = int(total * 0.05)
    end   = int(total * 0.90)
    span  = end - start
    seg   = span / num
    rng   = random.Random(seed)
    return [
        rng.randint(
            int(start + i * seg),
            max(int(start + i * seg), int(start + (i + 1) * seg) - 1),
        )
        for i in range(num)
    ]


def extract_frames(video_path: str, num: int, seed: int) -> list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < num:
        raise ValueError("Video is too short.")
    frames = []
    for pos in _sample_positions(total, num, seed):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError("Could not read any frames from the video.")
    return frames


# ═══════════════════════════════════════════════════════════════════════════
# Scoring — expression-first
# ═══════════════════════════════════════════════════════════════════════════

_NOSE_TIP = 1


def _bs(blendshapes: list, name: str) -> float:
    for b in blendshapes:
        if b.category_name == name:
            return b.score
    return 0.0


def score_frame(frame: np.ndarray) -> float:
    """Expression-first scoring. Centred subjects are NOT penalised."""
    h, w  = frame.shape[:2]
    scale = min(640 / w, 1.0)
    small = cv2.resize(frame, (int(w * scale), int(h * scale)))
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    sharpness = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 400.0, 1.0)
    mean_v    = gray.mean() / 255.0
    exposure  = max(0.0, 1.0 - abs(mean_v - 0.45) * 2.2)
    text_pen  = _text_density(frame)

    rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = get_landmarker().detect(mp_img)

    if not result.face_landmarks:
        base = sharpness * 0.65 + exposure * 0.35
        return max(0.0, base - text_pen * 0.50)

    idx, area, lm = _largest_face(result)
    bs = result.face_blendshapes[idx] if result.face_blendshapes else []

    blink       = max(_bs(bs, "eyeBlinkLeft"), _bs(bs, "eyeBlinkRight"))
    eye_score   = 1.0 - blink
    jaw         = _bs(bs, "jawOpen")
    mouth_score = max(0.0, 1.0 - jaw * 2.5)
    face_score  = min(area * 8.0, 1.0)

    nose_x       = lm[_NOSE_TIP].x
    thirds_dist  = min(abs(nose_x - 0.333), abs(nose_x - 0.667))
    thirds_bonus = max(0.0, 0.05 - thirds_dist * 0.15)

    smile       = (_bs(bs, "mouthSmileLeft") + _bs(bs, "mouthSmileRight")) / 2
    smile_bonus = min(smile * 0.20, 0.10)

    base = (
        eye_score   * 0.35
        + mouth_score * 0.25
        + sharpness   * 0.15
        + face_score  * 0.10
        + exposure    * 0.08
        + thirds_bonus
        + smile_bonus
    )
    return max(0.0, base - text_pen * 0.50)


# ═══════════════════════════════════════════════════════════════════════════
# Face-anchored 3:2 crop
# ═══════════════════════════════════════════════════════════════════════════

def frame_to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def make_cropped_raw(frame: np.ndarray) -> Image.Image:
    """Face-anchored 3:2 crop. No blur, no padding."""
    h, w   = frame.shape[:2]
    result = _detect_faces(frame)
    has_face = bool(result.face_landmarks)

    if has_face:
        _, _, lm = _largest_face(result)
        cx, cy, fw, fh = _face_bbox_pixels(lm, w, h)

        if w / h >= TARGET_RATIO:
            crop_h = h
            crop_w = int(h * TARGET_RATIO)
        else:
            crop_w = w
            crop_h = int(w / TARGET_RATIO)

        min_crop_h = int(fh * 3.0)
        min_crop_w = int(min_crop_h * TARGET_RATIO)

        if crop_h < min_crop_h and min_crop_w <= w:
            crop_h, crop_w = min_crop_h, min_crop_w
        elif crop_w < min_crop_w and min_crop_h <= h:
            crop_w, crop_h = min_crop_w, min_crop_h

        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)
        if crop_w / crop_h > TARGET_RATIO:
            crop_w = int(crop_h * TARGET_RATIO)
        else:
            crop_h = int(crop_w / TARGET_RATIO)

        x0 = int(cx - crop_w * 0.45)
        y0 = int(cy - crop_h * 0.38)
        x0 = max(0, min(x0, w - crop_w))
        y0 = max(0, min(y0, h - crop_h))
    else:
        if w / h > TARGET_RATIO:
            crop_w = int(h * TARGET_RATIO)
            crop_h = h
            x0 = (w - crop_w) // 2
            y0 = 0
        else:
            crop_w = w
            crop_h = int(w / TARGET_RATIO)
            x0 = 0
            y0 = max(0, (h - crop_h) // 4)

    cropped = frame[y0 : y0 + crop_h, x0 : x0 + crop_w]
    r = cv2.resize(cropped, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LANCZOS4)
    return frame_to_pil(r)


def make_original_raw(frame: np.ndarray) -> Image.Image:
    h, w = frame.shape[:2]
    nh   = int(h * TARGET_W / w)
    r    = cv2.resize(frame, (TARGET_W, nh), interpolation=cv2.INTER_LANCZOS4)
    return frame_to_pil(r)


# ═══════════════════════════════════════════════════════════════════════════
# Metadata helpers  (web-adapted: operate on bytes, not file paths)
# ═══════════════════════════════════════════════════════════════════════════

def pil_to_jpeg_bytes(img: Image.Image, quality: int = 95) -> bytes:
    """JPEG with default WebMD EXIF tags."""
    exif = img.getexif()
    exif[0x010e] = METADATA_DESC
    exif[0x013b] = METADATA_COPYRIGHT
    exif[0x8298] = METADATA_COPYRIGHT
    exif[0x9c9c] = (METADATA_KEYWORDS + "\x00").encode("utf-16-le")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes(), quality=quality, optimize=True)
    return buf.getvalue()


def _cleanup(*paths: str):
    """Delete temp files and their iptcinfo3 backup siblings."""
    for p in paths:
        for candidate in (p, p + "~"):
            try:
                if os.path.exists(candidate):
                    os.unlink(candidate)
            except Exception:
                pass


def extract_source_metadata(file_bytes: bytes, filename: str) -> dict:
    """Read IPTC + EXIF from raw bytes via a short-lived temp file."""
    meta = {
        "exif_dict":      None,
        "iptc_keywords":  [],
        "iptc_caption":   "",
        "iptc_copyright": "",
        "source":         filename,
    }
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        import piexif
        try:
            meta["exif_dict"] = piexif.load(tmp_path)
        except Exception as exc:
            log.debug("piexif read: %s", exc)
        try:
            from iptcinfo3 import IPTCInfo
            info   = IPTCInfo(tmp_path, force=True)
            raw_kw = info["keywords"] or []
            meta["iptc_keywords"] = [
                k.decode("utf-8", errors="replace") if isinstance(k, bytes) else str(k)
                for k in raw_kw
            ]
            for field, key in [
                ("caption/abstract", "iptc_caption"),
                ("copyright notice", "iptc_copyright"),
            ]:
                val = info[field]
                if val:
                    meta[key] = (
                        val.decode("utf-8", errors="replace")
                        if isinstance(val, bytes) else str(val)
                    )
        except Exception as exc:
            log.debug("iptcinfo3 read: %s", exc)
    finally:
        _cleanup(tmp_path)
    return meta


def sibling_bytes_with_metadata(img: Image.Image, src_meta: dict) -> bytes:
    """Return styled JPEG bytes with the source IPTC/EXIF injected."""
    import piexif

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # ── EXIF ──────────────────────────────────────────────────────────
        exif_dict = src_meta.get("exif_dict")
        if exif_dict:
            w, h = img.size
            exif_dict["0th"][piexif.ImageIFD.ImageWidth]  = w
            exif_dict["0th"][piexif.ImageIFD.ImageLength] = h
            exif_dict.pop("thumbnail", None)
            exif_dict["1st"] = {}
            try:
                exif_bytes = piexif.dump(exif_dict)
            except Exception:
                exif_bytes = b""
        else:
            exif_bytes = b""

        if exif_bytes:
            img.save(tmp_path, "JPEG", quality=95, optimize=True, exif=exif_bytes)
        else:
            img.save(tmp_path, "JPEG", quality=95, optimize=True)

        # ── IPTC ──────────────────────────────────────────────────────────
        kw  = src_meta.get("iptc_keywords", [])
        cap = src_meta.get("iptc_caption",  "")
        cr  = src_meta.get("iptc_copyright","")
        if kw or cap or cr:
            try:
                from iptcinfo3 import IPTCInfo
                info = IPTCInfo(tmp_path, force=True)
                if kw:
                    info["keywords"] = [
                        k.encode("utf-8") if isinstance(k, str) else k for k in kw
                    ]
                if cap:
                    info["caption/abstract"] = cap
                if cr:
                    info["copyright notice"] = cr
                info.save()
            except Exception as exc:
                log.debug("iptcinfo3 write: %s", exc)

        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        _cleanup(tmp_path)


# ═══════════════════════════════════════════════════════════════════════════
# Session-state initialisation
# ═══════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "pairs":        [],     # list of (cropped_raw PIL, orig_raw PIL)
        "source_meta":  [],     # list of metadata dicts  (sibling mode only)
        "mode":         "video",
        "shuffle_seed": 0,
        "video_bytes":  None,   # raw bytes of uploaded video
        "video_name":   "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════

def _build_sidebar():
    """Render the sidebar mode selector."""
    with st.sidebar:
        if os.path.isfile(LOGO_PATH):
            st.image(LOGO_PATH, width=150)

        st.markdown("## Phototeam Buddy")
        st.caption(f"v{VERSION}  •  WebMD Thumbnail Toolkit")
        st.divider()

        st.markdown("**Mode**")
        mode_map = {
            "🎬  Process a Video":      "video",
            "🖼️  Batch Process Photos": "photos",
            "🔗  Generate Sibling":     "sibling",
        }
        choice   = st.radio("mode_radio", list(mode_map.keys()),
                             label_visibility="collapsed")
        new_mode = mode_map[choice]

        # Clear results when switching modes
        if new_mode != st.session_state.mode:
            st.session_state.mode        = new_mode
            st.session_state.pairs       = []
            st.session_state.source_meta = []


# ═══════════════════════════════════════════════════════════════════════════
# Results grid  (shared by all three modes)
# ═══════════════════════════════════════════════════════════════════════════

def _results_grid():
    pairs     = st.session_state.pairs
    meta_list = st.session_state.source_meta
    mode      = st.session_state.mode

    if not pairs:
        return

    st.success(f"✅ {len(pairs)} image(s) ready — click any button to download")
    st.divider()

    cols_per_row = 3
    for row_start in range(0, len(pairs), cols_per_row):
        cols = st.columns(cols_per_row, gap="medium")
        for offset, col in enumerate(cols):
            idx = row_start + offset
            if idx >= len(pairs):
                break
            cropped_raw, orig_raw = pairs[idx]

            with col:
                st.image(cropped_raw, use_container_width=True)

                ow, oh = orig_raw.size
                st.caption(f"**1800 × 1200**  •  original {ow} × {oh}")

                if mode == "sibling":
                    meta = meta_list[idx] if idx < len(meta_list) else {}
                    kw_n = len(meta.get("iptc_keywords", []))
                    st.caption(f"📄 `{meta.get('source','?')}`  •  {kw_n} keywords")
                    data = sibling_bytes_with_metadata(cropped_raw, meta)
                    base = os.path.splitext(meta.get("source", f"image_{idx+1}"))[0]
                    st.download_button(
                        "📥 Download with Metadata",
                        data,
                        file_name=f"{base}_1800x1200.jpg",
                        mime="image/jpeg",
                        key=f"sib_{idx}",
                        use_container_width=True,
                    )
                else:
                    d1, d2 = st.columns(2)
                    with d1:
                        st.download_button(
                            "📥 1800 × 1200",
                            pil_to_jpeg_bytes(cropped_raw),
                            file_name=f"webmd_thumb_{idx+1}_1800x1200.jpg",
                            mime="image/jpeg",
                            key=f"crop_{idx}",
                            use_container_width=True,
                        )
                    with d2:
                        st.download_button(
                            "📥 Original",
                            pil_to_jpeg_bytes(orig_raw),
                            file_name=f"webmd_thumb_{idx+1}_original.jpg",
                            mime="image/jpeg",
                            key=f"orig_{idx}",
                            use_container_width=True,
                        )


# ═══════════════════════════════════════════════════════════════════════════
# Mode UIs
# ═══════════════════════════════════════════════════════════════════════════

def _video_mode():
    st.subheader("🎬 Process a Video")
    st.caption(
        "Upload an MP4 and the app samples 30 candidate frames, scores each "
        "for expression, sharpness, and text-avoidance, then returns the best 6."
    )

    uploaded = st.file_uploader(
        "Select a video file",
        type=["mp4", "mov", "avi", "mkv"],
        key="video_upload",
    )
    if uploaded:
        # Cache the bytes so Shuffle can re-process without re-uploading
        st.session_state.video_bytes = uploaded.read()
        st.session_state.video_name  = uploaded.name

    has_video   = st.session_state.video_bytes is not None
    has_results = bool(st.session_state.pairs)

    c1, c2 = st.columns([1, 1])
    with c1:
        process_btn = st.button(
            "▶  Process Video", disabled=not has_video,
            type="primary", use_container_width=True,
        )
    with c2:
        shuffle_btn = st.button(
            "🔀  Shuffle Frames", disabled=not (has_video and has_results),
            use_container_width=True,
        )

    if process_btn or shuffle_btn:
        if shuffle_btn:
            st.session_state.shuffle_seed += 1
        _run_video()


def _run_video():
    seed = st.session_state.shuffle_seed
    ext  = os.path.splitext(st.session_state.video_name)[-1] or ".mp4"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(st.session_state.video_bytes)
        tmp_path = tmp.name

    try:
        bar = st.progress(0, text="Sampling candidate frames...")
        candidates = extract_frames(tmp_path, NUM_CANDIDATES, seed)

        scored = []
        for i, frame in enumerate(candidates):
            scored.append((i, frame, score_frame(frame)))
            bar.progress(0.20 + 0.50 * (i + 1) / len(candidates),
                         text=f"Scoring frame {i+1} / {len(candidates)}…")

        scored.sort(key=lambda t: t[2], reverse=True)
        best = [candidates[i] for i, _, _ in scored[:NUM_FRAMES]]
        best = [candidates[i] for i in sorted(i for i, _, _ in scored[:NUM_FRAMES])]

        pairs = []
        for j, frame in enumerate(best):
            bar.progress(0.70 + 0.28 * (j + 1) / len(best),
                         text=f"Cropping image {j+1} / {len(best)}…")
            pairs.append((make_cropped_raw(frame), make_original_raw(frame)))

        st.session_state.pairs       = pairs
        st.session_state.source_meta = []
        bar.progress(1.0, text="Done!")

    except Exception as exc:
        st.error(f"Processing failed: {exc}")
    finally:
        _cleanup(tmp_path)


def _photos_mode():
    st.subheader("🖼️ Batch Process Photos")
    st.caption(
        "Upload one or more images. The app scores them for expression and "
        "sharpness, selects the best, and applies a face-anchored 1800 × 1200 crop."
    )

    files = st.file_uploader(
        "Select photos",
        type=["jpg","jpeg","png","bmp","tiff","webp"],
        accept_multiple_files=True,
        key="photo_upload",
    )

    if st.button("▶  Process Photos", disabled=not files,
                 type="primary"):
        frames = []
        bar    = st.progress(0, text="Loading images…")
        for i, uf in enumerate(files):
            raw = np.frombuffer(uf.read(), np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(img)
            bar.progress((i + 1) / len(files))

        if not frames:
            st.error("No valid images could be read.")
            return

        if len(frames) > NUM_FRAMES:
            scored = []
            for i, f in enumerate(frames):
                bar.progress(i / len(frames),
                             text=f"Scoring image {i+1} / {len(frames)}…")
                scored.append((i, f, score_frame(f)))
            scored.sort(key=lambda t: t[2], reverse=True)
            frames = [f for _, f, _ in scored[:NUM_FRAMES]]

        pairs = []
        for j, frame in enumerate(frames):
            bar.progress(0.80 + 0.18 * (j + 1) / len(frames),
                         text=f"Cropping image {j+1} / {len(frames)}…")
            pairs.append((make_cropped_raw(frame), make_original_raw(frame)))

        st.session_state.pairs       = pairs
        st.session_state.source_meta = []
        bar.progress(1.0, text="Done!")


def _sibling_mode():
    st.subheader("🔗 Generate Sibling")
    st.caption(
        "Upload raw Getty images. The app extracts each file's IPTC/EXIF metadata, "
        "applies a face-anchored 1800 × 1200 crop, and injects the original "
        "metadata into the output — ready for CMS upload."
    )

    files = st.file_uploader(
        "Select Getty images",
        type=["jpg","jpeg","tiff","png"],
        accept_multiple_files=True,
        key="sibling_upload",
    )

    if st.button("▶  Process Images", disabled=not files,
                 type="primary"):
        frames, metas = [], []
        bar = st.progress(0, text="Loading and extracting metadata…")

        for i, uf in enumerate(files):
            file_bytes = uf.read()
            raw = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if img is None:
                continue
            frames.append(img)
            metas.append(extract_source_metadata(file_bytes, uf.name))
            bar.progress((i + 1) / len(files),
                         text=f"Loaded {uf.name}")

        if not frames:
            st.error("No valid images could be read.")
            return

        pairs = []
        for j, frame in enumerate(frames):
            bar.progress(0.80 + 0.18 * (j + 1) / len(frames),
                         text=f"Cropping image {j+1} / {len(frames)}…")
            pairs.append((make_cropped_raw(frame), make_original_raw(frame)))

        st.session_state.pairs       = pairs
        st.session_state.source_meta = metas
        bar.progress(1.0, text="Done!")


# ═══════════════════════════════════════════════════════════════════════════
# App entry point  (st.set_page_config must be the first st.* call)
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Phototeam Buddy",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded",
)

_init_state()
_build_sidebar()

# Header
st.markdown(
    """
    <div style="background:#203d76;padding:16px 24px;border-radius:8px;
                margin-bottom:8px;">
        <span style="color:white;font-size:22px;font-weight:700;">
            📸 Phototeam Buddy
        </span>
        &nbsp;
        <span style="color:#7a9cc8;font-size:13px;">
            WebMD Photo &amp; Video Thumbnail Toolkit
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Route to the active mode
mode = st.session_state.mode
if mode == "video":
    _video_mode()
elif mode == "photos":
    _photos_mode()
else:
    _sibling_mode()

_results_grid()
