import csv
import gc
import os
import shutil
import tempfile
import uuid
import zipfile
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image as PILImage
from streamlit_cropper import st_cropper

PREVIEW_MAX_DIM = 1800


# --- Utility functions ---
def decode_image_bytes(data, source_name):
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode uploaded image '{source_name}'.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_image_from_upload(uploaded_file):
    uploaded_file.seek(0)
    data = uploaded_file.read()
    uploaded_file.seek(0)
    return decode_image_bytes(data, uploaded_file.name)


def load_image_from_path(path, source_name=None):
    with open(path, "rb") as fh:
        data = fh.read()
    return decode_image_bytes(data, source_name or os.path.basename(path))


def scale_for_preview(img, max_dim=PREVIEW_MAX_DIM):
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img
    factor = max_dim / float(longest)
    new_w = max(1, int(round(w * factor)))
    new_h = max(1, int(round(h * factor)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def encode_image_bytes(img, filename):
    ext = os.path.splitext(filename)[1].lower() or ".png"
    if ext not in {".png", ".jpg", ".jpeg"}:
        ext = ".png"
    success, im_buf = cv2.imencode(ext, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not success:
        raise ValueError(f"Could not encode processed image '{filename}'.")
    return im_buf.tobytes(), ext


def apply_normalization(img, alpha):
    if alpha is None:
        return img
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)


def ensure_session_defaults():
    if "scale_set" not in st.session_state:
        st.session_state.scale_set = False
        st.session_state.scale = {"factor": 1.0, "unit": "pixels"}
    if "session_dir" not in st.session_state or not os.path.isdir(st.session_state.session_dir):
        st.session_state.session_dir = tempfile.mkdtemp(prefix="orbis_")
    if "upload_manifest" not in st.session_state:
        st.session_state.upload_manifest = []
    if "upload_signature" not in st.session_state:
        st.session_state.upload_signature = None
    if "generated_zip" not in st.session_state:
        st.session_state.generated_zip = None


def clear_generated_zip():
    generated_zip = st.session_state.get("generated_zip")
    if generated_zip and os.path.exists(generated_zip["path"]):
        os.remove(generated_zip["path"])
    st.session_state.generated_zip = None


def clear_upload_state(remove_session_dir=False):
    clear_generated_zip()
    session_dir = st.session_state.get("session_dir")
    if session_dir and os.path.isdir(session_dir):
        uploads_dir = os.path.join(session_dir, "uploads")
        if os.path.isdir(uploads_dir):
            shutil.rmtree(uploads_dir)
    st.session_state.upload_manifest = []
    st.session_state.upload_signature = None
    if remove_session_dir:
        if session_dir and os.path.isdir(session_dir):
            shutil.rmtree(session_dir, ignore_errors=True)
        st.session_state.session_dir = tempfile.mkdtemp(prefix="orbis_")


def upload_signature(files):
    return tuple((f.name, getattr(f, "size", None)) for f in files)



def current_processing_signature(manifest, settings, scale):
    manifest_sig = tuple((item["name"], item["size"], item["path"]) for item in manifest)
    settings_sig = tuple(sorted(settings.items()))
    scale_sig = (scale["factor"], scale["unit"])
    return manifest_sig, settings_sig, scale_sig


def prepare_upload_manifest(files):
    session_dir = st.session_state.session_dir
    uploads_dir = os.path.join(session_dir, "uploads")
    if os.path.isdir(uploads_dir):
        shutil.rmtree(uploads_dir)
    os.makedirs(uploads_dir, exist_ok=True)

    manifest = []
    gray_means = []
    prep_progress = st.progress(0, text="Preparing uploaded images...")

    total = len(files)
    for idx, uploaded_file in enumerate(files, start=1):
        uploaded_file.seek(0)
        buffer = uploaded_file.getbuffer()
        ext = os.path.splitext(uploaded_file.name)[1] or ".bin"
        token = f"{idx:04d}_{uuid.uuid4().hex}{ext}"
        file_path = os.path.join(uploads_dir, token)
        with open(file_path, "wb") as fh:
            fh.write(buffer)

        img = decode_image_bytes(buffer, uploaded_file.name)
        gray_mean = float(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).mean())
        h, w = img.shape[:2]
        manifest.append(
            {
                "name": uploaded_file.name,
                "size": getattr(uploaded_file, "size", len(buffer)),
                "path": file_path,
                "width": w,
                "height": h,
                "gray_mean": gray_mean,
            }
        )
        gray_means.append(gray_mean)
        del img
        gc.collect()
        prep_progress.progress(idx / total, text=f"Preparing uploaded images... ({idx}/{total})")

    prep_progress.empty()

    global_mean = float(np.mean(gray_means)) if gray_means else 1.0
    for item in manifest:
        item["norm_alpha"] = global_mean / (item["gray_mean"] + 1e-6)
    return manifest


# Circular crop
def circular_crop(img):
    """Crop image to a central circle."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    radius = min(center)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    return cv2.bitwise_and(img, img, mask=mask)


# Optional background subtraction via morphological opening
def subtract_background_auto(img, ksize):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    k = max(1, int(ksize))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), np.uint8)
    background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    return cv2.absdiff(gray, background)


# Denoise and threshold helpers
def remove_noise(img, ksize):
    k = max(1, int(ksize))
    if k % 2 == 0:
        k += 1
    return cv2.medianBlur(img, k)


def threshold_image(img, thresh_val, invert):
    _, th = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_not(th) if invert else th


def close_holes(img, ksize):
    k = max(1, int(ksize))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


# --- Processing pipeline ---
def process_image(img, settings):
    h, w = img.shape[:2]
    cf = settings["zoom"]
    ch = int(h / cf)
    cw = int(w / cf)
    y1, x1 = (h - ch) // 2, (w - cw) // 2
    cropped = img[y1 : y1 + ch, x1 : x1 + cw]

    if settings["contrast"] != 1.0:
        cropped = cv2.convertScaleAbs(cropped, alpha=settings["contrast"], beta=0)
    if settings["circular_crop"]:
        cropped = circular_crop(cropped)
    proc = (
        subtract_background_auto(cropped, settings["bg_ks"])
        if settings["bg_sub"]
        else cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    )
    if settings["noise"] > 0:
        proc = remove_noise(proc, settings["noise"])
    proc = threshold_image(proc, settings["th_val"], settings["invert"])
    if settings["hole_fill"] > 0:
        proc = close_holes(proc, settings["hole_fill"])
    contours, _ = cv2.findContours(proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if settings["single"] and contours:
        cx, cy = cw // 2, ch // 2
        md = float("inf")
        chosen = None
        for c in contours:
            moments = cv2.moments(c)
            if moments["m00"]:
                cx0 = int(moments["m10"] / moments["m00"])
                cy0 = int(moments["m01"] / moments["m00"])
                d = (cx0 - cx) ** 2 + (cy0 - cy) ** 2
                if d < md:
                    md, chosen = d, c
        contours = [chosen] if chosen is not None else []
    area_px = sum(cv2.contourArea(c) for c in contours)
    overlay = img.copy()
    roi = overlay[y1 : y1 + ch, x1 : x1 + cw]
    cv2.drawContours(roi, contours, -1, (0, 255, 0), 2)
    overlay[y1 : y1 + ch, x1 : x1 + cw] = roi
    return overlay, area_px


def build_results_zip(manifest, settings, scale):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = os.path.join(st.session_state.session_dir, f"OrbisResults_{ts}.zip")
    unit = scale["unit"]
    used_names = set()
    rows = []
    progress = st.progress(0, text="Processing images...")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        total = len(manifest)
        for idx, item in enumerate(manifest, start=1):
            img = load_image_from_path(item["path"], item["name"])
            if settings["norm"]:
                img = apply_normalization(img, item["norm_alpha"])
            overlay, area_px = process_image(img, settings)
            encoded_bytes, ext = encode_image_bytes(overlay, item["name"])

            archive_name = item["name"]
            if archive_name in used_names:
                stem, _ = os.path.splitext(item["name"])
                suffix = 2
                while f"{stem}__{suffix}{ext}" in used_names:
                    suffix += 1
                archive_name = f"{stem}__{suffix}{ext}"
            used_names.add(archive_name)

            zf.writestr(archive_name, encoded_bytes)
            area = area_px * (scale["factor"] ** 2)
            rows.append((item["name"], area))
            del img, overlay, encoded_bytes
            gc.collect()
            progress.progress(idx / total, text=f"Processing images... ({idx}/{total})")

        csv_buf = tempfile.SpooledTemporaryFile(mode="w+", max_size=1_000_000, newline="")
        writer = csv.writer(csv_buf)
        writer.writerow(["filename", f"Area ({unit}^2)"])
        writer.writerows(rows)
        csv_buf.seek(0)
        zf.writestr(f"areas({unit}^2).csv", csv_buf.read())
        csv_buf.close()

        log = f"Date:{datetime.now()}\nScale:{scale}\nSettings:{settings}\n"
        zf.writestr("log.txt", log)

    progress.empty()
    return zip_path, os.path.basename(zip_path)


# --- Streamlit App ---
def main():
    ensure_session_defaults()

    full_logo = "LogoSpaceMicrobesLab_White.png"
    small_icon = "Icon Space Microbes Lab.png"
    st.logo(full_logo, icon_image=small_icon, size="large")

    st.image("Orbis Logo.png", width=400)

    # Scale calibration
    scale_img = st.sidebar.file_uploader("Upload ruler image (optional)", type=["png", "jpg", "jpeg"])
    if scale_img and not st.session_state.scale_set:
        st.sidebar.subheader("Scale Calibration")
        try:
            pil = PILImage.fromarray(load_image_from_upload(scale_img))
        except ValueError as exc:
            st.sidebar.error(str(exc))
            return
        cropped = st_cropper(
            pil,
            realtime_update=True,
            box_color="#FF0000",
            aspect_ratio=(10, 1),
            return_type="image",
            key="scale_cropper",
        )
        if cropped:
            px_len = cropped.width
            length = st.sidebar.number_input("Real-world length", 0.0, 1e6, 1.0)
            unit = st.sidebar.text_input("Unit label", "cm")
            if st.sidebar.button("Confirm scale and continue"):
                if px_len > 0 and length > 0:
                    st.session_state.scale = {"factor": length / px_len, "unit": unit}
                st.session_state.scale_set = True
        return

    scale = st.session_state.scale

    st.sidebar.markdown("---")

    zoom = st.sidebar.slider("Zoom factor", 1.0, 5.0, 1.0, 0.1)
    contrast = st.sidebar.slider("Contrast", 1.0, 3.0, 1.0, 0.1)
    circular_crop_opt = st.sidebar.checkbox("Circular crop")
    single = st.sidebar.checkbox("Single colony")
    th_val = st.sidebar.slider("Threshold", 0, 255, 127)

    advanced = st.sidebar.checkbox("Advanced mode")
    bg_sub = False
    bg_ks = 0
    invert = False
    noise = 0
    hole_fill = 0
    norm = True

    st.sidebar.markdown("---")
    if advanced:
        bg_sub = st.sidebar.checkbox("Subtract background [Experimental]")
        if bg_sub and st.session_state.scale_set:
            bg_ks = st.sidebar.slider("BG kernel size", 3, 101, 15, 2)
        invert = st.sidebar.checkbox("Invert threshold")
        noise = st.sidebar.slider("Noise removal", 0, 15, 0)
        hole_fill = st.sidebar.slider("Hole fill", 0, 50, 0)
        norm = st.sidebar.checkbox("Brightness normalization", value=True)

    general = {
        "zoom": zoom,
        "contrast": contrast,
        "circular_crop": circular_crop_opt,
        "bg_sub": bg_sub,
        "bg_ks": bg_ks,
        "th_val": th_val,
        "invert": invert,
        "noise": noise,
        "hole_fill": hole_fill,
        "single": single,
        "norm": norm,
    }

    imgs = st.file_uploader("Upload images to analyze", accept_multiple_files=True, type=["png", "jpg", "jpeg"])
    if not imgs:
        clear_upload_state()
        return

    sig = upload_signature(imgs)
    if sig != st.session_state.upload_signature:
        try:
            manifest = prepare_upload_manifest(imgs)
        except ValueError as exc:
            clear_upload_state()
            st.error(str(exc))
            return
        st.session_state.upload_manifest = manifest
        st.session_state.upload_signature = sig
        clear_generated_zip()

    manifest = st.session_state.upload_manifest
    if not manifest:
        st.warning("No valid images are currently loaded.")
        return

    current_sig = current_processing_signature(manifest, general, scale)
    generated_zip = st.session_state.generated_zip
    if generated_zip and generated_zip["signature"] != current_sig:
        clear_generated_zip()
        generated_zip = None

    st.header("Preview")
    try:
        preview_img = load_image_from_path(manifest[0]["path"], manifest[0]["name"])
        if general["norm"]:
            preview_img = apply_normalization(preview_img, manifest[0]["norm_alpha"])
        preview_img = scale_for_preview(preview_img)
        st.image(process_image(preview_img, general)[0], caption="Preview + Outline")
    except ValueError as exc:
        st.error(str(exc))
        return

    if st.button("Process and Download Results"):
        try:
            zip_path, download_name = build_results_zip(manifest, general, scale)
        except ValueError as exc:
            clear_generated_zip()
            st.error(str(exc))
            return
        clear_generated_zip()
        st.session_state.generated_zip = {
            "path": zip_path,
            "download_name": download_name,
            "signature": current_sig,
        }
        generated_zip = st.session_state.generated_zip

    if generated_zip and generated_zip["signature"] == current_sig and os.path.exists(generated_zip["path"]):
        with open(generated_zip["path"], "rb") as fh:
            st.download_button(
                "Download Results",
                fh,
                file_name=generated_zip["download_name"],
                mime="application/zip",
            )


if __name__ == "__main__":
    main()
