import streamlit as st
import cv2
import numpy as np
import os
import csv
import io
import zipfile
from datetime import datetime
from PIL import Image as PILImage
from streamlit_cropper import st_cropper

# --- Utility functions ---
# Load and preprocess images

def load_image_from_upload(uploaded_file):
    uploaded_file.seek(0)
    arr = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    uploaded_file.seek(0)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
    if k % 2 == 0: k += 1
    kernel = np.ones((k, k), np.uint8)
    background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    return cv2.absdiff(gray, background)

# Denoise and threshold helpers

def remove_noise(img, ksize):
    k = max(1, int(ksize))
    if k % 2 == 0: k += 1
    return cv2.medianBlur(img, k)

def threshold_image(img, thresh_val, invert):
    _, th = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_not(th) if invert else th

def close_holes(img, ksize):
    k = max(1, int(ksize))
    if k % 2 == 0: k += 1
    kernel = np.ones((k, k), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# Brightness normalization across images

def normalize_brightness(images):
    grays = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).mean() for img in images]
    global_mean = np.mean(grays)
    normed = []
    for img, m in zip(images, grays):
        alpha = global_mean / (m + 1e-6)
        normed.append(cv2.convertScaleAbs(img, alpha=alpha, beta=0))
    return normed

# --- Processing pipeline ---
def process_image(img, settings):
    h, w = img.shape[:2]
    cf = settings['zoom']; ch = int(h/cf); cw = int(w/cf)
    y1, x1 = (h-ch)//2, (w-cw)//2
    cropped = img[y1:y1+ch, x1:x1+cw]

    if settings['contrast'] != 1.0:
        cropped = cv2.convertScaleAbs(cropped, alpha=settings['contrast'], beta=0)
    if settings['circular_crop']:
        cropped = circular_crop(cropped)
    proc = subtract_background_auto(cropped, settings['bg_ks']) if settings['bg_sub'] else cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    if settings['noise'] > 0:
        proc = remove_noise(proc, settings['noise'])
    proc = threshold_image(proc, settings['th_val'], settings['invert'])
    if settings['hole_fill'] > 0:
        proc = close_holes(proc, settings['hole_fill'])
    contours, _ = cv2.findContours(proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if settings['single'] and contours:
        cx, cy = cw//2, ch//2; md=float('inf'); chosen=None
        for c in contours:
            M = cv2.moments(c)
            if M['m00']:
                cx0, cy0 = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                d = (cx0-cx)**2 + (cy0-cy)**2
                if d < md: md, chosen = d, c
        contours = [chosen] if chosen is not None else []
    area_px = sum(cv2.contourArea(c) for c in contours)
    overlay = img.copy(); roi = overlay[y1:y1+ch, x1:x1+cw]
    cv2.drawContours(roi, contours, -1, (0,255,0),2); overlay[y1:y1+ch, x1:x1+cw]=roi
    return overlay, area_px

# --- Streamlit App ---
def main():
    logo = PILImage.open("LogoSpaceMicrobesLab_White_with_background.png")

    # two columns, left for the title, right for the logo
    col1, col2 = st.columns([1, 8])
    col1.title("Orbis v2.0")
    col2.image(logo, width=80)

    st.title("Orbis v2.0")
    if 'scale_set' not in st.session_state:
        st.session_state.scale_set = False
        st.session_state.scale = {'factor':1.0,'unit':'pixels'}

    # Scale calibration
    scale_img = st.sidebar.file_uploader("Upload ruler image (optional)", type=['png','jpg','jpeg'])
    if scale_img and not st.session_state.scale_set:
        st.sidebar.subheader("Scale Calibration")
        pil = PILImage.fromarray(load_image_from_upload(scale_img))
        cropped = st_cropper(
            pil,
            realtime_update=True,
            box_color="#FF0000",
            aspect_ratio=None,
            return_type='image',
            key='scale_cropper'
        )
        if cropped:
            px_len = cropped.width
            length = st.sidebar.number_input("Real-world length",0.0,1e6,1.0)
            unit = st.sidebar.text_input("Unit label", 'cm')
            if st.sidebar.button("Confirm scale and continue"):
                if px_len>0 and length>0:
                    st.session_state.scale = {'factor':length/px_len,'unit':unit}
                st.session_state.scale_set = True
        return

    scale = st.session_state.scale
    st.sidebar.subheader("Analysis Configuration")
    st.sidebar.markdown("---")

    zoom = st.sidebar.slider("Zoom factor",1.0,5.0,1.0,0.1)
    contrast = st.sidebar.slider("Contrast",1.0,3.0,1.0,0.1)
    circular_crop_opt = st.sidebar.checkbox("Circular crop")
    single = st.sidebar.checkbox("Single colony")
    th_val = st.sidebar.slider("Threshold",0,255,127)

    advanced = st.sidebar.checkbox("Advanced mode")
    bg_sub = False; bg_ks = 0; invert = False; noise = 0; hole_fill = 0; norm = True
    if advanced:
        bg_sub = st.sidebar.checkbox("Subtract background [Experimental]")
        if bg_sub and st.session_state.scale_set:
            bg_ks = st.sidebar.slider("BG kernel size",3,101,15,2)
        invert = st.sidebar.checkbox("Invert threshold")
        noise = st.sidebar.slider("Noise removal",0,15,0)
        hole_fill = st.sidebar.slider("Hole fill",0,50,0)
        norm = st.sidebar.checkbox("Brightness normalization", value=True)

    general = {
        'zoom': zoom,
        'contrast': contrast,
        'circular_crop': circular_crop_opt,
        'bg_sub': bg_sub,
        'bg_ks': bg_ks,
        'th_val': th_val,
        'invert': invert,
        'noise': noise,
        'hole_fill': hole_fill,
        'single': single,
        'norm': norm
    }

    imgs = st.file_uploader("Upload images to analyze",accept_multiple_files=True,type=['png','jpg','jpeg'])
    if not imgs:
        return
    images = [load_image_from_upload(f) for f in imgs]
    if general['norm']:
        images = normalize_brightness(images)

    settings_list = []
    settings_list = [general] * len(images)
    st.header("Preview")
    st.image(process_image(images[0], general)[0],caption="Preview + Outline")

    if st.button("Process and Download Results"):
        # package results in-memory and offer as zip
        unit = st.session_state.scale['unit']
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            rows = []
            for img, f, s in zip(images, imgs, settings_list):
                ov, area_px = process_image(img, s)
                ext = os.path.splitext(f.name)[1]
                _, im_buf = cv2.imencode(ext, cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
                z.writestr(f.name, im_buf.tobytes())
                area = area_px * (st.session_state.scale['factor'] ** 2)
                rows.append((f.name, area))
            # write CSV
            csv_buf = io.StringIO()
            writer = csv.writer(csv_buf)
            writer.writerow(['filename', f"Area ({unit}^2)"])
            writer.writerows(rows)
            z.writestr(f"areas({unit}^2).csv", csv_buf.getvalue())
            # write log
            log = f"Date:{datetime.now()}\nMode:{mode}\nScale:{st.session_state.scale}\nSettings:{general}\n"
            z.writestr('log.txt', log)
        buf.seek(0)
        st.download_button("Download Results", buf, file_name=f"OrbisResults_{ts}.zip", mime="application/zip")

if __name__ == '__main__':
    main()