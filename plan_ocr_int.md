Now I have the full picture. Here are the complete agent instructions:

---

## Agent Instructions: Add OCR to `app_coral_test.py`

```
You are working on the file `app_coral_test.py` in the ANPR Coral project.
The app already has 3 working models (vehicle detector, color classifier, brand detector).
Your job is to do three things:
  1. Install dependencies and download the OCR model (cct-xs-v1-global-model)
  2. The OCR is already wired into the app (load_ocr(), run calls, benchmark) — just make it work
  3. Add a test dataset selector in Tab 1 (Pipeline completo) and Tab 2 (Modelos individuales)
     so the user can pick from sample images instead of uploading one by one

─────────────────────────────────────────────────────────────────
STEP 1 — INSTALL DEPENDENCIES
─────────────────────────────────────────────────────────────────

Run this in the project root (same environment where streamlit runs):

    pip install "fast-plate-ocr[onnx]"

Then verify it works:

    python -c "from fast_plate_ocr import LicensePlateRecognizer; print('OK')"

Add to requirements.txt if it exists:

    fast-plate-ocr[onnx]

─────────────────────────────────────────────────────────────────
STEP 2 — DOWNLOAD MODEL WEIGHTS + CONFIG
─────────────────────────────────────────────────────────────────

Create a script called `scripts/download_ocr_model.py` with this content:

    import pathlib
    from fast_plate_ocr.inference.hub import download_model

    save_dir = pathlib.Path("models/ocr")
    model_path, config_path = download_model(
        "cct-xs-v1-global-model",
        save_directory=save_dir,
    )
    print(f"✅ Model:  {model_path}")
    print(f"✅ Config: {config_path}")

Then run it once:

    python scripts/download_ocr_model.py

This will download two files into models/ocr/:
  - cct_xs_v1_global.onnx
  - cct_xs_v1_global_plate_config.yaml

NOTE: The app's load_ocr() already calls:
    LicensePlateRecognizer("cct-xs-v1-global-model")
This auto-downloads to ~/.cache/fast-plate-ocr/ on first run if the
script above hasn't been run yet — so the download step is a fallback
safety net, not strictly required. But running it explicitly means
the model is available offline for the demo.

─────────────────────────────────────────────────────────────────
STEP 3 — ADD TEST DATASET SELECTOR
─────────────────────────────────────────────────────────────────

The goal: instead of uploading images one by one, the user can select
from a folder of sample images already in the repo.

3a. Create the sample images folder:

    mkdir -p test_images/

    Place at least 5-10 sample car images with visible plates in there.
    Name them descriptively: car_blue_toyota.jpg, moto_red.jpg, etc.
    Accepted formats: .jpg, .jpeg, .png

3b. Add this constant near the top of app_coral_test.py, right after the
    MODELS_CONFIG block (around line 153):

    # ─── Test dataset ─────────────────────────────────────────────────────────
    TEST_IMAGES_DIR = PROJECT_DIR / "test_images"

    def get_test_images():
        """Return sorted list of image paths in test_images/."""
        if not TEST_IMAGES_DIR.exists():
            return []
        return sorted([
            p for p in TEST_IMAGES_DIR.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

3c. MODIFY TAB 1 — PIPELINE COMPLETO

Find this block in Tab 1 (around line 306):

    col_upload, col_cam = st.columns([3, 1])
    with col_upload:
        uploaded = st.file_uploader("📁 Sube una imagen de vehículo con placa", type=["jpg", "jpeg", "png"])
    with col_cam:
        st.markdown("<br>", unsafe_allow_html=True)
        cam_img = st.camera_input("📷 O toma una foto")

    img_source = cam_img or uploaded

REPLACE it with:

    test_imgs = get_test_images()

    if test_imgs:
        src_mode = st.radio(
            "Fuente de imagen:",
            ["📂 Dataset de prueba", "📁 Subir imagen", "📷 Cámara"],
            horizontal=True,
            key="pipeline_src_mode",
        )
    else:
        src_mode = st.radio(
            "Fuente de imagen:",
            ["📁 Subir imagen", "📷 Cámara"],
            horizontal=True,
            key="pipeline_src_mode",
        )

    img_source = None
    frame_bgr = None

    if src_mode == "📂 Dataset de prueba" and test_imgs:
        names = [p.name for p in test_imgs]
        selected_name = st.selectbox(
            "Selecciona una imagen de prueba:",
            names,
            key="pipeline_test_select",
        )
        selected_path = TEST_IMAGES_DIR / selected_name
        col_prev, _ = st.columns([1, 2])
        with col_prev:
            st.image(str(selected_path), caption=selected_name, use_container_width=True)
        # Load directly into frame_bgr so the pipeline below works unchanged
        frame_bgr = cv2.imread(str(selected_path))

    elif src_mode == "📁 Subir imagen":
        uploaded = st.file_uploader(
            "📁 Sube una imagen de vehículo con placa",
            type=["jpg", "jpeg", "png"],
            key="pipeline_upload",
        )
        img_source = uploaded

    else:  # Cámara
        col_upload, col_cam = st.columns([3, 1])
        with col_cam:
            img_source = st.camera_input("📷 Toma una foto")

    # Decode uploaded/camera source into frame_bgr if not already set
    if img_source and frame_bgr is None:
        file_bytes = np.asarray(bytearray(img_source.read()), dtype=np.uint8)
        frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame_bgr is not None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        col_img, col_results = st.columns([1, 1])
        with col_img:
            st.image(frame_rgb, caption="Imagen de entrada", use_container_width=True)

    # NOTE: The rest of Tab 1 from `if img_source:` onward checks `frame_bgr is not None`
    # Replace `if img_source:` with `if frame_bgr is not None:` on that line.
    # Everything inside (the button, spinner, pipeline execution) stays exactly the same.

3d. MODIFY TAB 2 — MODELOS INDIVIDUALES

Find this block in Tab 2 (around line 491):

    uploaded2 = st.file_uploader("📁 Imagen de prueba", type=["jpg", "jpeg", "png"], key="ind_upload")

    if uploaded2:
        file_bytes2 = np.asarray(bytearray(uploaded2.read()), dtype=np.uint8)
        frame2_bgr = cv2.imdecode(file_bytes2, cv2.IMREAD_COLOR)
        frame2_rgb = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2RGB)
        st.image(frame2_rgb, caption="Imagen seleccionada", use_container_width=True)

REPLACE it with:

    test_imgs2 = get_test_images()
    frame2_bgr = None

    if test_imgs2:
        src_mode2 = st.radio(
            "Fuente de imagen:",
            ["📂 Dataset de prueba", "📁 Subir imagen"],
            horizontal=True,
            key="ind_src_mode",
        )
    else:
        src_mode2 = "📁 Subir imagen"

    if src_mode2 == "📂 Dataset de prueba" and test_imgs2:
        names2 = [p.name for p in test_imgs2]
        selected_name2 = st.selectbox(
            "Selecciona una imagen de prueba:",
            names2,
            key="ind_test_select",
        )
        selected_path2 = TEST_IMAGES_DIR / selected_name2
        st.image(str(selected_path2), caption=selected_name2, use_container_width=True)
        frame2_bgr = cv2.imread(str(selected_path2))

    else:
        uploaded2 = st.file_uploader(
            "📁 Imagen de prueba",
            type=["jpg", "jpeg", "png"],
            key="ind_upload",
        )
        if uploaded2:
            file_bytes2 = np.asarray(bytearray(uploaded2.read()), dtype=np.uint8)
            frame2_bgr = cv2.imdecode(file_bytes2, cv2.IMREAD_COLOR)

    if frame2_bgr is not None:
        frame2_rgb = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2RGB)
        st.image(frame2_rgb, caption="Imagen seleccionada", use_container_width=True)

    # NOTE: Replace `if uploaded2:` (the outer guard for the run button) with
    # `if frame2_bgr is not None:` — everything inside stays unchanged.

─────────────────────────────────────────────────────────────────
WHAT NOT TO TOUCH
─────────────────────────────────────────────────────────────────

- Do NOT modify load_ocr(), the OCR pipeline block in Tab 1 (lines 374-389),
  the OCR block in Tab 2 (lines 504-524), or the benchmark OCR logic.
  They are already correctly implemented.
- Do NOT change MODELS_CONFIG["ocr"] — it is intentionally file=None
  because OCR runs via ONNX, not TFLite.
- Do NOT change coral_simulator imports or CoralInterpreter usage.
- The sidebar model status badge for OCR (lines 243-246) already works correctly.

─────────────────────────────────────────────────────────────────
FINAL FOLDER STRUCTURE AFTER CHANGES
─────────────────────────────────────────────────────────────────

project/
├── app_coral_test.py              ← modified (dataset selector added)
├── test_images/                   ← NEW: put sample car+plate images here
│   ├── car_blue_toyota.jpg
│   ├── moto_red.jpg
│   └── ...
├── models/
│   ├── tflite_exports/            ← existing TFLite models (unchanged)
│   └── ocr/                       ← NEW: downloaded by download_ocr_model.py
│       ├── cct_xs_v1_global.onnx
│       └── cct_xs_v1_global_plate_config.yaml
├── scripts/
│   ├── download_ocr_model.py      ← NEW
│   └── ... (existing scripts unchanged)
└── requirements.txt               ← add fast-plate-ocr[onnx]
```