
import os
from datetime import datetime
import io
import numpy as np
import pandas as pd
import streamlit as st
import gdown
import tensorflow as tf
import rasterio
import matplotlib.pyplot as plt

from rasterio.io import MemoryFile
from pyproj import Transformer

# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

PATCH_SIZE = 256
THRESHOLD = 0.5

VV_MIN = -23.85825157
VV_MAX = -2.39960358
VH_MIN = -34.16553364
VH_MAX = -9.51392214

MODEL_PATHS = {
    "U-Net": os.path.join(MODEL_DIR, "best_unet.keras"),
    "U-Net++": os.path.join(MODEL_DIR, "best_unetpp.keras"),
    "Attention U-Net": os.path.join(
        MODEL_DIR, "best_attention_unet_40epochs.keras"
    )
}

MODEL_DESCRIPTIONS = {
    "U-Net": "A baseline encoder-decoder architecture with skip connections.",
    "U-Net++": "A nested architecture designed to improve feature fusion.",
    "Attention U-Net": "Uses attention gates to focus on relevant image regions."
}

METRICS_PATH = os.path.join(
    RESULTS_DIR, "metrics", "final_model_comparison.csv"
)


# =====================================================
# GOOGLE DRIVE DEPLOYMENT ASSETS
# =====================================================

ASSET_FILE_IDS = {
    "best_unet.keras": "1lo3RcM0x2ey0FMBIw5q1HCYH5mngptWB",
    "best_unetpp.keras": "1uPsMM7dcIOXCxPR4Q-eia7ROwtOvbJuX",
    "best_attention_unet_40epochs.keras": "1jgR3_Yx513942BSSdWaAHOshL56uBoHr",
    "final_model_comparison.csv": "17ySwfCYTzEayn4WjZ0Bf-_7nIpjwrTHS",
    "test_X.npy": "1p_UzUwnD_pDoA81LypsR9c8Xj-Tt9OM6",
    "test_Y.npy": "13BZHwVCnfWdfpAS-Hn6Cg3xF2c9Zs3Oe",
    "test_valid.npy": "1ueIH5pJJWBrBPQUpRMD1Rzqbj47Fi1sx"
}

def ensure_asset(file_path):
    """Download a deployment asset from public Google Drive if missing."""

    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        return file_path

    filename = os.path.basename(file_path)
    file_id = ASSET_FILE_IDS.get(filename)

    if not file_id:
        raise FileNotFoundError(
            f"No Google Drive file ID configured for {filename}"
        )

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    st.info(f"Downloading {filename} from Google Drive. Please wait...")

    downloaded = gdown.download(
        id=file_id,
        output=file_path,
        quiet=False
    )

    if not downloaded or not os.path.exists(file_path):
        raise RuntimeError(f"Download failed: {filename}")

    if os.path.getsize(file_path) == 0:
        raise RuntimeError(f"Downloaded file is empty: {filename}")

    return file_path


# =====================================================
# 2. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="FloodVision AI | Intelligent Flood Monitoring",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 3. CUSTOM DESIGN
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #07111f 0%, #0b1729 55%, #0a1220 100%);
    color: #e5edf8;
}

[data-testid="stHeader"] {
    background: rgba(7, 17, 31, 0.85);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1a2c 0%, #101d31 100%);
    border-right: 1px solid #203650;
}

[data-testid="stSidebar"] * {
    color: #dce8f7;
}

h1, h2, h3, h4 {
    color: #f2f7ff !important;
    font-weight: 700 !important;
}

p, label, li {
    color: #b8c8dc;
}

a {
    color: #38bdf8 !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

.hero {
    background: linear-gradient(115deg, #102b49, #123b5d 55%, #075985);
    padding: 32px;
    border-radius: 20px;
    border: 1px solid #25638b;
    margin-bottom: 25px;
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.20);
}

.hero h1 {
    font-size: 34px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 15px;
    color: #d5e8f7;
    margin-bottom: 0;
}

.eyebrow {
    color: #67e8f9;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.section-label {
    color: #67e8f9;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.metric-card {
    background: linear-gradient(145deg, #12243a, #101d30);
    border: 1px solid #263e5a;
    border-radius: 16px;
    padding: 21px;
    min-height: 135px;
    box-shadow: 0 7px 22px rgba(0, 0, 0, 0.12);
}

.metric-card:hover {
    border-color: #38bdf8;
}

.metric-label {
    color: #9db2ca;
    font-size: 13px;
    font-weight: 500;
    margin-bottom: 12px;
}

.metric-value {
    color: #67e8f9;
    font-size: 29px;
    font-weight: 800;
}

.metric-note {
    color: #91a8c2;
    font-size: 11px;
    margin-top: 7px;
}

.feature-card {
    background: #101f32;
    border: 1px solid #263d58;
    border-radius: 15px;
    padding: 20px;
    min-height: 165px;
}

.feature-card h3 {
    color: #67e8f9 !important;
    font-size: 18px;
}

.feature-card p {
    font-size: 13px;
    line-height: 1.7;
}

.best-card {
    background: linear-gradient(135deg, #10374a, #10283c);
    border: 1px solid #1d8ca4;
    border-radius: 16px;
    padding: 22px;
}

.best-card h3 {
    color: #67e8f9 !important;
}

.info-card {
    background: #101f32;
    border: 1px solid #263d58;
    border-radius: 14px;
    padding: 20px;
}

.stButton > button {
    background: linear-gradient(90deg, #0284c7, #0891b2);
    color: white;
    border: 1px solid #1599c7;
    border-radius: 10px;
    padding: 11px 20px;
    font-weight: 700;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #0369a1, #0e7490);
    border-color: #67e8f9;
    color: white;
}

.stDownloadButton > button {
    background: #142b42;
    color: #67e8f9;
    border: 1px solid #28506c;
    border-radius: 9px;
}

[data-testid="stFileUploader"] {
    background: #102238;
    border: 1px dashed #3185ad;
    border-radius: 14px;
    padding: 15px;
}

[data-testid="stMetric"] {
    background: #12243a;
    border: 1px solid #263e5a;
    padding: 16px;
    border-radius: 13px;
}

[data-testid="stMetricLabel"] {
    color: #a9bdd2;
}

[data-testid="stMetricValue"] {
    color: #67e8f9;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #0284c7, #67e8f9);
}

hr {
    border-color: #263d58;
}

[data-testid="stDataFrame"] {
    border: 1px solid #263d58;
    border-radius: 10px;
}

.sidebar-brand {
    text-align: center;
    padding: 15px 5px 22px 5px;
}

.sidebar-brand h2 {
    color: #67e8f9 !important;
    font-size: 23px;
}

.sidebar-brand p {
    font-size: 11px;
    color: #91a8c2;
}

.footer {
    text-align: center;
    color: #71869e;
    font-size: 11px;
    padding: 25px 0 5px 0;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 4. REUSABLE UI FUNCTIONS
# =====================================================

def hero(title, subtitle, eyebrow="FLOODVISION AI"):
    st.markdown(f"""
    <div class="hero">
        <div class="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def metric_card(label, value, note=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def feature_card(title, description):
    st.markdown(f"""
    <div class="feature-card">
        <h3>{title}</h3>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)


def section(title):
    st.markdown(
        f'<div class="section-label">{title}</div>',
        unsafe_allow_html=True
    )


def make_composite(image):
    return np.stack([
        image[:, :, 0],
        image[:, :, 1],
        (image[:, :, 0] + image[:, :, 1]) / 2
    ], axis=-1)


# =====================================================
# 5. MODEL LOADING
# =====================================================

@st.cache_resource
def load_model(model_path):
    ensure_asset(model_path)

    return tf.keras.models.load_model(
        model_path,
        compile=False
    )


# =====================================================
# 6. IMAGE NORMALIZATION
# =====================================================

def normalize_image(image):
    image = image.astype(np.float32)

    vv = np.clip(image[0], VV_MIN, VV_MAX)
    vh = np.clip(image[1], VH_MIN, VH_MAX)

    vv = (vv - VV_MIN) / (VV_MAX - VV_MIN)
    vh = (vh - VH_MIN) / (VH_MAX - VH_MIN)

    return np.stack([vv, vh], axis=-1).astype(np.float32)


# =====================================================
# 7. FLOOD PREDICTION
# =====================================================

def predict_flood(model, image):
    height, width, channels = image.shape

    if channels != 2:
        raise ValueError("Expected exactly two input channels.")

    padded_height = (
        (height + PATCH_SIZE - 1) // PATCH_SIZE
    ) * PATCH_SIZE

    padded_width = (
        (width + PATCH_SIZE - 1) // PATCH_SIZE
    ) * PATCH_SIZE

    padded = np.pad(
        image,
        (
            (0, padded_height - height),
            (0, padded_width - width),
            (0, 0)
        ),
        mode="edge"
    )

    output = np.zeros(
        (padded_height, padded_width),
        dtype=np.float32
    )

    for row in range(0, padded_height, PATCH_SIZE):
        for col in range(0, padded_width, PATCH_SIZE):

            patch = padded[
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE,
                :
            ]

            prediction = model.predict(
                np.expand_dims(patch, axis=0),
                verbose=0
            )

            output[
                row:row + PATCH_SIZE,
                col:col + PATCH_SIZE
            ] = prediction[0, :, :, 0]

    return output[:height, :width]


# =====================================================
# 8. FLOOD AREA ESTIMATION
# =====================================================

def calculate_area(mask, transform, crs):
    flood_pixels = int(np.count_nonzero(mask))

    if crs is None:
        return flood_pixels, None

    try:
        transformer = Transformer.from_crs(
            crs,
            "EPSG:6933",
            always_xy=True
        )

        row = mask.shape[0] // 2
        col = mask.shape[1] // 2

        corners = [
            transform * (col, row),
            transform * (col + 1, row),
            transform * (col + 1, row + 1),
            transform * (col, row + 1)
        ]

        projected = [
            transformer.transform(x, y)
            for x, y in corners
        ]

        x = np.array([p[0] for p in projected])
        y = np.array([p[1] for p in projected])

        pixel_area = abs(
            np.dot(x, np.roll(y, 1))
            - np.dot(y, np.roll(x, 1))
        ) / 2

        area = flood_pixels * pixel_area / 1_000_000

        return flood_pixels, area

    except Exception:
        return flood_pixels, None


# =====================================================
# 9. SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h2>🌊 FloodVision AI</h2>
        <p>INTELLIGENT FLOOD MONITORING</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
      "NAVIGATION",
      [
        "Dashboard",
        "Flood Detection",
        "Flood Visualization",
        "Geographical Flood Map",
        "Actual vs Predicted",
        "Model Comparison",
        "Model Architectures",
        "Flood Analytics",
        "About Project"
      ],
      label_visibility="visible"
    )

    st.divider()

    st.markdown("### 🛰️ System Information")
    st.caption("Dataset: Sen1Floods11")
    st.caption("Input: Sentinel-1 VV / VH")
    st.caption("Models: 3 Deep Learning Architectures")

    st.divider()

    st.markdown(
        '<div class="footer">FloodVision AI<br>B.Tech Capstone Project<br>2026</div>',
        unsafe_allow_html=True
    )


# =====================================================
# 10. DASHBOARD
# =====================================================

if page == "Dashboard":

    hero(
        "Intelligent Flood Monitoring",
        "Analyze satellite imagery and identify potential flood-affected regions using deep learning.",
        "AI-POWERED GEOSPATIAL ANALYSIS"
    )

    section("PROJECT OVERVIEW")
    st.subheader("Your flood intelligence at a glance")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Deep Learning Models", "03", "Segmentation architectures")

    with c2:
        metric_card("Training Images", "1,008", "Training dataset")

    with c3:
        metric_card("Validation Images", "356", "Validation dataset")

    with c4:
        metric_card("Test Images", "360", "Testing dataset")

    st.write("")
    section("BEST PERFORMING ARCHITECTURE")

    st.markdown("""
    <div class="best-card">
        <div class="eyebrow">TOP MODEL ON EVALUATION DATA</div>
        <h3>🏆 U-Net++</h3>
        <p>U-Net++ achieved the highest Dice Score and IoU among the three evaluated architectures.</p>
    </div>
    """, unsafe_allow_html=True)

    a, b, c = st.columns(3)

    with a:
        metric_card("Dice Score", "80.87%", "Highest among evaluated models")

    with b:
        metric_card("IoU Score", "67.88%", "Highest among evaluated models")

    with c:
        metric_card("Recall", "80.27%", "Highest among evaluated models")

    st.write("")
    section("AVAILABLE ARCHITECTURES")
    st.subheader("Three models. One flood intelligence platform.")

    cols = st.columns(3)

    for col, (name, description) in zip(
        cols, MODEL_DESCRIPTIONS.items()
    ):
        with col:
            feature_card(name, description)

    st.write("")
    section("SYSTEM CAPABILITIES")

    c1, c2 = st.columns(2)

    with c1:
        feature_card(
            "🛰️ Satellite Image Processing",
            "Process two-band Sentinel-1 VV and VH GeoTIFF imagery."
        )
        feature_card(
            "🧠 Deep Learning Segmentation",
            "Generate flood probability maps using three trained architectures."
        )
        feature_card(
            "📍 Geospatial Analysis",
            "Estimate flood pixel counts and approximate affected area."
        )

    with c2:
        feature_card(
            "📊 Model Comparison",
            "Compare Accuracy, Precision, Recall, F1, Dice and IoU."
        )
        feature_card(
            "🎨 Flood Visualization",
            "Inspect satellite composites, predicted masks and flood overlays."
        )
        feature_card(
            "📈 Flood Analytics",
            "Compare predicted coverage and estimated area across models."
        )


# =====================================================
# 11. FLOOD DETECTION
# =====================================================

elif page == "Flood Detection":

    hero(
        "Satellite Flood Detection",
        "Upload Sentinel-1 imagery and run your trained deep learning models.",
        "PREDICTION CENTER"
    )

    section("INPUT SATELLITE IMAGE")

    st.info(
        "Supported input: two-band GeoTIFF containing VV and VH polarization data."
    )

    uploaded_file = st.file_uploader(
        "Choose your Sentinel-1 GeoTIFF image",
        type=["tif", "tiff"],
        help="Upload a two-band VV/VH GeoTIFF file."
    )

    selected_models = st.multiselect(
        "Choose prediction models",
        list(MODEL_PATHS.keys()),
        default=list(MODEL_PATHS.keys())
    )

    if uploaded_file is not None:

        fingerprint = (
            uploaded_file.name,
            uploaded_file.size
        )

        if st.session_state.get("input_fingerprint") != fingerprint:
            st.session_state.pop("predictions", None)
            st.session_state["input_fingerprint"] = fingerprint

        try:
            with MemoryFile(uploaded_file.getvalue()) as memfile:
                with memfile.open() as src:
                    satellite = src.read()
                    transform = src.transform
                    crs = src.crs
                    image_name = uploaded_file.name

            if satellite.shape[0] != 2:
                st.error("This application requires exactly two bands (VV and VH).")
                st.stop()

            normalized_image = normalize_image(satellite)
            height, width = normalized_image.shape[:2]

            st.success("Satellite image loaded successfully.")

            c1, c2, c3 = st.columns(3)

            with c1:
                metric_card("Image Height", f"{height}px", "Input dimensions")

            with c2:
                metric_card("Image Width", f"{width}px", "Input dimensions")

            with c3:
                metric_card("Input Bands", "02", "VV + VH polarization")

            section("SATELLITE IMAGE PREVIEW")

            fig, ax = plt.subplots(figsize=(11, 5))
            ax.imshow(make_composite(normalized_image))
            ax.set_title("Sentinel-1 False-Color Composite", color="white")
            ax.axis("off")
            fig.patch.set_facecolor("#101f32")
            ax.set_facecolor("#101f32")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.write("")

            if st.button(
                "🚀 RUN FLOOD DETECTION",
                use_container_width=True
            ):

                if not selected_models:
                    st.warning("Please select at least one model.")
                    st.stop()

                predictions = {}
                progress = st.progress(0)
                status = st.empty()

                try:
                    for index, name in enumerate(selected_models):

                        status.info(f"Processing with {name}...")

                        model_path = MODEL_PATHS[name]

                        if not os.path.exists(model_path):
                            raise FileNotFoundError(
                                f"Model file not found: {model_path}"
                            )

                        model = load_model(model_path)

                        probability = predict_flood(
                            model, normalized_image
                        )

                        binary_mask = (
                            probability >= THRESHOLD
                        ).astype(np.uint8)

                        pixels, area = calculate_area(
                            binary_mask, transform, crs
                        )

                        predictions[name] = {
                            "probability": probability,
                            "mask": binary_mask,
                            "coverage": float(binary_mask.mean() * 100),
                            "pixels": pixels,
                            "area": area
                        }

                        progress.progress(
                            (index + 1) / len(selected_models)
                        )

                    st.session_state["predictions"] = predictions
                    st.session_state["normalized_image"] = normalized_image
                    st.session_state["image_name"] = image_name
                    st.session_state["geo_transform"] = transform
                    st.session_state["geo_crs"] = crs
                    st.session_state["geo_height"] = height
                    st.session_state["geo_width"] = width


                    status.success(
                        "All selected models completed their predictions!"
                    )

                except Exception as error:
                    status.empty()
                    st.error(f"Prediction error: {error}")

        except Exception as error:
            st.error(f"Unable to read the uploaded image: {error}")

    if "predictions" in st.session_state:

        st.divider()
        section("PREDICTION RESULTS")
        st.subheader("Flood Detection Summary")

        st.caption(
            f"Analyzed image: {st.session_state.get('image_name', 'Uploaded image')}"
        )

        predictions = st.session_state["predictions"]

        cols = st.columns(len(predictions))

        for col, (name, result) in zip(cols, predictions.items()):

            with col:
                st.markdown(
                    f'<div class="info-card"><h3>{name}</h3></div>',
                    unsafe_allow_html=True
                )

                st.metric(
                    "Flood Coverage",
                    f"{result['coverage']:.2f}%"
                )

                st.metric(
                    "Flood Pixels",
                    f"{result['pixels']:,}"
                )

                if result["area"] is not None:
                    st.metric(
                        "Estimated Area",
                        f"{result['area']:.4f} km²"
                    )
                else:
                    st.metric("Estimated Area", "Unavailable")

                st.progress(
                    min(result["coverage"] / 100, 1.0)
                )


# =====================================================

    # =============================================
    # FINAL OUTCOME REPORT - UPLOADED IMAGE
    # =============================================

    st.divider()
    section("FINAL OUTCOME REPORT")

    st.write(
        "Download a detailed report containing the uploaded "
        "satellite image, model predictions and flood statistics."
    )

    if "predictions" in st.session_state:

        outcome_predictions = st.session_state["predictions"]
        outcome_image = st.session_state["normalized_image"]
        outcome_name = st.session_state.get(
            "image_name", "Uploaded Sentinel-1 Image"
        )

        outcome_height, outcome_width = outcome_image.shape[:2]

        outcome_rows = []

        for model_name, result in outcome_predictions.items():
            outcome_rows.append({
                "Model": model_name,
                "Flood Coverage (%)": round(result["coverage"], 2),
                "Flood Pixels": result["pixels"],
                "Estimated Area (km²)": (
                    round(result["area"], 4)
                    if result["area"] is not None
                    else "Unavailable"
                )
            })

        outcome_df = pd.DataFrame(outcome_rows)

        outcome_buffer = io.BytesIO()

        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(outcome_buffer) as pdf:

            # -------------------------------------
            # PAGE 1: UPLOADED IMAGE SUMMARY
            # -------------------------------------

            fig, ax = plt.subplots(figsize=(11.7, 8.3))
            ax.axis("off")

            fig.suptitle(
                "FLOODVISION AI - FINAL OUTCOME REPORT",
                fontsize=19,
                fontweight="bold",
                y=0.94
            )

            summary = (
                f"Uploaded Image: {outcome_name}\n\n"
                f"Image Dimensions: {outcome_width} x "
                f"{outcome_height} pixels\n\n"
                f"Input Bands: Sentinel-1 VV and VH\n\n"
                f"Models Executed: {len(outcome_predictions)}\n\n"
                f"Prediction Threshold: {THRESHOLD}\n\n"
                f"Coordinate Reference System: "
                f"{st.session_state.get('geo_crs') or 'Unavailable'}\n\n"
                f"Report Generated: "
                f"{datetime.now().strftime('%d-%m-%Y %H:%M')}\n\n"
                "This report summarizes AI-based flood segmentation "
                "predictions for the uploaded satellite image. It presents "
                "the estimated flood coverage, predicted flood pixels, and "
                "estimated flooded area for each model. These results do "
                "not represent ground-truth accuracy measurements."
            )

            ax.text(
                0.08, 0.80,
                summary,
                fontsize=12,
                va="top",
                linespacing=1.8,
                transform=ax.transAxes
            )

            ax.text(
                0.08, 0.29,
                "FLOOD DETECTION RESULTS",
                fontsize=14,
                fontweight="bold",
                transform=ax.transAxes
            )

            table = ax.table(
                cellText=outcome_df.values,
                colLabels=outcome_df.columns,
                cellLoc="center",
                loc="center",
                bbox=[0.05, 0.08, 0.90, 0.18]
            )

            table.auto_set_font_size(False)
            table.set_fontsize(8)

            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # -------------------------------------
            # PAGE 2: SATELLITE IMAGE
            # -------------------------------------

            fig, ax = plt.subplots(figsize=(11.7, 8.3))

            ax.imshow(make_composite(outcome_image))
            ax.set_title(
                "Uploaded Sentinel-1 Satellite Image",
                fontsize=17,
                fontweight="bold"
            )
            ax.axis("off")

            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # -------------------------------------
            # PAGE 3: MODEL PREDICTION MASKS
            # -------------------------------------

            model_count = len(outcome_predictions)

            fig, axes = plt.subplots(
                1, model_count,
                figsize=(5 * model_count, 5)
            )

            if model_count == 1:
                axes = [axes]

            fig.suptitle(
                "Flood Segmentation Predictions",
                fontsize=17,
                fontweight="bold"
            )

            for ax, (model_name, result) in zip(
                axes, outcome_predictions.items()
            ):
                ax.imshow(
                    result["mask"].squeeze() * 255,
                    cmap="gray",
                    vmin=0,
                    vmax=255
                )
                ax.set_title(model_name, fontsize=12)
                ax.axis("off")

            fig.tight_layout(rect=[0, 0, 1, 0.90])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # -------------------------------------
            # PAGE 4: FLOOD COVERAGE COMPARISON
            # -------------------------------------

            fig, ax = plt.subplots(figsize=(11.7, 8.3))

            names = [
                row["Model"] for row in outcome_rows
            ]

            coverages = [
                row["Flood Coverage (%)"] for row in outcome_rows
            ]

            bars = ax.bar(names, coverages)

            ax.bar_label(
                bars,
                labels=[f"{v:.2f}%" for v in coverages],
                padding=4
            )

            ax.set_title(
                "Flood Coverage by Model",
                fontsize=18,
                fontweight="bold"
            )

            ax.set_ylabel("Flood Coverage (%)")
            ax.set_ylim(0, max(max(coverages) * 1.25, 1))
            ax.grid(axis="y", alpha=0.25)

            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        outcome_buffer.seek(0)

        st.download_button(
            label="📄 Download Final Outcome Report",
            data=outcome_buffer.getvalue(),
            file_name="FloodVision_AI_Final_Outcome_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    else:
        st.info(
            "Upload a Sentinel-1 image and run flood detection "
            "to generate its Final Outcome Report."
        )


# 12. FLOOD VISUALIZATION
# =====================================================



elif page == "Geographical Flood Map":

    import folium
    from streamlit_folium import st_folium
    from rasterio.transform import array_bounds
    from rasterio.warp import transform_bounds
    from folium.raster_layers import ImageOverlay

    hero(
        "Interactive Geographical Flood Map",
        "Explore satellite imagery and predicted flood regions on an interactive map.",
        "GEO-SPATIAL FLOOD INTELLIGENCE"
    )

    if "predictions" not in st.session_state:
        st.info(
            "No predictions available. Go to Flood Detection, upload a "
            "GeoTIFF and run a model first."
        )

    elif (
        st.session_state.get("geo_transform") is None
        or st.session_state.get("geo_crs") is None
    ):
        st.warning(
            "The uploaded GeoTIFF does not contain valid geographic metadata."
        )

    else:
        try:
            predictions = st.session_state["predictions"]
            image = st.session_state["normalized_image"]
            transform = st.session_state["geo_transform"]
            crs = st.session_state["geo_crs"]

            height, width = image.shape[:2]

            # Folium image overlays use north-up rectangular bounds.
            if abs(transform.b) > 1e-10 or abs(transform.d) > 1e-10:
                raise ValueError(
                    "This GeoTIFF uses a rotated or skewed grid. "
                    "A simple rectangular map overlay could be inaccurate."
                )

            bounds = array_bounds(height, width, transform)

            west, south, east, north = transform_bounds(
                crs,
                "EPSG:4326",
                *bounds,
                densify_pts=21
            )

            if not all(
                np.isfinite(v) for v in [west, south, east, north]
            ):
                raise ValueError("Invalid geographic coordinates.")

            if (
                west < -180 or east > 180
                or south < -90 or north > 90
                or west >= east or south >= north
            ):
                raise ValueError("Geographic bounds are outside valid limits.")

            center_lat = (south + north) / 2
            center_lon = (west + east) / 2

            st.subheader("Satellite Image Location")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("West Longitude", f"{west:.5f}°")
            c2.metric("East Longitude", f"{east:.5f}°")
            c3.metric("South Latitude", f"{south:.5f}°")
            c4.metric("North Latitude", f"{north:.5f}°")

            st.caption(
                f"Source CRS: {crs} | Display CRS: EPSG:4326 | "
                f"Image: {st.session_state.get('image_name', 'Satellite image')}"
            )

            st.divider()
            st.subheader("Interactive Flood Map")

            model_names = list(predictions.keys())

            selected_model = st.selectbox(
                "Select model prediction",
                model_names,
                key="geo_map_selected_model"
            )

            result = predictions[selected_model]
            mask = result["mask"]

            # Create the interactive OpenStreetMap basemap.
            flood_map = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=12,
                tiles="OpenStreetMap",
                control_scale=True
            )

            image_bounds = [
                [south, west],
                [north, east]
            ]

            # Satellite composite as an optional map layer.
            composite = make_composite(image)
            composite = np.asarray(composite)

            if composite.dtype.kind == "f":
                if composite.max() <= 1.0:
                    composite = composite * 255

            composite = np.clip(composite, 0, 255).astype(np.uint8)

            if composite.shape[-1] == 4:
                composite = composite[:, :, :3]

            satellite_layer = folium.FeatureGroup(
                name="Sentinel-1 Satellite Composite",
                show=False
            )

            ImageOverlay(
                image=composite,
                bounds=image_bounds,
                opacity=0.75,
                name="Satellite Composite",
                interactive=False,
                cross_origin=False,
                zindex=1
            ).add_to(satellite_layer)

            satellite_layer.add_to(flood_map)

            # Transparent red pixels show predicted flood regions.
            flood_rgba = np.zeros(
                (height, width, 4),
                dtype=np.uint8
            )

            flood_rgba[mask > 0] = [255, 30, 30, 175]

            flood_layer = folium.FeatureGroup(
                name=f"{selected_model} Flood Prediction",
                show=True
            )

            ImageOverlay(
                image=flood_rgba,
                bounds=image_bounds,
                opacity=1.0,
                name="Predicted Flood Regions",
                interactive=False,
                cross_origin=False,
                zindex=2
            ).add_to(flood_layer)

            flood_layer.add_to(flood_map)

            # Mark the approximate center of the image footprint.
            folium.Marker(
                [center_lat, center_lon],
                tooltip="Satellite Image Center",
                popup=(
                    f"Latitude: {center_lat:.6f}<br>"
                    f"Longitude: {center_lon:.6f}"
                )
            ).add_to(flood_map)

            folium.LayerControl(collapsed=False).add_to(flood_map)

            flood_map.fit_bounds(image_bounds)

            st_folium(
                flood_map,
                width=1100,
                height=650,
                returned_objects=[]
            )

            st.caption(
                "Use the layer control to switch the satellite composite "
                "and flood prediction overlays on or off."
            )

            m1, m2, m3 = st.columns(3)

            m1.metric(
                "Selected Model",
                selected_model
            )

            m2.metric(
                "Flood Coverage",
                f"{result['coverage']:.2f}%"
            )

            m3.metric(
                "Predicted Flood Pixels",
                f"{result['pixels']:,}"
            )

            st.info(
                "The red overlay represents the model's predicted flood mask. "
                "OpenStreetMap is the background map; it does not independently "
                "verify the predicted flood regions."
            )

        except Exception as error:
            st.error(f"Unable to generate interactive map: {error}")

elif page == "Flood Visualization":

    hero(
        "Flood Segmentation Visualization",
        "Explore predicted flood regions through satellite imagery, segmentation masks and overlays.",
        "VISUAL INTELLIGENCE"
    )

    if "predictions" not in st.session_state:
        st.info(
            "No predictions available. Go to Flood Detection and run a prediction first."
        )

    else:

        predictions = st.session_state["predictions"]
        image = st.session_state["normalized_image"]
        composite = make_composite(image)

        section("PREDICTION VISUALIZATIONS")

        for name, result in predictions.items():

            st.subheader(name)

            mask = result["mask"]

            fig, axes = plt.subplots(1, 3, figsize=(16, 5))

            axes[0].imshow(composite)
            axes[0].set_title("Original Satellite Image", color="white")

            axes[1].imshow(mask, cmap="Blues", vmin=0, vmax=1)
            axes[1].set_title("Predicted Flood Mask", color="white")

            axes[2].imshow(composite)
            axes[2].imshow(
                mask,
                cmap="Reds",
                alpha=0.5,
                vmin=0,
                vmax=1
            )
            axes[2].set_title("Flood Region Overlay", color="white")

            for ax in axes:
                ax.axis("off")
                ax.set_facecolor("#101f32")

            fig.patch.set_facecolor("#101f32")
            plt.tight_layout()

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            c1, c2 = st.columns(2)

            with c1:
                st.metric(
                    "Flood Coverage",
                    f"{result['coverage']:.2f}%"
                )

            with c2:
                st.metric(
                    "Flood Pixels",
                    f"{result['pixels']:,}"
                )

            st.divider()


# =====================================================
# 13. MODEL COMPARISON
# =====================================================

elif page == "Model Comparison":

    hero(
        "Model Performance Comparison",
        "Evaluate and compare the segmentation performance of all three deep learning architectures.",
        "MODEL EVALUATION"
    )

    try:
        ensure_asset(METRICS_PATH)
    except Exception as e:
        st.error(f"Unable to download model metrics: {e}")
        st.stop()

    if os.path.exists(METRICS_PATH):

        df = pd.read_csv(METRICS_PATH)

        required = [
            "Model", "Accuracy", "Precision", "Recall",
            "F1_Score", "Dice", "IoU"
        ]

        if not all(col in df.columns for col in required):
            st.error("The metrics CSV is missing one or more required columns.")
            st.stop()

        metric_cols = [
            "Accuracy", "Precision", "Recall",
            "F1_Score", "Dice", "IoU"
        ]

        # Support both decimal values and percentage values.
        if df[metric_cols].max().max() > 1:
            df[metric_cols] = df[metric_cols] / 100

        section("EVALUATION RESULTS")

        display_df = df.copy()
        display_df[metric_cols] = display_df[metric_cols] * 100

        display_df = display_df.rename(columns={
            "F1_Score": "F1 Score",
            "Dice": "Dice Score",
            "IoU": "IoU Score"
        })

        st.dataframe(
            display_df.style.format({
                col: "{:.2f}%" for col in [
                    "Accuracy", "Precision", "Recall",
                    "F1 Score", "Dice Score", "IoU Score"
                ]
            }),
            use_container_width=True,
            hide_index=True
        )

        section("BEST MODEL")

        best_idx = df["Dice"].idxmax()
        best_model = df.loc[best_idx, "Model"]

        st.markdown(f"""
        <div class="best-card">
            <div class="eyebrow">HIGHEST DICE SCORE</div>
            <h3>🏆 {best_model}</h3>
            <p>Best Dice score: {df.loc[best_idx, 'Dice'] * 100:.2f}%</p>
            <p>IoU score: {df.loc[best_idx, 'IoU'] * 100:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        section("INTERACTIVE METRIC ANALYSIS")

        metric = st.selectbox(
            "Choose a metric to compare",
            metric_cols,
            format_func=lambda x: {
                "F1_Score": "F1 Score",
                "Dice": "Dice Score",
                "IoU": "IoU Score"
            }.get(x, x)
        )

        chart_df = df.set_index("Model")[[metric]] * 100

        st.bar_chart(
            chart_df,
            color="#22B8E6",
            y_label="Score (%)"
        )

        st.caption(
            "All metrics are based on the saved final evaluation results."
        )

    else:
        st.warning("Model comparison metrics file was not found.")


# =====================================================
# 14. FLOOD ANALYTICS
# =====================================================


# =====================================================
# 14. MODEL ARCHITECTURES
# =====================================================

elif page == "Model Architectures":

    hero(
        "Deep Learning Model Architectures",
        "Understand the three segmentation architectures used in FloodVision AI.",
        "MODEL ARCHITECTURE CENTER"
    )

    st.markdown("""
    FloodVision AI uses three deep learning architectures to identify
    potential flood regions from Sentinel-1 satellite imagery.

    Each model receives a two-channel VV/VH input image and produces
    a pixel-level flood segmentation prediction.
    """)

    st.divider()


    def draw_architecture_diagram(title, nodes, edges):
        from matplotlib.patches import FancyBboxPatch

        fig, ax = plt.subplots(figsize=(11, 7))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 7)
        ax.axis("off")
        ax.set_title(title, fontsize=17, fontweight="bold", pad=18)

        colors = {
            "input": "#DCEBFF",
            "encoder": "#B9D7FF",
            "bottleneck": "#D8C7FF",
            "decoder": "#BDEBD5",
            "attention": "#FFE3A3",
            "output": "#A7E8E0",
            "nested": "#D6E8FF"
        }

        # Draw connections behind the nodes
        for source, target, connection_type, curve in edges:
            x1, y1 = nodes[source][1:3]
            x2, y2 = nodes[target][1:3]

            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "#64748B" if connection_type == "normal" else "#D97706",
                    "linewidth": 1.8,
                    "linestyle": "-" if connection_type == "normal" else "--",
                    "connectionstyle": f"arc3,rad={curve}",
                    "shrinkA": 27,
                    "shrinkB": 27
                }
            )

        # Draw architecture blocks
        for label, x, y, kind in nodes:
            box = FancyBboxPatch(
                (x - 1.0, y - 0.36),
                2.0, 0.72,
                boxstyle="round,pad=0.08,rounding_size=0.12",
                facecolor=colors[kind],
                edgecolor="#334155",
                linewidth=1.4
            )
            ax.add_patch(box)
            ax.text(
                x, y, label,
                ha="center", va="center",
                fontsize=9, fontweight="bold",
                color="#172033"
            )

        ax.text(
            5, 0.05,
            "Solid arrows: feature flow     Dashed arrows: skip / attention connections",
            ha="center", fontsize=9, color="#64748B"
        )

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # U-Net visual architecture

    st.subheader("1. U-Net")

    st.markdown("""
    **Overview**

    U-Net is an encoder-decoder convolutional neural network.
    The encoder extracts image features, while the decoder reconstructs
    a segmentation mask. Skip connections transfer spatial information
    from encoder layers to corresponding decoder layers.

    **Conceptual architecture**
    """)

    draw_architecture_diagram(
        "U-Net Architecture",
        [
            ("VV/VH Input", 5, 0.8, "input"),
            ("Encoder 1", 2, 2.0, "encoder"),
            ("Encoder 2", 2, 3.6, "encoder"),
            ("Bottleneck", 5, 5.0, "bottleneck"),
            ("Decoder 2", 8, 3.6, "decoder"),
            ("Decoder 1", 8, 2.0, "decoder"),
            ("Flood Mask", 5, 6.3, "output")
        ],
        [
            (0, 1, "normal", 0),
            (1, 2, "normal", 0),
            (2, 3, "normal", 0),
            (3, 4, "normal", 0),
            (4, 5, "normal", 0),
            (5, 6, "normal", 0),
            (1, 5, "skip", -0.28),
            (2, 4, "skip", -0.22)
        ]
    )

    with st.expander("U-Net: Key characteristics"):
        st.markdown("""
        - Encoder-decoder structure.
        - Skip connections preserve spatial details.
        - Suitable for pixel-level image segmentation.
        - Provides the baseline architecture for comparison.
        """)

    st.divider()

    # U-NET++
    st.subheader("2. U-Net++")

    st.markdown("""
    **Overview**

    U-Net++ extends the original U-Net using nested and redesigned
    skip connections. These connections create pathways that help
    reduce the semantic gap between encoder and decoder features.

    **Conceptual architecture**
    """)

    draw_architecture_diagram(
        "U-Net++ Architecture",
        [
            ("VV/VH Input", 5, 0.6, "input"),
            ("Encoder 1", 1.5, 2.0, "encoder"),
            ("Encoder 2", 1.5, 3.6, "encoder"),
            ("Encoder 3", 1.5, 5.2, "encoder"),
            ("Nested 1", 4.2, 2.0, "nested"),
            ("Nested 2", 4.2, 3.6, "nested"),
            ("Nested 3", 4.2, 5.2, "nested"),
            ("Decoder 2", 7.2, 3.6, "decoder"),
            ("Decoder 1", 7.2, 2.0, "decoder"),
            ("Flood Mask", 7.2, 5.8, "output")
        ],
        [
            (0, 1, "normal", 0),
            (1, 2, "normal", 0),
            (2, 3, "normal", 0),
            (1, 4, "skip", 0),
            (2, 4, "skip", -0.15),
            (2, 5, "skip", 0),
            (3, 5, "skip", -0.15),
            (3, 6, "skip", 0),
            (4, 5, "normal", 0),
            (5, 6, "normal", 0),
            (4, 8, "skip", -0.2),
            (5, 7, "skip", -0.15),
            (6, 7, "normal", 0),
            (7, 8, "normal", 0),
            (8, 9, "normal", 0)
        ]
    )

    with st.expander("U-Net++: Key characteristics"):
        st.markdown("""
        - Nested skip connections.
        - More extensive feature fusion.
        - Designed to improve information flow between encoder
          and decoder stages.
        - More complex than the baseline U-Net.
        """)

    st.divider()

    # ATTENTION U-NET
    st.subheader("3. Attention U-Net")

    st.markdown("""
    **Overview**

    Attention U-Net introduces attention gates into the skip
    connections. These gates help the network emphasize relevant
    image regions and suppress less useful features.

    **Conceptual architecture**
    """)

    draw_architecture_diagram(
        "Attention U-Net Architecture",
        [
            ("VV/VH Input", 5, 0.6, "input"),
            ("Encoder 1", 2, 2.0, "encoder"),
            ("Encoder 2", 2, 3.8, "encoder"),
            ("Bottleneck", 5, 5.5, "bottleneck"),
            ("Attention Gate 2", 5, 3.8, "attention"),
            ("Decoder 2", 8, 3.8, "decoder"),
            ("Attention Gate 1", 5, 2.0, "attention"),
            ("Decoder 1", 8, 2.0, "decoder"),
            ("Flood Mask", 5, 6.5, "output")
        ],
        [
            (0, 1, "normal", 0),
            (1, 2, "normal", 0),
            (2, 3, "normal", 0),
            (3, 5, "normal", 0),
            (5, 7, "normal", 0),
            (7, 8, "normal", 0),
            (2, 4, "skip", 0),
            (4, 5, "skip", 0),
            (1, 6, "skip", 0),
            (6, 7, "skip", 0)
        ]
    )

    with st.expander("Attention U-Net: Key characteristics"):
        st.markdown("""
        - Uses attention gates.
        - Highlights relevant spatial features.
        - Filters less relevant information in skip pathways.
        - Designed to improve focus on target regions.
        """)

    st.divider()

    # COMPARISON
    st.subheader("Architecture Comparison")

    architecture_df = pd.DataFrame([
        {
            "Architecture": "U-Net",
            "Main Feature": "Standard skip connections",
            "Design": "Encoder-decoder",
            "Role": "Baseline segmentation model"
        },
        {
            "Architecture": "U-Net++",
            "Main Feature": "Nested skip connections",
            "Design": "Nested encoder-decoder",
            "Role": "Enhanced feature fusion"
        },
        {
            "Architecture": "Attention U-Net",
            "Main Feature": "Attention gates",
            "Design": "Attention-based encoder-decoder",
            "Role": "Focus on relevant image regions"
        }
    ])

    st.dataframe(
        architecture_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("How These Models Work in FloodVision AI")

    st.markdown("""
    1. **Input:** A Sentinel-1 GeoTIFF containing VV and VH bands.
    2. **Preprocessing:** The input bands are normalized using the
       project's configured normalization ranges.
    3. **Segmentation:** Each trained model predicts flood probabilities
       for individual pixels.
    4. **Thresholding:** The probability map is converted into a binary
       flood mask using the configured threshold.
    5. **Analysis:** Flood pixels, coverage, and estimated area are
       calculated and displayed.
    """)

    st.info(
        "These diagrams describe the conceptual architecture of each "
        "model. They are not exact layer-by-layer diagrams of the saved "
        "Keras model implementations."
    )

    st.caption(
        "FloodVision AI | U-Net • U-Net++ • Attention U-Net"
    )


elif page == "Flood Analytics":

    hero(
        "Flood Analytics Center",
        "Understand predicted flood coverage, affected pixels and estimated area.",
        "PREDICTION ANALYTICS"
    )

    if "predictions" not in st.session_state:

        st.info(
            "Run flood detection first to generate analytics."
        )

    else:

        predictions = st.session_state["predictions"]

        analytics = []

        for name, result in predictions.items():

            analytics.append({
                "Model": name,
                "Flood Coverage (%)": result["coverage"],
                "Flood Pixels": result["pixels"],
                "Area (km²)": (
                    result["area"]
                    if result["area"] is not None
                    else np.nan
                )
            })

        df = pd.DataFrame(analytics)

        section("FLOOD IMPACT SUMMARY")

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card(
                "Models Analyzed",
                str(len(df)),
                "Selected architectures"
            )

        with c2:
            metric_card(
                "Highest Coverage",
                f"{df['Flood Coverage (%)'].max():.2f}%",
                "Among selected models"
            )

        with c3:
            metric_card(
                "Lowest Coverage",
                f"{df['Flood Coverage (%)'].min():.2f}%",
                "Among selected models"
            )

        st.write("")
        section("DETAILED ANALYTICS")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Flood Coverage Comparison")

        coverage_df = df.set_index("Model")[["Flood Coverage (%)"]]

        st.bar_chart(
            coverage_df,
            color="#22B8E6",
            y_label="Coverage (%)"
        )

        if len(df) > 1:

            difference = (
                df["Flood Coverage (%)"].max()
                - df["Flood Coverage (%)"].min()
            )

            st.metric(
                "Difference between highest and lowest coverage",
                f"{difference:.2f} percentage points"
            )

        st.caption(
            "Flood coverage values are model predictions, not ground-truth measurements."
        )


# =====================================================
# 15. ABOUT PROJECT
# =====================================================

elif page == "About Project":

    hero(
        "About FloodVision AI",
        "An AI-powered satellite image segmentation platform for identifying potential flood-affected regions.",
        "PROJECT INFORMATION"
    )

    section("PROJECT INTRODUCTION")

    st.markdown("""
    FloodVision AI is a deep-learning-based flood detection
    and satellite image segmentation system.

    The application analyzes Sentinel-1 satellite imagery
    using three segmentation architectures to identify
    potential flood-affected regions.
    """)

    st.divider()

    section("DATASET")

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("Training Images", "1,008", "Training split")

    with c2:
        metric_card("Validation Images", "356", "Validation split")

    with c3:
        metric_card("Testing Images", "360", "Testing split")

    st.markdown("""
    **Dataset name:** Sen1Floods11

    **Satellite:** Sentinel-1

    **Input channels:** VV and VH polarization bands

    **Labels:** Ground-truth flood segmentation masks
    """)

    st.divider()

    section("DEEP LEARNING ARCHITECTURES")

    c1, c2, c3 = st.columns(3)

    with c1:
        feature_card(
            "U-Net",
            "An encoder-decoder convolutional neural network using skip connections to preserve spatial information."
        )

    with c2:
        feature_card(
            "U-Net++",
            "An enhanced U-Net architecture with nested skip pathways and improved feature fusion."
        )

    with c3:
        feature_card(
            "Attention U-Net",
            "An attention-based architecture that emphasizes relevant image features."
        )

    st.divider()

    section("EVALUATION METRICS")

    st.markdown("""
    - **Accuracy:** Overall proportion of correctly classified pixels.
    - **Precision:** Proportion of predicted flood pixels that are correct.
    - **Recall:** Proportion of actual flood pixels detected.
    - **F1 Score:** Harmonic mean of precision and recall.
    - **Dice Score:** Measures overlap between prediction and ground truth.
    - **IoU:** Intersection over Union of predicted and actual regions.
    """)

    st.divider()

    section("TECHNOLOGY STACK")

    st.markdown("""
    | Component | Technology |
    |---|---|
    | Programming Language | Python |
    | Deep Learning | TensorFlow / Keras |
    | Web Application | Streamlit |
    | Image Processing | NumPy, Rasterio |
    | Data Analysis | Pandas |
    | Visualization | Matplotlib |
    | Geospatial Processing | PyProj |
    """)

    st.divider()

    section("PROJECT OBJECTIVE")

    st.markdown("""
    The primary objective is to develop an accessible
    flood-monitoring application that processes satellite
    imagery, generates segmentation masks, estimates
    potential flood coverage and compares multiple
    deep learning architectures.
    """)



# =====================================================
# 16. ACTUAL VS PREDICTED FLOOD COMPARISON
# =====================================================

elif page == "Actual vs Predicted":

    hero(
        "Actual vs Predicted Flood Comparison",
        "Compare ground-truth flood masks with predictions from all three trained models.",
        "MODEL VALIDATION"
    )

    @st.cache_resource
    def load_comparison_data():

        data_dir = os.path.join(
            PROJECT_DIR, "dataset", "processed"
        )

        ensure_asset(os.path.join(data_dir, "test_X.npy"))
        ensure_asset(os.path.join(data_dir, "test_Y.npy"))
        ensure_asset(os.path.join(data_dir, "test_valid.npy"))

        X_test = np.load(
            os.path.join(data_dir, "test_X.npy")
        )

        Y_test = np.load(
            os.path.join(data_dir, "test_Y.npy")
        )

        valid_test = np.load(
            os.path.join(data_dir, "test_valid.npy")
        )

        return X_test, Y_test, valid_test

    try:

        with st.spinner("Loading test dataset..."):
            X_test, Y_test, valid_test = load_comparison_data()

        st.success(
            f"Loaded {len(X_test)} test images successfully."
        )

        image_index = st.number_input(
            "Select test image index",
            min_value=0,
            max_value=len(X_test) - 1,
            value=0,
            step=1
        )

        image_index = int(image_index)

        sample = X_test[image_index]
        actual = Y_test[image_index, :, :, 0].astype(bool)
        valid = valid_test[image_index, :, :, 0].astype(bool)

        valid_pixels = int(np.sum(valid))

        if valid_pixels == 0:
            st.warning("This test image has no valid pixels.")
            st.stop()

        actual_pixels = int(np.sum(actual & valid))
        actual_coverage = actual_pixels / valid_pixels * 100

        st.divider()
        st.subheader("Satellite Image")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                sample[:, :, 0],
                caption="Sentinel-1 VV Band",
                clamp=True,
                use_container_width=True
            )

        with col2:
            st.image(
                sample[:, :, 1],
                caption="Sentinel-1 VH Band",
                clamp=True,
                use_container_width=True
            )

        st.divider()
        st.subheader("Actual Ground-Truth Flood Mask")

        st.image(
            (actual & valid).astype(np.uint8) * 255,
            caption="Actual Flood Mask",
            clamp=True,
            use_container_width=True
        )

        st.metric(
            "Actual Flood Coverage",
            f"{actual_coverage:.2f}%"
        )

        # Run all three models
        results = []
        prediction_masks = {}

        st.divider()
        st.subheader("Predictions from Three Models")

        with st.spinner("Running all three models..."):

            for model_name, model_path in MODEL_PATHS.items():

                model = load_model(model_path)

                probabilities = model.predict(
                    sample[np.newaxis, ...],
                    verbose=0
                )[0, :, :, 0]

                predicted = probabilities >= THRESHOLD

                prediction_masks[model_name] = predicted

                TP = int(np.sum(actual & predicted & valid))
                FP = int(np.sum(~actual & predicted & valid))
                FN = int(np.sum(actual & ~predicted & valid))
                TN = int(np.sum(~actual & ~predicted & valid))

                total = TP + FP + FN + TN

                accuracy = (
                    (TP + TN) / total * 100 if total else 0
                )

                precision = (
                    TP / (TP + FP) * 100
                    if TP + FP else 0
                )

                recall = (
                    TP / (TP + FN) * 100
                    if TP + FN else 0
                )

                dice = (
                    2 * TP / (2 * TP + FP + FN) * 100
                    if 2 * TP + FP + FN else 0
                )

                iou = (
                    TP / (TP + FP + FN) * 100
                    if TP + FP + FN else 0
                )

                coverage = (
                    np.sum(predicted & valid) / valid_pixels * 100
                )

                results.append({
                    "Model": model_name,
                    "Accuracy (%)": round(accuracy, 2),
                    "Precision (%)": round(precision, 2),
                    "Recall (%)": round(recall, 2),
                    "Dice (%)": round(dice, 2),
                    "IoU (%)": round(iou, 2),
                    "Predicted Coverage (%)": round(coverage, 2)
                })

        st.success("All three models completed prediction.")

        # Display masks
        columns = st.columns(3)

        for column, model_name in zip(
            columns, prediction_masks.keys()
        ):

            with column:

                st.markdown(f"**{model_name}**")

                st.image(
                    (
                        (prediction_masks[model_name] & valid)
                        .astype(np.uint8) * 255
                    ),
                    caption=f"{model_name} Prediction",
                    clamp=True,
                    use_container_width=True
                )

        st.divider()
        st.subheader("Performance Comparison")

        comparison_df = pd.DataFrame(results)

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Actual vs Predicted Flood Coverage")

        coverage_df = pd.DataFrame({
            "Category": [
                "Actual Flood",
                "U-Net",
                "U-Net++",
                "Attention U-Net"
            ],
            "Coverage (%)": [
                actual_coverage,
                *[
                    row["Predicted Coverage (%)"]
                    for row in results
                ]
            ]
        })

        st.bar_chart(
            coverage_df.set_index("Category")
        )

        st.caption(
            "Metrics shown here are calculated for the selected "
            "test image only. They are not the full test-set results."
        )

        # =============================================
        # DOWNLOADABLE FLOOD REPORTS
        # =============================================

        st.divider()
        st.subheader("Test Dataset Evaluation Downloads")

        report_df = comparison_df.copy()
        report_df.insert(
            1,
            "Actual Coverage (%)",
            round(actual_coverage, 2)
        )

        csv_data = report_df.to_csv(index=False).encode("utf-8")

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                label="Download Test Evaluation CSV",
                data=csv_data,
                file_name="FloodVision_AI_Test_Dataset_Evaluation.csv",
                mime="text/csv",
                use_container_width=True
            )

        
        # Generate Professional Multi-Page PDF Report
        pdf_buffer = io.BytesIO()

        from matplotlib.backends.backend_pdf import PdfPages

        results_df = pd.DataFrame(results)
        model_names = results_df["Model"].tolist()

        metric_columns = [
            "Accuracy (%)",
            "Precision (%)",
            "Recall (%)",
            "Dice (%)",
            "IoU (%)"
        ]

        # Identify strongest model using selected-image Dice score
        best_dice_index = results_df["Dice (%)"].idxmax()
        best_dice_model = results_df.loc[best_dice_index, "Model"]
        best_dice_score = results_df.loc[best_dice_index, "Dice (%)"]

        best_iou_index = results_df["IoU (%)"].idxmax()
        best_iou_model = results_df.loc[best_iou_index, "Model"]
        best_iou_score = results_df.loc[best_iou_index, "IoU (%)"]

        with PdfPages(pdf_buffer) as pdf:

            # =========================================
            # PAGE 1: COVER AND PROJECT SUMMARY
            # =========================================

            fig = plt.figure(figsize=(11.7, 8.3))
            fig.patch.set_facecolor("#F4F7FC")

            fig.text(
                0.5, 0.84,
                "FLOODVISION AI",
                fontsize=30,
                fontweight="bold",
                color="#123B70",
                ha="center"
            )

            fig.text(
                0.5, 0.77,
                "AI-Powered Flood Detection and Satellite Image Segmentation",
                fontsize=13,
                color="#405570",
                ha="center"
            )

            fig.text(
                0.5, 0.67,
                "FLOOD ANALYSIS REPORT",
                fontsize=21,
                fontweight="bold",
                color="#1674B8",
                ha="center"
            )

            fig.text(
                0.5, 0.60,
                f"Selected Test Image: {image_index}",
                fontsize=13,
                ha="center",
                color="#26384D"
            )

            fig.text(
                0.5, 0.53,
                "PROJECT OVERVIEW",
                fontsize=15,
                fontweight="bold",
                ha="center",
                color="#123B70"
            )

            overview = (
                "FloodVision-AI uses Sentinel-1 satellite imagery and "
                "deep-learning segmentation models to identify flood-affected "
                "areas. This report presents the ground-truth mask, predictions "
                "from three models, and evaluation metrics for the selected "
                "test image."
            )

            fig.text(
                0.5, 0.44,
                overview,
                fontsize=12,
                ha="center",
                va="center",
                wrap=True,
                color="#26384D",
                linespacing=1.7
            )

            fig.text(
                0.5, 0.32,
                f"Actual Flood Coverage: {actual_coverage:.2f}%",
                fontsize=17,
                fontweight="bold",
                ha="center",
                color="#B52B2B"
            )

            fig.text(
                0.5, 0.25,
                f"Best Dice Score: {best_dice_model} ({best_dice_score:.2f}%)",
                fontsize=13,
                ha="center",
                color="#123B70"
            )

            fig.text(
                0.5, 0.20,
                f"Best IoU Score: {best_iou_model} ({best_iou_score:.2f}%)",
                fontsize=13,
                ha="center",
                color="#123B70"
            )

            fig.text(
                0.5, 0.08,
                "B.Tech Capstone Project | FloodVision-AI",
                fontsize=10,
                ha="center",
                color="#607080"
            )

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # =========================================
            # PAGE 2: SATELLITE IMAGE AND FLOOD MASKS
            # =========================================

            fig, axes = plt.subplots(2, 3, figsize=(11.7, 8.3))
            fig.suptitle(
                "Satellite Image and Flood Segmentation Results",
                fontsize=17,
                fontweight="bold"
            )

            axes = axes.flatten()

            image_panels = [
                (sample[:, :, 0], "Sentinel-1 VV Band", "gray"),
                (sample[:, :, 1], "Sentinel-1 VH Band", "gray"),
                ((actual & valid).astype(np.uint8) * 255,
                 "Actual Ground-Truth Mask", "gray"),
                ((prediction_masks["U-Net"] & valid).astype(np.uint8) * 255,
                 "U-Net Prediction", "gray"),
                ((prediction_masks["U-Net++"] & valid).astype(np.uint8) * 255,
                 "U-Net++ Prediction", "gray"),
                ((prediction_masks["Attention U-Net"] & valid).astype(np.uint8) * 255,
                 "Attention U-Net Prediction", "gray")
            ]

            for ax, (image_data, title, cmap) in zip(axes, image_panels):
                ax.imshow(image_data, cmap=cmap)
                ax.set_title(title, fontsize=10, fontweight="bold")
                ax.axis("off")

            fig.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # =========================================
            # PAGE 3: PERFORMANCE METRICS TABLE
            # =========================================

            fig, ax = plt.subplots(figsize=(11.7, 8.3))
            ax.axis("off")

            fig.suptitle(
                "Model Performance Comparison",
                fontsize=19,
                fontweight="bold",
                y=0.94
            )

            ax.text(
                0.5, 0.82,
                "Evaluation results for the selected test image",
                ha="center",
                fontsize=12,
                transform=ax.transAxes
            )

            table = ax.table(
                cellText=report_df.round(2).values,
                colLabels=report_df.columns,
                cellLoc="center",
                loc="center"
            )

            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.15, 2.2)

            ax.text(
                0.5, 0.12,
                "Metrics are calculated using the valid pixels of the selected image.",
                ha="center",
                fontsize=10,
                transform=ax.transAxes
            )

            fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.90])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # =========================================
            # PAGE 4: FLOOD COVERAGE COMPARISON
            # =========================================

            fig, ax = plt.subplots(figsize=(11.7, 8.3))

            bars = ax.bar(
                coverage_df["Category"],
                coverage_df["Coverage (%)"]
            )

            ax.set_title(
                "Actual vs Predicted Flood Coverage",
                fontsize=18,
                fontweight="bold",
                pad=20
            )

            ax.set_ylabel("Flood Coverage (%)")
            ax.set_xlabel("Ground Truth and Segmentation Models")
            ax.grid(axis="y", alpha=0.25)

            ax.bar_label(
                bars,
                labels=[
                    f"{value:.2f}%"
                    for value in coverage_df["Coverage (%)"]
                ],
                padding=4
            )

            ax.set_ylim(
                0,
                max(coverage_df["Coverage (%)"].max() * 1.25, 1)
            )

            plt.xticks(rotation=12)
            fig.tight_layout()

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # =========================================
            # PAGE 5: MODEL METRICS CHART
            # =========================================

            fig, ax = plt.subplots(figsize=(11.7, 8.3))

            x = np.arange(len(metric_columns))
            width = 0.23

            for i, model_name in enumerate(model_names):
                values = [
                    float(results_df.loc[
                        results_df["Model"] == model_name, metric
                    ].iloc[0])
                    for metric in metric_columns
                ]

                ax.bar(
                    x + (i - 1) * width,
                    values,
                    width,
                    label=model_name
                )

            ax.set_title(
                "Comparison of Segmentation Performance Metrics",
                fontsize=17,
                fontweight="bold",
                pad=20
            )

            ax.set_ylabel("Score (%)")
            ax.set_xticks(x)
            ax.set_xticklabels(metric_columns)
            ax.set_ylim(0, 105)
            ax.legend()
            ax.grid(axis="y", alpha=0.25)

            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # =========================================
            # PAGE 6: CONCLUSION
            # =========================================

            fig, ax = plt.subplots(figsize=(11.7, 8.3))
            ax.axis("off")

            fig.suptitle(
                "Analysis and Conclusion",
                fontsize=20,
                fontweight="bold",
                y=0.93
            )

            conclusion = (
                f"For the selected test image, {best_dice_model} achieved "
                f"the highest Dice score of {best_dice_score:.2f}%. "
                f"{best_iou_model} achieved the highest IoU score of "
                f"{best_iou_score:.2f}%. These results describe this "
                f"individual image only.\n\n"
                f"The actual ground-truth flood coverage is "
                f"{actual_coverage:.2f}%. The predicted coverage values "
                f"are shown in the flood coverage comparison section.\n\n"
                "U-Net, U-Net++, and Attention U-Net provide three different "
                "segmentation approaches. Their performance should be "
                "evaluated using multiple metrics rather than accuracy alone."
            )

            ax.text(
                0.08, 0.72,
                conclusion,
                fontsize=13,
                va="top",
                wrap=True,
                linespacing=1.8,
                transform=ax.transAxes
            )

            ax.text(
                0.5, 0.13,
                "End of FloodVision-AI Analysis Report",
                fontsize=12,
                fontweight="bold",
                ha="center",
                transform=ax.transAxes
            )

            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        pdf_buffer.seek(0)

        with col2:
            st.download_button(
                label="Download Test Evaluation PDF",
                data=pdf_buffer.getvalue(),
                file_name="FloodVision_AI_Test_Dataset_Evaluation.pdf",
                mime="application/pdf",
                use_container_width=True
            )


    except Exception as e:

        st.error(
            f"Unable to process the comparison: {e}"
        )


# =====================================================
# 17. GLOBAL FOOTER
# =====================================================

st.markdown("""
<div class="footer">
    FLOODVISION AI &nbsp; | &nbsp;
    AI-POWERED FLOOD DETECTION &nbsp; | &nbsp;
    B.TECH CAPSTONE PROJECT
</div>
""", unsafe_allow_html=True)
