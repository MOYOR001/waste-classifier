import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Waste Sorting Classifier", page_icon="♻️", layout="wide")

ROOT = Path(__file__).resolve().parent
HF_REPO_ID = "moyosore-ai/__waste__"

# emoji, group, short disposal hint (general guidance only; local rules vary)
CATEGORY_INFO = {
    "battery": ("🔋", "Hazardous waste", "Keep out of normal bins. Take to a battery or e-waste collection point."),
    "biological": ("🍌", "Organic waste", "Suitable for composting or an organic waste bin."),
    "cardboard": ("📦", "Recyclable", "Flatten it and keep it dry before recycling."),
    "clothes": ("👕", "Textiles", "Donate if still wearable, otherwise use a textile recycling point."),
    "glass": ("🍾", "Recyclable", "Rinse and sort by colour where your area requires it."),
    "metal": ("🥫", "Recyclable", "Rinse cans and containers before recycling."),
    "paper": ("📄", "Recyclable", "Keep it clean and dry. Greasy or wet paper often can't be recycled."),
    "plastic": ("🧴", "Recyclable (check type)", "Rinse it and check the recycling number on the item."),
    "shoes": ("👟", "Textiles", "Donate if wearable, or use a shoe recycling programme."),
    "trash": ("🗑️", "General waste", "Goes in the general waste bin."),
}

st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 2rem; max-width: 1150px;}
    .hero {
        background: linear-gradient(120deg, #1b7f4b 0%, #2fb36d 60%, #7fd6a0 100%);
        padding: 2rem 2.2rem; border-radius: 18px; color: white; margin-bottom: 1.5rem;
    }
    .hero h1 {margin: 0; font-size: 2.2rem; color: white;}
    .hero p {margin: .5rem 0 0 0; font-size: 1.05rem; opacity: .95;}
    .result-card {
        border-radius: 16px; padding: 1.4rem 1.6rem; margin-bottom: 1rem;
        border: 1px solid rgba(128,128,128,.25); background: rgba(47,179,109,.08);
    }
    .result-emoji {font-size: 3rem; line-height: 1;}
    .result-label {font-size: 1.9rem; font-weight: 700; margin: .3rem 0 0 0;}
    .pill {
        display: inline-block; padding: .2rem .8rem; border-radius: 999px;
        background: #1b7f4b; color: white; font-size: .85rem; margin-top: .4rem;
    }
    .tip {margin-top: .8rem; font-size: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def locate_file(filename: str) -> Path | None:
    candidates = [ROOT / filename, ROOT / "outputs" / filename]
    for path in candidates:
        if path.exists():
            return path
    for remote_name in (filename, f"outputs/{filename}"):
        try:
            return Path(
                hf_hub_download(repo_id=HF_REPO_ID, filename=remote_name, repo_type="model")
            )
        except Exception:
            continue
    return None


@st.cache_resource(show_spinner="Loading the trained waste classifier...")
def load_artifacts():
    model_path = locate_file("best_model.keras")
    summary_path = locate_file("results_summary.json")
    if model_path is None:
        raise FileNotFoundError("Cannot find best_model.keras in the repo or on Hugging Face.")
    model = tf.keras.models.load_model(model_path, compile=False)
    summary = {}
    if summary_path is not None:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    classes = summary.get("class_names")
    if not classes:
        classes_path = locate_file("class_names.json")
        if classes_path:
            with classes_path.open("r", encoding="utf-8") as f:
                classes = json.load(f)
    if not classes:
        raise FileNotFoundError("Class names are missing from results_summary.json / class_names.json.")
    size = summary.get("img_size", [160, 160])
    normalization = summary.get("normalization", "mobilenet_v2")
    return model, classes, (int(size[0]), int(size[1])), normalization, summary


def classify(image: Image.Image, model, image_size, normalization):
    resized = image.resize((image_size[1], image_size[0]))
    batch = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
    if normalization == "mobilenet_v2":
        batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)
    elif normalization == "rescale_0_1":
        batch = batch / 255.0
    # "none": the EfficientNetB0 model rescales internally
    return np.asarray(model.predict(batch, verbose=0))[0]


# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
        <h1>♻️ Waste Sorting Classifier</h1>
        <p>Upload or capture a photo of one waste item and the model will predict its category.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model, class_names, image_size, normalization, summary = load_artifacts()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("Model information")
    st.write(f"**Architecture:** {summary.get('architecture', model.name)}")
    st.write(f"**Input size:** {image_size[0]} × {image_size[1]}")
    st.write(f"**Categories:** {len(class_names)}")
    if summary.get("test_accuracy") is not None:
        st.metric("Notebook test accuracy", f"{summary['test_accuracy']:.1%}")
    st.caption("Test accuracy is the notebook's reported metric, not a guarantee for new photos.")
    st.divider()
    st.subheader("Tips for best results")
    st.markdown(
        "- One item per photo\n- Plain background\n- Good, even lighting\n- Item fills most of the frame"
    )

# ---------- Tabs ----------
tab_classify, tab_model = st.tabs(["🔍 Classify", "📊 About the model"])

with tab_classify:
    left, right = st.columns([1, 1], gap="large")

    with left:
        source = st.radio("Image source", ["Upload a file", "Use camera"], horizontal=True)
        if source == "Upload a file":
            file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
        else:
            file = st.camera_input("Take a photo")

        image = None
        if file:
            try:
                image = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
                st.image(image, caption="Your image", use_container_width=True)
            except Exception as exc:
                st.error(f"Could not read this image: {exc}")

    with right:
        if image is None:
            st.info("👈 Add a clear photo of one waste item to see the prediction here.")
        else:
            try:
                with st.spinner("Classifying image..."):
                    scores = classify(image, model, image_size, normalization)
                if len(scores) != len(class_names):
                    st.error(
                        f"Model returned {len(scores)} scores but there are {len(class_names)} class names."
                    )
                    st.stop()

                order = np.argsort(scores)[::-1]
                best = int(order[0])
                label = class_names[best]
                conf = float(scores[best])
                emoji, group, tip = CATEGORY_INFO.get(label, ("♻️", "Unknown", ""))

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-emoji">{emoji}</div>
                        <div class="result-label">{label.capitalize()}</div>
                        <span class="pill">{group}</span>
                        <div class="tip">💡 {tip}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                c1, c2 = st.columns(2)
                c1.metric("Model confidence", f"{conf:.1%}")
                runner_up = class_names[int(order[1])]
                c2.metric("Second guess", runner_up.capitalize(), f"{float(scores[int(order[1])]):.1%}")

                if conf < 0.5:
                    st.error("Low confidence. Try a clearer photo with only one item.")
                elif conf < 0.8:
                    st.warning("Moderate confidence. Double-check this prediction.")
                else:
                    st.success("High confidence.")

                st.markdown("**Top predictions**")
                for idx in order[: min(5, len(order))]:
                    name = class_names[int(idx)]
                    e = CATEGORY_INFO.get(name, ("♻️",))[0]
                    st.write(f"{e} **{name.capitalize()}** — {float(scores[int(idx)]):.1%}")
                    st.progress(float(np.clip(scores[int(idx)], 0, 1)))

                st.caption(
                    "Confidence is the model's score, not a calibrated probability. "
                    "Disposal hints are general; follow your local recycling rules."
                )
            except Exception as exc:
                st.error(f"Could not classify this image: {exc}")

with tab_model:
    st.subheader("How well does the model perform?")

    report = summary.get("per_class_test_report", {})
    rows = {
        k: v["f1-score"]
        for k, v in report.items()
        if k in class_names and isinstance(v, dict) and "f1-score" in v
    }
    if rows:
        st.markdown("**Test F1-score per category**")
        df = pd.DataFrame({"F1-score": rows}).sort_values("F1-score")
        st.bar_chart(df)
        st.caption("Lower bars (such as trash and metal) are the categories the model confuses most often.")

    val = summary.get("validation_results", {})
    if val:
        st.markdown("**Model comparison (validation)**")
        comp = pd.DataFrame(val).T[["val_accuracy", "val_loss", "params_M", "train_min"]]
        comp.columns = ["Val accuracy", "Val loss", "Params (M)", "Train time (min)"]
        st.dataframe(comp, use_container_width=True)

    light = summary.get("lighting_robustness", {})
    if light.get("factors") and light.get("accuracy"):
        st.markdown("**Accuracy under different brightness levels**")
        ldf = pd.DataFrame(
            {"Accuracy": light["accuracy"]},
            index=[f"{f}x" for f in light["factors"]],
        )
        st.line_chart(ldf)
        st.caption("1.0x is the original brightness. Accuracy drops most at very bright (2.0x) images.")

st.divider()
st.caption("Research/demo use only. Real-world performance depends on image quality and training data.")
