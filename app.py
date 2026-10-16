import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Waste Sorting Classifier", page_icon="♻️", layout="centered")
ROOT = Path(__file__).resolve().parent


ROOT = Path(__file__).resolve().parent

HF_REPO_ID = "moyosore-ai/__waste__"


def locate_file(filename: str) -> Path | None:
    # First, check whether the file is already in GitHub.
    candidates = [
        ROOT / filename,
        ROOT / "outputs" / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    # If it is not in GitHub, try downloading it from Hugging Face.
    remote_candidates = [
        filename,
        f"outputs/{filename}",
    ]

    for remote_name in remote_candidates:
        try:
            downloaded_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=remote_name,
                repo_type="model",
            )
            return Path(downloaded_path)
        except Exception:
            continue

    return None

@st.cache_resource(show_spinner="Loading the trained waste classifier...")
def load_artifacts():
    model_path = locate_file("best_model.keras")
    summary_path = locate_file("results_summary.json")
    if model_path is None:
        raise FileNotFoundError(
            "Cannot find best_model.keras. Train the notebook first, then copy outputs/best_model.keras "
            "into this app folder (or keep it in an outputs/ subfolder)."
        )
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
        raise FileNotFoundError(
            "Class names are missing. Keep results_summary.json or class_names.json beside the model."
        )
    size = summary.get("img_size", [160, 160])
    normalization = summary.get("normalization", "mobilenet_v2")
    return model, classes, (int(size[0]), int(size[1])), normalization, summary


st.title("♻️ Waste Sorting Classifier")
st.write(
    "Upload a photo of one waste item to get the model's predicted category. "
    "This is a demo: check the prediction before making real recycling or disposal decisions."
)

try:
    model, class_names, image_size, normalization, summary = load_artifacts()
except Exception as exc:
    st.error(str(exc))
    st.info(
        "The trained model is not included in the starter files. Run the training notebook, "
        "then copy best_model.keras and results_summary.json from its outputs folder here."
    )
    st.stop()

with st.sidebar:
    st.subheader("Model information")
    st.write(f"**Architecture:** {summary.get('architecture', model.name)}")
    st.write(f"**Input size:** {image_size[0]} × {image_size[1]}")
    st.write(f"**Categories:** {len(class_names)}")
    if summary.get("test_accuracy") is not None:
        st.write(f"**Notebook test accuracy:** {summary['test_accuracy']:.1%}")
    st.caption("Test accuracy is the notebook's reported metric, not a guarantee for new photos.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
if uploaded:
    try:
        image = Image.open(uploaded)
        image = ImageOps.exif_transpose(image).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
        resized = image.resize((image_size[1], image_size[0]))
        arr = np.asarray(resized, dtype=np.float32)
        batch = np.expand_dims(arr, axis=0)
        if normalization == "mobilenet_v2":
            batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)
        elif normalization == "rescale_0_1":
            batch = batch / 255.0
        elif normalization == "none":
            pass  # EfficientNetB0 model in the notebook rescales internally.
        else:
            st.error(f"Unknown normalization setting in results_summary.json: {normalization}")
            st.stop()

        with st.spinner("Classifying image..."):
            scores = np.asarray(model.predict(batch, verbose=0))[0]
        if len(scores) != len(class_names):
            st.error(
                f"Model returned {len(scores)} scores but metadata lists {len(class_names)} classes. "
                "Check that the model and results_summary.json came from the same notebook run."
            )
            st.stop()
        order = np.argsort(scores)[::-1]
        best = int(order[0])
        st.subheader("Prediction")
        st.success(f"Predicted category: **{class_names[best]}**")
        st.metric("Model confidence", f"{float(scores[best]):.1%}")
        st.caption("Confidence is the model's score, not a calibrated probability of being correct.")
        st.subheader("Top predictions")
        for idx in order[: min(5, len(order))]:
            st.write(f"**{class_names[int(idx)]}** — {float(scores[int(idx)]):.1%}")
            st.progress(float(np.clip(scores[int(idx)], 0, 1)))
    except Exception as exc:
        st.error(f"Could not classify this image: {exc}")
else:
    st.info("Upload a clear photo with the waste item visible and reasonably well lit.")

st.divider()
st.caption("Research/demo use only. Real-world sorting performance depends on image quality and training data.")
