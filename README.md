# Waste Sorting Classifier — Streamlit demo

This project contains the Streamlit app and the notebook used to train the waste image classifier.

## Important: model output is not pre-included

The `outputs/` folder is a placeholder until you run the notebook. I cannot include real `best_model.keras` or `results_summary.json` files before the training notebook has run and produced them.

## 1. Generate the model files

1. Open `Waste_Classification_Combined_Fixed.ipynb` in Google Colab or Jupyter.
2. Run all cells from top to bottom. The notebook downloads/uses its dataset and trains/evaluates models; this may take a while and is easiest with a GPU.
3. At the final cell, it saves `waste_classifier_outputs.zip` in the notebook work directory. In Google Colab, download it from `/content/work/waste_classifier_outputs.zip` (or uncomment/run the `files.download(...)` line at the end of the notebook).
4. Extract the ZIP and copy `best_model.keras` and `results_summary.json` into this project's `outputs/` folder.

## 2. Files to upload to GitHub

Upload `app.py`, `requirements.txt`, `README.md`, and `Waste_Classification_Combined_Fixed.ipynb`. Once generated, add `outputs/best_model.keras` and `outputs/results_summary.json` too, provided the model file is within GitHub's file limits. If it is too large, host the model separately and modify the app to fetch it at startup, or use Git LFS if your hosting setup supports it.

## 3. Test locally

Use Python 3.10 or 3.11. From this folder, run:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown in the terminal, usually http://localhost:8501.

## 4. Deploy

Push the project files to a GitHub repository, then create an app at https://share.streamlit.io/ by selecting the repository, branch, and `app.py` as the main file. Confirm the trained model files are available before expecting predictions.
