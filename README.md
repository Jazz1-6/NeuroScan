# NeuroScan — Brain MRI Classifier (Research Project)

This is a desktop app I built to classify brain MRI scans into four categories — glioma, meningioma, pituitary tumor, or no tumor — using a fine-tuned EfficientNetB0 model. It runs entirely on your own machine: no cloud, no API calls, no uploading your images anywhere.

**Before anything else: this is not a medical device.** It doesn't diagnose anything, it hasn't been validated for clinical use, and it should never be treated as a substitute for an actual radiologist. I built this as a learning project around image classification and model explainability — not as something anyone should rely on for real medical decisions.

## What it actually does

You pick an MRI image (PNG/JPG), and the app:
- Classifies it into one of the four categories above, with a confidence score
- Shows a Grad-CAM heatmap so you can see *what part of the image* the model was actually looking at when it made its decision — this matters a lot, because a model can be "confident" for the wrong reasons, and this is the closest thing to a sanity check you get
- Flags low-confidence results as uncertain instead of pretending to be sure
- Lets you generate a local HTML report and view stored model-performance stats

It only ever looks at one 2D image at a time — it doesn't do full MRI volume analysis, doesn't locate or outline tumors, and a "no tumor" result doesn't rule anything out. Every result needs a real professional to actually look at it.

## Getting it running

You'll need Python 3.11+.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python brain_tumor_detector.py
```

It expects `brain_tumor_model.keras` and `class_names.json` to be sitting in the same folder — they're already included in this repo, so you shouldn't need to do anything extra.

## How well does it actually work?

Honestly, not perfectly — and I'd rather tell you the real numbers than oversell it.

Current test-set accuracy: **~93.8%**, but that number hides some real variation between classes:

| Class | Accuracy |
|---|---|
| Pituitary | 100% |
| No tumor | ~99.75% |
| Meningioma | ~92.5% |
| Glioma | ~83% |

Glioma is clearly the weakest spot — the model mixes it up with meningioma more than I'd like. This isn't just a "my model is bad" thing — glioma is genuinely the hardest of the four to tell apart visually, and there are some known label-quality issues in the public dataset this was trained on for that specific class. I tried class-weighting the loss function to push the model to focus more on glioma, but it actually made things slightly worse overall, so I reverted that. What actually moved the needle was un-freezing a lot more of the EfficientNet backbone during fine-tuning instead of leaving most of it frozen — accuracy jumped from ~83% to ~93.8% just from that one change.

Full metrics (confusion matrix, precision/recall per class) are in `model_metrics.json`, and viewable inside the app under "Model Performance."

## Retraining it yourself

If you want to retrain on your own copy of the dataset:

1. Drop `archive.zip` (a zip of the dataset, with `Training/` and `Testing/` folders inside, each split into `glioma/`, `meningioma/`, `notumor/`, `pituitary/`) into the project folder
2. Run:
   ```powershell
   python brain_tumor_detector.py --train
   ```

It'll unzip the archive, train in two phases (frozen backbone, then partial fine-tuning), and overwrite the model file with whatever it produces, along with fresh metrics. Expect this to take a while without a GPU — think an hour or more, not minutes.

I used the public [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) on Kaggle (glioma/meningioma/notumor/pituitary, ~7,000 images).

## What's actually in this repo

```
brain_tumor_detector.py   — the whole app: training, inference, GUI, Grad-CAM, reporting
brain_tumor_model.keras   — the trained model
class_names.json          — class label order the model was trained on
model_metrics.json        — real accuracy/confusion-matrix numbers from the last training run
requirements.txt          — the three packages you need
```

Not included (and gitignored on purpose): the raw dataset, generated HTML reports, and the usual Python environment clutter.

## Why I built it this way

I wanted something that didn't just spit out a label with fake confidence — the whole reason the Grad-CAM view exists is that a black-box "94% glioma" answer isn't actually useful or trustworthy on its own. If the model's attention map is lighting up somewhere that makes no anatomical sense, that's a signal something's wrong, even if the confidence score looks high. That's the whole point of a tool like this being explainable rather than just accurate-on-paper.

If you find something broken or have ideas to push glioma accuracy further, I'm all ears — this is very much a project I'm still iterating on.
