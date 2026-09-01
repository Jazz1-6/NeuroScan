# NeuroScan — AI-Assisted MRI Analysis
## Complete Project Documentation

---

## 1. Introduction & Purpose

**NeuroScan** is a local, desktop application that classifies 2D brain MRI images into one of four categories using a deep learning model:

| Class | Meaning |
|---|---|
| `glioma` | Features consistent with a glioma-like pattern |
| `meningioma` | Features consistent with a meningioma-like pattern |
| `pituitary` | Features consistent with a pituitary tumour-like pattern |
| `notumor` | No tumour-like features detected |

The application is explicitly built as a **research-use screening tool**, not a diagnostic device. Every screen in the GUI reinforces this: results are always framed as "AI classification," never as a "diagnosis," and the app repeatedly states that a qualified healthcare professional must review any output.

**Two operating modes:**
1. `python brain_tumor_detector.py` — opens the GUI application
2. `python brain_tumor_detector.py --train` — (re)trains the model from the dataset and exits

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                      brain_tumor_detector.py              │
│                                                             │
│  ┌───────────────┐   ┌──────────────────┐   ┌───────────┐ │
│  │ Data Pipeline │──▶│  Model (Keras/TF)│──▶│  GUI (Tk) │ │
│  │ zip → dataset │   │ EfficientNetB0   │   │ 3 screens │ │
│  └───────────────┘   └──────────────────┘   └───────────┘ │
│         ▲                     │                     │      │
│         │            model_metrics.json      report (.html)│
│      archive.zip     brain_tumor_model.keras              │
│                       class_names.json                     │
└─────────────────────────────────────────────────────────┘
```

The program has two independent halves that share only the trained model file:

- **Training half** (`prepare_dataset`, `make_dataset`, `build_model`, `train_model`) — runs once, offline, produces `brain_tumor_model.keras`, `class_names.json`, and `model_metrics.json`.
- **Application half** (`NeuroScanApp` and everything it calls) — loads the already-trained model and never touches the dataset or training code.

Everything runs **locally**. No image, prediction, or report is uploaded anywhere — this is stated in the module docstring and repeated in the GUI's Terms screen and Help dialog.

---

## 3. Tech Stack

| Layer | Technology | Role |
|---|---|---|
| Language | Python 3 | Entire application |
| Deep learning | TensorFlow / Keras | Model definition, training, inference |
| Base architecture | EfficientNetB0 (ImageNet weights) | Transfer-learning backbone |
| Image handling | Pillow (PIL) | Load, resize, enhance, blend images |
| Numerics | NumPy | Array math, quality metrics, Grad-CAM math |
| GUI | Tkinter + ttk | Desktop interface |
| Data interchange | JSON | Class names, metrics |
| Packaging of results | HTML (self-contained, base64 images) | Downloadable report |
| Dataset packaging | ZIP (`archive.zip`) | Source of the training images |

---

## 4. Project File Structure

```
project/
├── brain_tumor_detector.py     # this file — everything lives in one module
├── archive.zip                 # (training only) raw dataset archive
├── dataset/                    # (training only) extracted Training/Testing folders
├── brain_tumor_model.keras     # trained model weights + architecture
├── class_names.json            # ["glioma","meningioma","notumor","pituitary"]
├── model_metrics.json          # accuracy, precision, recall, F1, confusion matrix
└── reports/                    # auto-created; timestamped HTML reports land here
```

`ROOT` is computed as the folder containing the script (`Path(__file__).resolve().parent`), so every one of these paths is relative to wherever the `.py` file lives — the app is fully portable.

---

## 5. Code Walkthrough

### 5.1 Configuration & Constants

```python
MODEL_VERSION, MODEL_ARCHITECTURE = "v1.1", "EfficientNetB0"
LOW_CONFIDENCE_THRESHOLD, LOW_SEPARATION_THRESHOLD = 0.60, 0.12
CLASS_NAMES, IMAGE_SIZE = ["glioma", "meningioma", "notumor", "pituitary"], (224, 224)
```

- `IMAGE_SIZE = (224, 224)` matches what EfficientNetB0 expects.
- `LOW_CONFIDENCE_THRESHOLD (0.60)` — if the model's top score is below 60%, the result is flagged "low confidence" regardless of which class won.
- `LOW_SEPARATION_THRESHOLD (0.12)` — if the gap between the top two predicted probabilities is under 12 percentage points, the model is "unsure between two classes," which is flagged separately from raw confidence. This is important: a model can be 65% confident but still be nearly tied with the second class (e.g., 65% vs 60%) — separation catches that case, confidence alone would not.
- `C = {...}` is the single source of truth for the entire dark violet/teal colour palette used across every screen (`bg`, `surface`, `card`, `violet`, `teal`, `coral` for danger states, `amber` for warnings, etc.). Every widget colour in the GUI is pulled from this one dictionary, so re-theming the whole app means editing this one line.

Two small pure functions turn a raw class label into user-facing copy:

- `display_name(label)` → e.g. `"glioma"` → `"Glioma-like pattern"`
- `summary_text(label)` → a one-paragraph explanation shown under "AI Analysis Summary"
- `CLASS_INFO` → a longer, static explanatory paragraph per class (what the tissue type actually is), shown for user education, not diagnosis.

### 5.2 Data Preparation

**`prepare_dataset() -> Path`**
Idempotent dataset extraction. If `dataset/Training` already exists, it's reused; otherwise `archive.zip` is unzipped into `dataset/`. Raises `FileNotFoundError` if no zip exists, and `RuntimeError` if the zip's contents don't match the expected layout.

**`make_dataset(folder, subset=None) -> tf.data.Dataset`**
Thin wrapper around `tf.keras.utils.image_dataset_from_directory`:
- Labels come from folder names, restricted to `CLASS_NAMES` (so directory order never matters).
- `label_mode="categorical"` → one-hot vectors (needed for `CategoricalCrossentropy`).
- When `subset` is given (`"training"` / `"validation"`), an 80/20 split is carved out of the same folder with a fixed `seed=42` for reproducibility.
- `.prefetch(tf.data.AUTOTUNE)` overlaps disk I/O with GPU/CPU computation.

### 5.3 Model Building & Training

**`build_model()`**

```
Input (224,224,3)
   │
Data Augmentation (flip, rotate ±8%, translate ±6%, zoom ±10%, contrast ±10%)
   │
EfficientNetB0 (frozen, ImageNet weights, no top)
   │
GlobalAveragePooling2D
   │
Dropout(0.35)
   │
Dense(4, softmax)
```

The augmentation layers only run during training (`training=True`); at inference time they're a no-op, so the same model graph is used for both.

**`train_model()` — two-phase transfer learning:**

1. **Phase 1 — Head training (up to 20 epochs).** The EfficientNetB0 base is completely frozen (`base.trainable = False`). Only the new `Dense(4)` head (plus pooling/dropout) learns. Optimizer: `Adam(1e-3)`.
2. **Phase 2 — Fine-tuning (up to 25 more epochs).** `base.trainable = True`, but everything except the **last 120 layers** is re-frozen, and all `BatchNormalization` layers are explicitly kept frozen regardless of position (standard practice — retraining BatchNorm statistics on a small dataset destabilizes training). Optimizer drops to `Adam(3e-5)` — two orders of magnitude smaller, since we're now nudging pretrained ImageNet weights rather than learning from scratch.

**Callbacks (used in both phases):**
- `EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True)` — stops early if validation accuracy stalls for 4 epochs, and rolls back to the best-performing weights seen.
- `ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.3)` — cuts the learning rate by 70% if validation loss stalls for 2 epochs.

**Loss:** `CategoricalCrossentropy(label_smoothing=0.04)` — the 4% label smoothing softens the one-hot targets slightly (e.g. `[1,0,0,0]` → `[0.97, 0.01, 0.01, 0.01]`), which discourages the model from becoming overconfident.

**After training:**
- Model saved to `brain_tumor_model.keras`; class list saved to `class_names.json`.
- If a `Testing/` folder exists, the model is evaluated: predictions vs. ground truth build a 4×4 confusion matrix, from which per-class precision, recall, and F1 are derived manually (not via a library), then macro-averaged.
- Everything (architecture, dataset sizes, accuracy, precision/recall/F1, the full confusion matrix) is written to `model_metrics.json`, which is what the GUI's "Model info" screen reads and renders — meaning that screen always reflects the *actual* last training run, never hardcoded numbers.

### 5.4 The GUI — `NeuroScanApp`

The whole application is one class. It does **not** use separate `Frame` subclasses per screen — instead, each "screen" is a method that:
1. Calls `self._clear_root()` to destroy every widget currently in the window.
2. Rebuilds that screen's widgets from scratch into `self.root`.

This is simple and robust (no stale-widget bugs) at the cost of rebuilding widgets on every navigation — acceptable for an app with only 3–4 screens.

**Shared helpers:**
- `_label(...)` — a `tk.Label` factory pre-wired to the app's palette, so every text label is one line instead of six.
- `_panel(parent)` — returns `(outer, inner)`: `outer` is a 1px dark frame that acts as a hairline border, `inner` is the actual card surface. This is how every "card" in the UI gets a subtle outline without using `ttk` styling quirks.
- `_header(step, title, subtitle)` — draws the top nav bar (`NeuroScan` wordmark + "MODEL READY" indicator) and, if `step` is 1/2/3, the three-step progress tracker (Terms → MRI analysis → Diagnosis summary) with ✓ for completed steps. If `step=None` (used only by the Model Info screen), it instead draws a simple "← Back" bar.
- `_go_back()` — returns to whatever screen last set `self._return_view`, so "Back" always goes somewhere sensible regardless of which screen you're on.

**Screen 1 — `show_terms()`**
A single centered card: an icon, three plain-language bullet points (research-use only / needs professional review / processed locally), a required checkbox, and a "Continue" button that stays disabled (`state="disabled"`) until `self.accept_terms` (a `tk.BooleanVar`) is checked — enforced by `_update_terms_button()`, bound to the checkbox's `command`.

**Screen 2 — `show_workspace()`** *(Input / analysis screen)*
Two side-by-side panels built by `_left_panel()` and `_analysis_panel()`:

- **Left — MRI Explorer.** A `Canvas` (`self.viewer`) that, before any image is loaded, renders an interactive **drop-zone** (dashed border, upload icon, "Click to choose an MRI image") built by `_draw_dropzone()`. Clicking it — handled in `_viewer_click()` — opens the file picker only while no image is loaded (`self._viewer_has_image` is `False`); once an image is loaded, the same canvas click instead starts panning (`pan_start`), and the cursor switches from a pointing hand to a four-way move icon. Hovering the empty drop-zone (`<Enter>`/`<Leave>` → `_set_dropzone_hover`) brightens its border and background for affordance. A toolbar below offers Open / Zoom + / Zoom − / Fit / Brightness / Contrast, all built generically via a small `toolbar_item()` closure so adding a new tool is a one-line addition.
- **Right — AI Explainability panel.** A 4-step pipeline (MRI loaded → Image quality checked → AI classification completed → Explainability map generated), each step rendered as a small status **icon** (not text glyphs): a solid teal circle with a check for "done," an animated spinning violet ring for "in progress" (driven by `_icon_spinner()` / `_stop_spinner()`, which self-schedule via `root.after(80, …)` until cancelled), and a dashed outline circle for "pending." Below that sits a live status line, a result placeholder, and the "Continue to diagnosis summary" button (disabled until analysis finishes).

**`open_image()`** — file picker → validation (must be PNG/JPEG, decodable, ≥32×32px) → on success, stores the image, resets any previous Grad-CAM/overlay, fits it to the viewer, flips `self._viewer_has_image = True`, switches the cursor, and immediately calls `self.start_analysis()`. Three distinct `except` blocks give tailored error dialogs for a corrupted file, an OS-level read failure, and a validation failure (too small / wrong format) — the user is never shown a raw traceback.

**Analysis pipeline (background thread):**

```
start_analysis()               ← runs on the main/UI thread
   │  sets step 1 to "spinning", steps 2-4 to "pending"
   ▼
threading.Thread(target=self.worker)   ← runs off the UI thread
   │
   ├─ quality_check()          numpy math on the raw image
   ├─ model.predict(...)       the actual EfficientNetB0 forward pass
   └─ gradcam(...)             one extra gradient pass for the heatmap
   │
   ▼
root.after(0, self.finish_analysis)    ← hands control back to the UI thread
```

Running inference on a background `threading.Thread` keeps the Tkinter event loop responsive (no frozen window during prediction). Tkinter is not thread-safe for widget updates, so the worker never touches widgets directly — it only computes, then schedules `finish_analysis` back onto the main thread via `root.after(0, …)`, which is the standard safe pattern for this.

**`gradcam(array, index)`** — implements Grad-CAM (Gradient-weighted Class Activation Mapping):
1. Pulls out the model's internal layers by name (`sequential` = augmentation, `efficientnetb0` = the base, `global_average_pooling2d`, `dense`).
2. Runs a forward pass inside a `tf.GradientTape`, watching the base model's last convolutional feature map.
3. Takes the gradient of the *predicted class's* score with respect to that feature map, global-average-pools those gradients into per-channel "importance weights," and does a weighted sum over the feature map's channels.
4. `ReLU`s the result (keeps only positive influence) and normalizes to `[0, 1]`.
5. Resizes that low-res heat map up to the original image size and false-colours it (a custom violet→coral gradient consistent with the app's palette, not the default red/blue "jet" colormap).

If any layer lookup or gradient computation fails, `gradcam()` returns `None` rather than crashing the whole analysis — the app simply shows the original image in place of the attention map.

**`finish_analysis(scores, cam)`** — the decision logic that turns raw softmax scores into the three-way status badge:

```python
if scores[top] < 0.60 or separation < 0.12:
    → "LOW CONFIDENCE RESULT" (amber)
elif label == "notumor":
    → "NO TUMOUR CLASS SELECTED" (teal)
else:
    → "PROFESSIONAL REVIEW REQUIRED" (coral)
```

It also computes `reliability` (`"HIGH"` if separation ≥ 0.12, else `"LIMITED"`), builds the overlay image (`make_overlay()` — a 55%-opacity `Image.blend` of the original and the Grad-CAM heatmap), generates the plain-text report body (`make_summary()`), flips every pipeline icon to "done," and stores everything needed for the next screen in `self.pending_diagnosis` — the Continue button doesn't navigate directly; it calls `open_pending_diagnosis()`, which unpacks that tuple into `show_diagnosis(...)`.

**Screen 3 — `show_diagnosis(...)`** *(unchanged by this round of edits)*
Left side: an image viewer with three tabs (Original MRI / AI Attention / Overlay) via `set_tab()`/`_refresh_tabs()`. Right side, top to bottom: status badge → predicted class name → top probability → AI analysis summary → "why this prediction" shortcut to the attention tab → a two-column Image Quality / Prediction Reliability block → a full probability bar chart (`show_probabilities()`) → action buttons ("Generate AI analysis report" and "Analyze another image").

**Shared image-viewer utilities** (used by both the workspace and diagnosis screens):
`render_views()` picks the correct base image for whichever canvas is active (via `current_base()`, which branches on `self.viewing_result` and `self.mode`), applies brightness/contrast (`ImageEnhance`), and draws it (`_draw()`) with the current zoom/pan applied. `fit_image()`, `change_zoom()`, `wheel_zoom()`, `pan_start()`/`pan_move()`, `nudge_brightness()`/`nudge_contrast()`, `reset_view()`, and `toggle_fullscreen()` round out a fairly complete lightweight image viewer, all sharing the same zoom/pan state (`self.zoom`, `self.pan_x`, `self.pan_y`).

**`generate_report()`** — builds a fully self-contained HTML file (images embedded as base64 `data:` URIs, so it has zero external dependencies and can be emailed or opened anywhere) into `reports/neuroscan_report_<timestamp>.html`, then confirms via a message box that nothing was uploaded.

**`show_model_info()`** — a fourth, non-numbered screen (reachable via the "Model info" button in the header) that reads `model_metrics.json` fresh every time it's opened and renders three summary cards (Overview / Dataset / Performance) plus a rendered confusion matrix with per-class accuracy, colour-coded green/amber/red by accuracy band.

### 5.5 Entry Point

```python
def open_app():
    if model/labels missing: print instructions, exit
    load model, run one throwaway prediction to "warm up" the graph
    NeuroScanApp(model, labels).run()

if __name__ == "__main__":
    --train  → train_model()
    (default) → open_app()
```

The "warm-up" prediction on a dummy zero-array is deliberate: the first real call to a freshly loaded Keras model is always slower (graph tracing / kernel selection), so doing it once at startup means the *first real image* a user uploads doesn't take noticeably longer than the second.

---

## 6. Application Workflow (End-to-End User Journey)

```
1. Launch app
      │
      ▼
2. TERMS SCREEN
   read 3 bullet points → tick checkbox → "Continue" enabled
      │
      ▼
3. INPUT / WORKSPACE SCREEN
   click drop-zone → file picker → pick PNG/JPG
      │
      ├─ validation fails → error dialog, stay on this screen
      │
      ▼ validation passes
   image fitted into viewer, analysis starts automatically
   step 1 spins → (in background) quality check + model predict + Grad-CAM
   all 4 steps flip to ✓ → "Continue to diagnosis summary" enabled
      │
      ▼
4. DIAGNOSIS SUMMARY SCREEN
   status badge + class + confidence
   toggle Original / AI Attention / Overlay tabs, zoom/pan, adjust brightness/contrast
   read image-quality & reliability panel, full probability bars
      │
      ├─ "Analyze another image" → back to step 3, fresh drop-zone
      └─ "Generate AI analysis report" → self-contained HTML saved to reports/
```

Optional side-branch: from the workspace or diagnosis header, "Model info" opens the fourth screen showing training/eval metrics, then "← Back" returns to wherever you came from.

---

## 7. Functioning Details — Key Algorithms Explained

### 7.1 Image Quality Check

```python
gray = grayscale(image) / 255
brightness = mean(gray)
contrast   = std(gray)
blur       = mean(|Δ_row(gray)|) + mean(|Δ_col(gray)|)   # crude sharpness proxy
```

- **Brightness** flagged if outside `(0.12, 0.88)` — catches images that are almost entirely black or almost entirely white.
- **Contrast** flagged if `< 0.06` — catches flat, washed-out images with little tonal variation.
- **Blur** flagged if `< 0.015` — the sum of average absolute pixel-to-pixel differences (horizontally and vertically) is a cheap stand-in for a proper Laplacian-variance sharpness metric: a crisp image has large jumps between neighbouring pixels at edges; a blurry one doesn't.

Any single flag downgrades the result from `"Good"` to `"Needs attention"`, and the specific flagged dimension(s) are named in the detail text shown to the user.

### 7.2 Confidence vs. Reliability — Why Two Numbers?

A single softmax confidence score can be misleading on its own. NeuroScan reports two separate signals:

| Signal | What it measures | Example that would fail this check alone |
|---|---|---|
| **Confidence** (top score) | How strongly the model backs its #1 pick | 70% glioma — sounds confident |
| **Separation / Reliability** (top − second) | How much better the #1 pick is than the runner-up | ...but if meningioma also scored 65%, the model is really "guessing" between two classes despite a seemingly high top score |

Flagging on *either* condition (`score < 60%` **or** `separation < 12%`) catches both "the model just isn't sure" and "the model is torn between two specific classes" — two different failure modes that a single confidence threshold would miss.

### 7.3 Grad-CAM (Explainability)

Grad-CAM answers *"which pixels made the model choose this class?"* without needing any architecture changes or retraining. Intuition:

1. Every channel of the last convolutional feature map represents some learned visual pattern.
2. The gradient of the predicted class's score with respect to that feature map tells you, per channel, "if this pattern were stronger, would the predicted score go up or down?"
3. Averaging that gradient over space gives one importance weight per channel.
4. A weighted sum of the feature-map channels (weighted by those importances) produces a coarse map of "where in the image did the patterns that drove this prediction live?"

**Important caveat baked into the UI copy:** this is *influence*, not a tumour boundary. A region can be highlighted because it strongly resembles healthy tissue that helped rule *out* another class, not because it necessarily contains an abnormality. NeuroScan's disclaimer text is explicit about this in three separate places (analysis screen, diagnosis screen, generated report).

---

## 8. Solved Example — Full Walkthrough

Suppose a user uploads an MRI slice and the model outputs raw softmax scores:

| Class | Score |
|---|---|
| glioma | 0.84 |
| meningioma | 0.09 |
| pituitary | 0.04 |
| notumor | 0.03 |

**Step-by-step processing:**

1. **Top class:** `argmax` → index 0 → `"glioma"`.
2. **Top-two separation:** sort scores → `[0.03, 0.04, 0.09, 0.84]` → `first, second = 0.84, 0.09` → `separation = 0.84 - 0.09 = 0.75` (75 percentage points).
3. **Status decision:**
   - `0.84 ≥ 0.60` ✓ (not low confidence)
   - `0.75 ≥ 0.12` ✓ (not low separation)
   - `label != "notumor"` → falls into the `else` branch → **"PROFESSIONAL REVIEW REQUIRED"** (coral badge).
4. **Reliability:** `separation (0.75) ≥ 0.12` → **"HIGH"** — "Clear separation was observed between the highest predicted class and alternative classifications."
5. **Quality check:** say brightness = 0.51, contrast = 0.14, blur = 0.021 — all within range → **"Good."**
6. **Grad-CAM:** heat map computed, blended at 55% opacity over the original for the Overlay tab.
7. **Diagnosis screen shows:** badge "PROFESSIONAL REVIEW REQUIRED" → "Glioma-like pattern" → "84.0%" confidence → summary paragraph naming glioma → Quality: Good / Reliability: High → full bar chart (glioma bar far longer than the other three, coloured violet as the winning class; the rest muted grey).
8. **User clicks "Generate AI analysis report"** → an HTML file is written with this exact text summary plus the original MRI and the overlay image embedded as base64 — no network call is made.

**Contrast with a low-confidence example:** scores `[glioma 0.41, meningioma 0.38, pituitary 0.12, notumor 0.09]` → top = 0.41 (< 0.60) → automatically **"LOW CONFIDENCE RESULT"** regardless of separation, because the confidence check alone already fails. This is the scenario the two-signal design in §7.2 exists to catch.

---

## 9. Real-Life Implementation & Considerations

### 9.1 Deployment
- Ships as a single Python script — no server, no database, no internet dependency at runtime. Distribution options: a `PyInstaller`/`cx_Freeze` executable for non-technical users, or run directly with `python brain_tumor_detector.py` for anyone with the environment set up.
- The model file (`.keras`) and class list travel alongside the script; retraining is a separate, explicit `--train` invocation so end users never accidentally kick off a multi-hour training run.

### 9.2 Privacy & Safety Posture
- **Local-only processing** is a hard architectural property here, not just a claim — there is no networking code anywhere in the file. This matters enormously for medical imaging, which is sensitive personal data in most jurisdictions (e.g., HIPAA in the US, GDPR in the EU) — a tool that never transmits images sidesteps a large category of compliance risk by design.
- The **Terms screen gate** (checkbox required before entry) and the **repeated disclaimers** (Terms screen, analysis screen, diagnosis screen, generated report, Help dialog) are a deliberate, layered approach to keeping "this is a screening aid, not a diagnosis" in front of the user at every decision point, not just once at startup.

### 9.3 Clinical & Ethical Limitations (worth stating explicitly to any real-world stakeholder)
- Trained on a fixed, single-source dataset — real-world generalization to different scanners, imaging protocols, or patient populations is **not guaranteed** and should be validated before any pilot use.
- A single 2D slice cannot represent a full 3D MRI volume; the model has no access to adjacent slices, other imaging sequences, or clinical history.
- The model performs **classification**, not **segmentation or localization** — Grad-CAM approximates "where the model looked," but the app deliberately does not draw a tumour boundary, because it cannot make that claim reliably.
- Any real deployment in a clinical or pre-clinical setting would need: a much larger and more diverse validation dataset, formal clinical validation studies, regulatory clearance (e.g., FDA / CE marking, depending on jurisdiction and intended use), and a human-in-the-loop workflow enforced by policy, not just UI copy.

### 9.4 Extensibility — Natural Next Steps
- Swap `EfficientNetB0` for a larger backbone (`EfficientNetV2`, a ViT) if more compute/data become available.
- Add DICOM support (real clinical MRIs are rarely plain PNG/JPEG) — would need a `pydicom`-based loader ahead of the existing PIL pipeline.
- Multi-slice / volumetric input instead of single 2D images.
- Export the confusion matrix and metrics as a versioned artifact so multiple trained models can be compared side-by-side over time.

### 9.5 Error Handling Philosophy
Every user-facing failure path in this app converts a raw exception into a plain-language explanation *plus* the technical exception text (never a bare traceback dialog): unreadable files, OS read errors, out-of-memory during inference, and missing model files each get their own tailored message (see `open_image()` and `analysis_error()`). This matters for a research tool used by non-programmers — it keeps the app debuggable without being intimidating.

---

## 10. Glossary

| Term | Meaning |
|---|---|
| **Softmax** | Converts raw model outputs into a probability distribution that sums to 1 across the 4 classes |
| **Transfer learning** | Reusing a model (EfficientNetB0) pretrained on a large unrelated dataset (ImageNet) as a starting point, rather than training from random weights |
| **Fine-tuning** | Phase 2 of training — unfreezing part of the pretrained base so it can adapt slightly to the new task |
| **Grad-CAM** | Gradient-weighted Class Activation Mapping — a technique for visualizing which image regions influenced a CNN's prediction |
| **Label smoothing** | A regularization trick that softens one-hot training targets to discourage overconfidence |
| **Confusion matrix** | A table showing predicted vs. actual class for every test example, used to derive precision/recall/F1 |
| **Macro-average** | Averaging a metric (precision, recall, F1) equally across classes, regardless of how many examples each class has |

---

## 11. Quick Reference — Running the Project

```bash
# One-time: train the model (requires archive.zip in the project folder)
python brain_tumor_detector.py --train

# Every other time: just open the app
python brain_tumor_detector.py
```

Required files for the app to start: `brain_tumor_model.keras` and `class_names.json` (both produced by `--train`). If either is missing, the app prints a one-line instruction and exits rather than crashing.
