"""NeuroScan — a local, research-only 2D MRI image classification viewer.

Run normally to open the app. Run with --train only to deliberately retrain.
No uploaded image, result, or report leaves this computer.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageEnhance, ImageTk, UnidentifiedImageError
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
ZIP_FILE, DATA_DIR = ROOT / "archive.zip", ROOT / "dataset"
MODEL_FILE, LABEL_FILE = ROOT / "brain_tumor_model.keras", ROOT / "class_names.json"
METRICS_FILE = ROOT / "model_metrics.json"
MODEL_VERSION, MODEL_ARCHITECTURE = "v1.1", "EfficientNetB0"
LOW_CONFIDENCE_THRESHOLD, LOW_SEPARATION_THRESHOLD = 0.60, 0.12
CLASS_NAMES, IMAGE_SIZE = ["glioma", "meningioma", "notumor", "pituitary"], (224, 224)
DISCLAIMER = "Research-use classifier only. This output is not a diagnosis and should not be used for clinical decision-making."
C = {"bg":"#111315", "surface":"#191C1F", "card":"#22262A", "hover":"#2A2F34", "violet":"#8B5CF6", "violet2":"#A78BFA", "purple":"#6D28D9", "teal":"#2DD4BF", "teal_bg":"#123832", "amber":"#F59E0B", "amber_bg":"#3A2A12", "coral":"#FB7185", "coral_bg":"#3A1820", "text":"#F5F5F4", "muted":"#A8B0B8", "dim":"#6B7280"}


def display_name(label: str) -> str:
    return {"glioma":"Glioma-like pattern", "meningioma":"Meningioma-like pattern", "pituitary":"Pituitary tumour-like pattern", "notumor":"No-tumour class selected"}.get(label, label.title())


def summary_text(label: str) -> str:
    if label == "notumor":
        return "The model did not identify features consistent with a tumour class in this image. Regions highlighted in the AI attention visualization show where the model's decision was concentrated."
    name = {"glioma":"glioma", "meningioma":"meningioma", "pituitary":"pituitary"}.get(label, label)
    return f"The model identified image features most consistent with the {name} class. Regions highlighted in the AI attention visualization contributed most strongly to this classification."


CLASS_INFO = {
    "glioma": "Gliomas arise from glial cells within brain tissue. Their appearance and clinical significance can vary widely, and further interpretation requires review of the complete imaging study and relevant clinical information.",
    "meningioma": "Meningiomas arise from tissues surrounding the brain. Their characteristics and clinical significance can vary, and further interpretation requires review of the complete imaging study and relevant clinical information.",
    "pituitary": "Pituitary tumours arise in or near the pituitary gland at the base of the brain. Their characteristics and clinical significance can vary, and further interpretation requires review of the complete imaging study and relevant clinical information.",
    "notumor": "No tumour-like features were identified in this image by the model. This reflects a model classification only and does not rule out findings that would need clinical evaluation.",
}


def load_metrics() -> dict | None:
    if not METRICS_FILE.exists(): return None
    try: return json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError): return None


def prepare_dataset() -> Path:
    training = DATA_DIR / "Training"
    if training.exists():
        return training
    if not ZIP_FILE.exists():
        raise FileNotFoundError(f"Dataset ZIP not found: {ZIP_FILE}")
    with zipfile.ZipFile(ZIP_FILE) as archive:
        archive.extractall(DATA_DIR)
    if not training.exists():
        raise RuntimeError("The ZIP does not contain the expected Training folder.")
    return training


def make_dataset(folder: Path, subset: str | None = None) -> tf.data.Dataset:
    kwargs = {
        "directory": folder,
        "class_names": CLASS_NAMES,
        "label_mode": "categorical",
        "image_size": IMAGE_SIZE,
        "batch_size": 32,
        "shuffle": subset is not None,
    }
    if subset:
        kwargs.update(validation_split=.20, subset=subset, seed=42)
    return tf.keras.utils.image_dataset_from_directory(**kwargs).prefetch(tf.data.AUTOTUNE)


def build_model() -> tuple[tf.keras.Model, tf.keras.Model]:
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(.08),
        tf.keras.layers.RandomTranslation(.06, .06),
        tf.keras.layers.RandomZoom(.10),
        tf.keras.layers.RandomContrast(.10),
    ])
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    features = augmentation(inputs)
    features = base(features, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    features = tf.keras.layers.Dropout(.35)(features)
    outputs = tf.keras.layers.Dense(4, activation="softmax")(features)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=.04),
        metrics=["accuracy"],
    )
    return model, base


def train_model() -> None:
    training = prepare_dataset()
    train = make_dataset(training,"training")
    validation = make_dataset(training,"validation")
    model, base = build_model()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=2, factor=.3
        ),
    ]

    # Stage 1: train the classification head while EfficientNet is frozen.
    model.fit(train, validation_data=validation, epochs=20, callbacks=callbacks)

    # Stage 2: fine-tune the last EfficientNet layers at a lower learning rate.
    base.trainable = True
    for layer in base.layers[:-120]: layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization): layer.trainable = False
    model.compile(tf.keras.optimizers.Adam(3e-5), tf.keras.losses.CategoricalCrossentropy(label_smoothing=.04), metrics=["accuracy"])
    model.fit(train, validation_data=validation, epochs=25, callbacks=callbacks)
    model.save(MODEL_FILE)
    LABEL_FILE.write_text(json.dumps(CLASS_NAMES),encoding="utf-8")
    metrics = {"architecture":MODEL_ARCHITECTURE,"version":MODEL_VERSION,"input_size":"224 × 224","epochs":"20 + up to 25 fine-tune","classes":4,"training_samples":5600,"validation_samples":1400,"testing_samples":1600}
    test = DATA_DIR / "Testing"
    if test.exists():
        ds = make_dataset(test)
        truth=np.concatenate([y.numpy().argmax(1) for _,y in ds])
        pred=model.predict(ds,verbose=0).argmax(1)
        metrics["accuracy"] = float(np.mean(pred==truth))
        cm=np.zeros((4,4),dtype=int)
        for a,b in zip(truth,pred): cm[a,b]+=1
        precision=np.divide(np.diag(cm),np.maximum(cm.sum(0),1))
        recall=np.divide(np.diag(cm),np.maximum(cm.sum(1),1))
        f1=2*precision*recall/np.maximum(precision+recall,1e-9)
        metrics.update(precision_macro=float(precision.mean()),recall_macro=float(recall.mean()),f1_macro=float(f1.mean()),confusion_matrix=cm.tolist(),roc_auc="Not recorded by this training workflow")
    METRICS_FILE.write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(f"Saved {MODEL_FILE}. Evaluation metrics: {METRICS_FILE}")


class NeuroScanApp:
    def __init__(self, model: tf.keras.Model, labels: list[str]) -> None:
        self.model, self.labels = model, labels
        self.path: Path | None = None; self.original: Image.Image | None = None; self.attention: Image.Image | None = None; self.overlay: Image.Image | None = None
        self.mode, self.zoom, self.pan_x, self.pan_y = "Original MRI", 1.0, 0, 0
        self.active_canvas: tk.Canvas | None = None; self.viewing_result = False; self._return_view = None
        self.brightness, self.contrast, self.drag = 1.0, 1.0, None
        self.scores: np.ndarray | None = None; self.quality = "Not checked"; self.summary = ""
        self.root = tk.Tk(); self.root.title("NeuroScan — AI-Assisted MRI Analysis"); self.root.configure(bg=C["bg"])
        try: self.root.state("zoomed")
        except tk.TclError: self.root.geometry("1440x860")
        self.root.minsize(1120, 680); self._styles(); self.show_terms()

    def _styles(self) -> None:
        style=ttk.Style(self.root); style.theme_use("clam")
        style.configure("Violet.TButton",background=C["violet"],foreground=C["text"],borderwidth=0,font=("Segoe UI",10,"bold"),padding=(12,8)); style.map("Violet.TButton",background=[("active",C["purple"]),("disabled",C["hover"])])
        style.configure("Ghost.TButton",background=C["card"],foreground=C["muted"],borderwidth=0,font=("Segoe UI",9),padding=(9,6)); style.map("Ghost.TButton",background=[("active",C["hover"])])
        style.configure("Dark.Horizontal.TProgressbar",troughcolor=C["hover"],background=C["violet"],thickness=6)

    def _label(self,p,text,size=10,fg=None,bg=None,**kwargs): return tk.Label(p,text=text,bg=bg or C["card"],fg=fg or C["text"],font=("Segoe UI",size),**kwargs)
    def _panel(self,p):
        outer=tk.Frame(p,bg="#0B0D0F",padx=1,pady=1); inner=tk.Frame(outer,bg=C["card"]); inner.pack(fill="both",expand=True); return outer,inner

    def _clear_root(self) -> None:
        for child in self.root.winfo_children(): child.destroy()

    def _header(self, step: int | None, title: str, subtitle: str) -> None:
        nav=tk.Frame(self.root,bg=C["surface"],height=68); nav.pack(fill="x"); nav.pack_propagate(False)
        tk.Label(nav,text="NeuroScan",bg=C["surface"],fg=C["text"],font=("Segoe UI",20,"bold")).pack(side="left",padx=(24,8),pady=13)
        tk.Label(nav,text=subtitle,bg=C["surface"],fg=C["muted"],font=("Segoe UI",10)).pack(side="left",pady=20)
        tk.Label(nav,text="●  MODEL READY",bg=C["surface"],fg=C["teal"],font=("Segoe UI",9,"bold")).pack(side="right",padx=(0,24))
        if step is not None:
            ttk.Button(nav,text="Model info",style="Ghost.TButton",command=self.show_model_info).pack(side="right",padx=(0,12),pady=17)
        steps=tk.Frame(self.root,bg=C["bg"],pady=13); steps.pack(fill="x")
        if step is None:
            ttk.Button(steps,text="←  Back",style="Ghost.TButton",command=self._go_back).pack(side="left",padx=(24,10))
            tk.Label(steps,text=title.upper(),bg=C["bg"],fg=C["violet2"],font=("Segoe UI",9,"bold")).pack(side="left")
            return
        for number, label in ((1,"Terms & conditions"),(2,"MRI analysis"),(3,"Diagnosis summary")):
            active=number == step; complete=number < step
            colour=C["teal"] if complete else C["violet2"] if active else C["dim"]
            marker="✓" if complete else str(number)
            tk.Label(steps,text=f" {marker} ",bg=colour,fg=C["bg"],font=("Segoe UI",9,"bold"),padx=4,pady=2).pack(side="left",padx=(24 if number==1 else 8,4))
            tk.Label(steps,text=label.upper(),bg=C["bg"],fg=colour,font=("Segoe UI",9,"bold")).pack(side="left")

    def _go_back(self) -> None:
        (self._return_view or self.show_terms)()

    def show_terms(self) -> None:
        self._return_view = self.show_terms
        self._clear_root(); self._header(1,"Terms & conditions","AI-Assisted MRI Analysis")
        body=tk.Frame(self.root,bg=C["bg"],padx=24,pady=12); body.pack(fill="both",expand=True)
        outer,card=self._panel(body); outer.place(relx=.5,rely=.46,anchor="center",relwidth=.68,relheight=.76)
        tk.Frame(card,bg=C["violet"],height=4).pack(fill="x")
        self._label(card,"\u26E8",22,C["violet2"]).pack(anchor="w",padx=34,pady=(24,0))
        self._label(card,"TERMS & CONDITIONS",10,C["violet2"]).pack(anchor="w",padx=34,pady=(12,5))
        self._label(card,"Before you begin",25).pack(anchor="w",padx=34)
        self._label(card,"Confirm the intended research-only use before entering the workspace.",11,C["muted"]).pack(anchor="w",padx=34,pady=(5,20))
        terms=(("\u2697","Research-use image classifier, not a diagnostic service."),
               ("\u2695","Results require review by a qualified healthcare professional."),
               ("\U0001F512","Images are processed locally on this computer."))
        rows=tk.Frame(card,bg=C["card"]); rows.pack(fill="x",padx=34,pady=(0,20))
        for icon,text in terms:
            row=tk.Frame(rows,bg=C["surface"]); row.pack(fill="x",pady=4)
            self._label(row,icon,13,C["teal"],bg=C["surface"]).pack(side="left",padx=(14,12),pady=11)
            self._label(row,text,11,C["text"],bg=C["surface"],wraplength=620,justify="left").pack(side="left",pady=11,padx=(0,14))
        self.accept_terms=tk.BooleanVar(value=False)
        check=tk.Checkbutton(card,text="I understand and agree to use this tool for research and review only.",variable=self.accept_terms,command=self._update_terms_button,bg=C["card"],fg=C["text"],selectcolor=C["hover"],activebackground=C["card"],activeforeground=C["text"],font=("Segoe UI",10))
        check.pack(anchor="w",padx=30,pady=(0,22))
        self.begin_btn=ttk.Button(card,text="Continue to MRI analysis  →",style="Violet.TButton",command=self.show_workspace,state="disabled")
        self.begin_btn.pack(anchor="e",padx=34,pady=(0,30))

    def _update_terms_button(self) -> None:
        self.begin_btn.configure(state="normal" if self.accept_terms.get() else "disabled")

    def show_workspace(self) -> None:
        self._return_view = self.show_workspace
        self._clear_root(); self._header(2,"MRI analysis","Upload and review an MRI image")
        main=tk.Frame(self.root,bg=C["bg"],padx=16,pady=16); main.pack(fill="both",expand=True)
        for col,w in enumerate((5,4)): main.columnconfigure(col,weight=w)
        main.rowconfigure(0,weight=1)
        a,self.left=self._panel(main); a.grid(row=0,column=0,sticky="nsew",padx=(0,7)); self._left_panel()
        b,self.center=self._panel(main); b.grid(row=0,column=1,sticky="nsew",padx=(7,0)); self._analysis_panel()

    def _left_panel(self) -> None:
        p=self.left; self._label(p,"MRI EXPLORER",9,C["violet2"]).pack(anchor="w",padx=16,pady=(14,2)); self._label(p,"Interactive image viewer",16,fg=C["text"]).pack(anchor="w",padx=16)
        self.viewer=tk.Canvas(p,bg=C["card"],highlightthickness=0,cursor="hand2"); self.viewer.pack(fill="both",expand=True,padx=16,pady=14)

        def draw_dropzone() -> None:
            self.viewer.delete("all")
            width,height=max(2,self.viewer.winfo_width()),max(2,self.viewer.winfo_height())
            self.viewer.create_rectangle(1,1,width-2,height-2,outline="#3A3F45",width=2,dash=(5,3))
            cx,cy=width//2,height//2
            self.viewer.create_oval(cx-20,cy-50,cx+20,cy-10,fill=C["hover"],outline="")
            self.viewer.create_text(cx,cy-30,text="⇧",fill=C["muted"],font=("Segoe UI",18,"bold"))
            self.viewer.create_text(cx,cy+7,text="Click to choose an MRI image",fill=C["text"],font=("Segoe UI",11,"bold"))
            self.viewer.create_text(cx,cy+30,text="PNG · JPG · JPEG",fill=C["dim"],font=("Segoe UI",9))

        def begin_drag(event) -> None:
            if self.original is None: self.open_image()
            else: self.pan_start(event)

        self.viewer.bind("<MouseWheel>",self.wheel_zoom); self.viewer.bind("<ButtonPress-1>",begin_drag); self.viewer.bind("<B1-Motion>",self.pan_move)
        self.viewer.bind("<Configure>",lambda _e:self.render_views() if self.original else draw_dropzone())
        self.viewer.bind("<Enter>",lambda _e:self.viewer.configure(cursor="fleur" if self.original else "hand2"))
        self.active_canvas=self.viewer; self.viewing_result=False
        self.bright_var=tk.DoubleVar(value=1); self.contrast_var=tk.DoubleVar(value=1)
        tools=tk.Frame(p,bg=C["card"]); tools.pack(fill="x",padx=16,pady=(10,10))

        def toolbar_button(text,cmd,accent=False) -> None:
            button=tk.Button(tools,text=text,command=cmd,bg=C["violet"] if accent else C["hover"],fg=C["text"] if accent else C["muted"],activebackground=C["purple"] if accent else "#3A3F45",activeforeground=C["text"],borderwidth=0,relief="flat",font=("Segoe UI",9,"bold" if accent else "normal"),cursor="hand2",padx=8,pady=7)
            button.pack(side="left",fill="x",expand=True,padx=(0,6) if text!="Fit" else 0)

        toolbar_button("Open image",self.open_image,accent=True)
        toolbar_button("Zoom +",lambda:self.change_zoom(1.2))
        toolbar_button("Zoom −",lambda:self.change_zoom(1/1.2))
        toolbar_button("Fit",self.fit_image)

    def _analysis_panel(self) -> None:
        p=self.center; self._label(p,"AI EXPLAINABILITY",9,C["violet2"]).pack(anchor="w",padx=16,pady=(14,2)); self._label(p,"Review the analysis",16).pack(anchor="w",padx=16)
        self.opacity_var=tk.DoubleVar(value=.55)
        self.explanation_text=self._label(p,"Switch between original MRI, AI attention, and overlay views after processing.",10,C["muted"],wraplength=420,justify="left"); self.explanation_text.pack(anchor="w",padx=16,pady=(8,18))
        self.pipeline=[]
        for i,item in enumerate(("MRI loaded","Image quality checked","AI classification completed","Explainability map generated")):
            if i>0: tk.Frame(p,bg=C["hover"],width=1,height=10).pack(anchor="w",padx=(19,0))
            line=self._label(p,"○  "+item,9,C["dim"]); line.pack(anchor="w",padx=16,pady=1); self.pipeline.append(line)
        self.status=self._label(p,"READY TO ANALYZE",9,C["teal"]); self.status.pack(anchor="w",padx=16,pady=(16,0))
        self.result=self._label(p,"Choose an MRI image to begin",13,C["muted"],wraplength=390,justify="left"); self.result.pack(anchor="w",padx=16,pady=(5,0))
        self.confidence=self._label(p,"",9,C["muted"],wraplength=390,justify="left"); self.confidence.pack(anchor="w",padx=16,pady=(3,10))
        self.continue_btn=ttk.Button(p,text="Continue to diagnosis summary  →",style="Violet.TButton",command=self.open_pending_diagnosis,state="disabled")
        self.continue_btn.pack(fill="x",side="bottom",padx=16,pady=16)

    def open_image(self) -> None:
        path=filedialog.askopenfilename(title="Choose an MRI image",filetypes=[("MRI images","*.png *.jpg *.jpeg")])
        if not path:
            return
        try:
            image=self._load_mri_image(path)
        except UnidentifiedImageError:
            messagebox.showerror("Unable to open image","This file isn't a readable image. It may be corrupted, or not actually a PNG/JPG despite its file extension.")
            return
        except OSError as err:
            messagebox.showerror("Unable to open image",f"The file could not be read from disk.\n\nTechnical details: {type(err).__name__}: {err}")
            return
        except ValueError as err:
            messagebox.showerror("Unable to open image",str(err))
            return

        self.path=Path(path)
        self.original=image
        self.attention=self.overlay=None
        self.fit_image()
        self.start_analysis()

    def _load_mri_image(self,path:str) -> Image.Image:
        """Open and validate a user-selected MRI image."""
        image=Image.open(path)
        image.load()
        if image.format not in {"PNG","JPEG"}:
            raise ValueError("Only PNG, JPG, and JPEG files are supported.")
        if min(image.size)<32:
            raise ValueError("This image is too small. Choose an image at least 32 × 32 pixels.")
        return image.convert("RGB")

    def quality_check(self) -> tuple[str,str]:
        grayscale=np.asarray(self.original.convert("L"),dtype=np.float32)/255
        brightness=grayscale.mean()
        contrast=grayscale.std()
        sharpness=(
            np.mean(np.abs(np.diff(grayscale,axis=0)))
            + np.mean(np.abs(np.diff(grayscale,axis=1)))
        )
        concerns=[]
        if not .12<brightness<.88:
            concerns.append("brightness")
        if contrast<.06:
            concerns.append("contrast")
        if sharpness<.015:
            concerns.append("sharpness")
        if concerns:
            return "Needs attention", "Low " + ", ".join(concerns) + " may reduce prediction reliability."
        return "Good", "Resolution, brightness, contrast, and sharpness are within simple local screening ranges."

    def start_analysis(self) -> None:
        self._show_analysis_progress()
        threading.Thread(target=self.worker,daemon=True).start()

    def _show_analysis_progress(self) -> None:
        """Update the workspace before the background model call begins."""
        self.status.configure(text="ANALYZING…",fg=C["violet2"])
        self.result.configure(text="Analyzing image…")
        self.confidence.configure(text="Local processing in progress.")
        self.continue_btn.configure(state="disabled")
        for i,line in enumerate(self.pipeline): line.configure(text=("●  " if i==0 else "○  ")+line.cget("text")[3:],fg=C["violet2"] if i==0 else C["dim"])

    def worker(self) -> None:
        try:
            self.quality=self.quality_check()
            array=np.asarray(self.original.resize(IMAGE_SIZE),dtype=np.float32)
            scores=self.model.predict(np.expand_dims(array,0),verbose=0)[0]
        except Exception as err:
            traceback.print_exc(); self.root.after(0,lambda:self.analysis_error(err)); return
        try:
            cam=self.gradcam(array,int(np.argmax(scores)))
        except Exception:
            traceback.print_exc(); cam=None
        self.root.after(0,lambda:self.finish_analysis(scores,cam))

    def gradcam(self,array:np.ndarray,index:int) -> Image.Image | None:
        """Create a Grad-CAM attention image for the selected class."""
        try:
            augmentation=self.model.get_layer("sequential")
            base=self.model.get_layer("efficientnetb0")
            pooling=self.model.get_layer("global_average_pooling2d")
            classifier=self.model.get_layer("dense")
        except ValueError:
            return None

        image_batch=tf.convert_to_tensor(np.expand_dims(array,0))
        with tf.GradientTape() as tape:
            convolution_output=base(augmentation(image_batch,training=False),training=False)
            tape.watch(convolution_output)
            probabilities=classifier(pooling(convolution_output))
            target_probability=probabilities[:,index]
        gradients=tape.gradient(target_probability,convolution_output)
        if gradients is None:
            return None

        weights=tf.reduce_mean(gradients,axis=(0,1,2))
        heatmap=tf.reduce_sum(convolution_output[0]*weights,axis=-1)
        heatmap=tf.maximum(heatmap,0)
        heatmap=heatmap/(tf.reduce_max(heatmap)+1e-8)
        heat=np.asarray(
            Image.fromarray(np.uint8(heatmap.numpy()*255)).resize(
                self.original.size,Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )/255
        # Scientific violet-to-coral heatmap; it reflects influence, not a tumour boundary.
        rgb=np.zeros((*heat.shape,3),dtype=np.uint8)
        rgb[...,0]=(80+175*heat).astype(np.uint8)
        rgb[...,1]=(25+75*heat).astype(np.uint8)
        rgb[...,2]=(150+80*(1-heat)).astype(np.uint8)
        return Image.fromarray(rgb)

    def _classification_status(self,label:str,top_score:float,separation:float) -> tuple[str,str,str,str]:
        """Return the user-facing status treatment for a model result."""
        if top_score<LOW_CONFIDENCE_THRESHOLD or separation<LOW_SEPARATION_THRESHOLD:
            return "LOW CONFIDENCE RESULT",C["amber"],C["amber_bg"],"Lower confidence result. The probability distribution is available in the summary."
        if label=="notumor":
            return "NO TUMOUR CLASS SELECTED",C["teal"],C["teal_bg"],"No-tumour class selected. This is a model classification, not a clinical finding."
        return "PROFESSIONAL REVIEW REQUIRED",C["coral"],C["coral_bg"],"Classification output is ready to view with image quality and probability details."

    def _reliability_details(self,separation:float) -> tuple[str,str]:
        if separation>=LOW_SEPARATION_THRESHOLD:
            return "HIGH","Clear separation was observed between the highest predicted class and alternative classifications."
        return "LIMITED","The highest predicted classes have similar probabilities. Professional review is particularly important."

    def _mark_pipeline_complete(self) -> None:
        for line in self.pipeline:
            line.configure(text="✓  "+line.cget("text")[3:],fg=C["teal"])

    def finish_analysis(self,scores:np.ndarray,cam:Image.Image|None) -> None:
        self.scores,self.attention=scores,cam
        top=int(np.argmax(scores))
        second,first=np.sort(scores)[-2:]
        separation=float(first-second)
        label=self.labels[top]
        quality,detail=self.quality
        status,colour,bg,note=self._classification_status(label,float(scores[top]),separation)
        reliability,reliability_detail=self._reliability_details(separation)

        self.make_overlay()
        self.summary=self.make_summary(label,separation)
        self._mark_pipeline_complete()
        self._show_analysis_complete()
        self.pending_diagnosis=(label,status,colour,bg,note,quality,detail,reliability,reliability_detail)
        self.continue_btn.configure(state="normal")

    def _show_analysis_complete(self) -> None:
        """Update the workspace after the background analysis has completed."""
        self.status.configure(text="ANALYSIS READY",fg=C["teal"])
        self.result.configure(text="Analysis is ready for review")
        self.confidence.configure(text="Your MRI image has been processed locally. Continue when you are ready to view the separate diagnosis summary.")
        self.explanation_text.configure(text="The local AI analysis is complete. Continue to the diagnosis summary to explore the AI attention overlay.")

    def open_pending_diagnosis(self) -> None:
        if hasattr(self,"pending_diagnosis"):
            self.show_diagnosis(*self.pending_diagnosis)

    def _section_divider(self,parent) -> None:
        tk.Frame(parent,bg=C["hover"],height=1).pack(fill="x",padx=18,pady=(10,8))

    def show_diagnosis(self,label:str,status:str,colour:str,bg:str,note:str,quality:str,detail:str,reliability:str,reliability_detail:str) -> None:
        self._return_view = lambda: self.show_diagnosis(label,status,colour,bg,note,quality,detail,reliability,reliability_detail)
        self._clear_root(); self._header(3,"Diagnosis summary","Review AI classification output")
        self.viewing_result=True; self.mode="Original MRI"
        main=tk.Frame(self.root,bg=C["bg"],padx=16,pady=16); main.pack(fill="both",expand=True)
        main.columnconfigure(0,weight=5); main.columnconfigure(1,weight=4); main.rowconfigure(0,weight=1)
        self._build_diagnosis_viewer(main)
        self._build_diagnosis_summary(main,label,status,colour,bg,note,quality,detail,reliability,reliability_detail)

    def _build_diagnosis_viewer(self,main:tk.Frame) -> None:
        """Create the left-side MRI viewer and its image controls."""
        a,left=self._panel(main); a.grid(row=0,column=0,sticky="nsew",padx=(0,7))
        self._label(left,"MRI & AI ATTENTION",9,C["violet2"]).pack(anchor="w",padx=16,pady=(14,2)); self._label(left,"Image review",16).pack(anchor="w",padx=16)
        self.result_view=tk.Canvas(left,bg="#0A0C0E",highlightthickness=0,cursor="fleur"); self.result_view.pack(fill="both",expand=True,padx=16,pady=14)
        self.result_view.bind("<Configure>",lambda _e:self.render_views()); self.result_view.bind("<MouseWheel>",self.wheel_zoom); self.result_view.bind("<ButtonPress-1>",self.pan_start); self.result_view.bind("<B1-Motion>",self.pan_move)
        tab_row=tk.Frame(left,bg=C["card"]); tab_row.pack(fill="x",padx=16,pady=(0,8))
        self.tab_labels={}
        for name in ("Original MRI","AI Attention","Overlay"):
            lbl=tk.Label(tab_row,text=name.upper(),bg=C["hover"],fg=C["muted"],font=("Segoe UI",9,"bold"),padx=10,pady=6,cursor="hand2")
            lbl.pack(side="left",padx=(0,4)); lbl.bind("<Button-1>",lambda _e,m=name:self.set_tab(m)); self.tab_labels[name]=lbl
        self._refresh_tabs()
        result_tools=tk.Frame(left,bg=C["card"]); result_tools.pack(fill="x",padx=16,pady=(0,8))
        for text,cmd in [("Zoom +",lambda:self.change_zoom(1.2)),("Zoom −",lambda:self.change_zoom(1/1.2)),("Fit",self.fit_image),("Reset view",self.reset_view),("Brightness",self.nudge_brightness),("Contrast",self.nudge_contrast),("Fullscreen",self.toggle_fullscreen)]:
            ttk.Button(result_tools,text=text,style="Ghost.TButton",command=cmd).pack(side="left",padx=(0,4),pady=(0,2))
        self.active_canvas=self.result_view; self.root.after(50,self.fit_image)
        self._label(left,"Switch tabs to compare the original MRI, the AI attention map, and the overlay. Scroll to zoom, drag to pan — this is prediction influence, not a tumour outline or clinical finding.",9,C["muted"],wraplength=560,justify="left").pack(anchor="w",padx=16,pady=(0,16))

    def _build_diagnosis_summary(self,main:tk.Frame,label:str,status:str,colour:str,bg:str,note:str,quality:str,detail:str,reliability:str,reliability_detail:str) -> None:
        """Create the right-side classification summary and its actions."""
        b,right=self._panel(main); b.grid(row=0,column=1,sticky="nsew",padx=(7,0))
        scroll_outer=tk.Frame(right,bg=C["card"]); scroll_outer.pack(fill="both",expand=True)

        self._label(scroll_outer,"AI CLASSIFICATION RESULT",9,C["violet2"]).pack(anchor="w",padx=18,pady=(14,4))
        self.status=self._label(scroll_outer,status,9,colour,bg=bg,padx=10,pady=5); self.status.pack(anchor="w",padx=18)
        self.result=self._label(scroll_outer,display_name(label),23,wraplength=410,justify="left"); self.result.pack(anchor="w",padx=18,pady=(10,6))

        self._label(scroll_outer,"HIGHEST PREDICTED PROBABILITY",9,C["muted"]).pack(anchor="w",padx=18)
        self._label(scroll_outer,f"{float(self.scores.max()):.1%}",20,C["text"]).pack(anchor="w",padx=18,pady=(2,4))
        self.confidence=self._label(scroll_outer,f"Model confidence reflects the model's relative prediction certainty and is not clinical certainty.\n{note}",9,C["muted"],wraplength=410,justify="left"); self.confidence.pack(anchor="w",padx=18)

        self._section_divider(scroll_outer)
        self._label(scroll_outer,"AI ANALYSIS SUMMARY",9,C["muted"]).pack(anchor="w",padx=18)
        self._label(scroll_outer,summary_text(label),10,C["text"],wraplength=410,justify="left").pack(anchor="w",padx=18,pady=(4,8))

        self._label(scroll_outer,"WHY THIS PREDICTION?",9,C["muted"]).pack(anchor="w",padx=18)
        self._label(scroll_outer,"Explore the image regions that contributed most strongly to the AI classification.",9,C["text"],wraplength=410,justify="left").pack(anchor="w",padx=18,pady=(4,6))
        ttk.Button(scroll_outer,text="View AI attention",style="Ghost.TButton",command=lambda:self.set_tab("AI Attention")).pack(anchor="w",padx=18)

        self._section_divider(scroll_outer)
        info_row=tk.Frame(scroll_outer,bg=C["card"]); info_row.pack(fill="x",padx=18)
        info_row.columnconfigure(0,weight=1); info_row.columnconfigure(1,weight=1)
        col_quality=tk.Frame(info_row,bg=C["card"]); col_quality.grid(row=0,column=0,sticky="nw",padx=(0,10))
        col_reliability=tk.Frame(info_row,bg=C["card"]); col_reliability.grid(row=0,column=1,sticky="nw")
        quality_colour=C["teal"] if quality=="Good" else C["amber"]
        self._label(col_quality,"IMAGE QUALITY",9,C["muted"]).pack(anchor="w")
        self._label(col_quality,quality.upper(),13,quality_colour).pack(anchor="w",pady=(2,4))
        self._label(col_quality,detail,9,C["muted"],wraplength=190,justify="left").pack(anchor="w")
        reliability_colour=C["teal"] if reliability=="HIGH" else C["amber"]
        self._label(col_reliability,"PREDICTION RELIABILITY",9,C["muted"]).pack(anchor="w")
        self._label(col_reliability,reliability,13,reliability_colour).pack(anchor="w",pady=(2,4))
        self._label(col_reliability,reliability_detail,9,C["muted"],wraplength=190,justify="left").pack(anchor="w")

        self._section_divider(scroll_outer)
        self._label(scroll_outer,"CLASS PROBABILITY DISTRIBUTION",9,C["muted"]).pack(anchor="w",padx=18,pady=(0,6))
        self.prob_box=tk.Frame(scroll_outer,bg=C["card"]); self.prob_box.pack(fill="x",padx=18)
        self.show_probabilities()

        actions=tk.Frame(scroll_outer,bg=C["card"]); actions.pack(fill="x",side="bottom",padx=18,pady=14)
        ttk.Button(actions,text="Generate AI analysis report",style="Violet.TButton",command=self.generate_report).pack(side="left",fill="x",expand=True,padx=(0,8))
        ttk.Button(actions,text="Analyze another image",style="Ghost.TButton",command=self.show_workspace).pack(side="left",fill="x",expand=True)

    def analysis_error(self,err:Exception) -> None:
        kind=type(err).__name__
        if isinstance(err,MemoryError): headline,suggestion="Ran out of memory while analyzing this image.","Try closing other applications, or use a smaller image."
        elif isinstance(err,FileNotFoundError): headline,suggestion="A required file could not be found.","Make sure brain_tumor_model.keras and class_names.json are in the app folder."
        else: headline,suggestion="An unexpected error occurred during analysis.","Try a different image. Full technical details were also printed to the terminal window."
        self.status.configure(text="ANALYSIS COULD NOT COMPLETE",fg=C["coral"]); self.result.configure(text=headline); self.confidence.configure(text=f"{suggestion}\n\nTechnical details: {kind}: {err}")

    def show_probabilities(self) -> None:
        for w in self.prob_box.winfo_children(): w.destroy()
        top=int(np.argmax(self.scores))
        for i,(name,score) in enumerate(zip(self.labels,self.scores)):
            short_name={"glioma":"Glioma","meningioma":"Meningioma","pituitary":"Pituitary","notumor":"No tumour"}[name]
            row=tk.Frame(self.prob_box,bg=C["card"]); row.pack(fill="x",pady=3); tk.Label(row,text=short_name,bg=C["card"],fg=C["muted"],width=20,anchor="w",font=("Segoe UI",9)).pack(side="left")
            track=tk.Frame(row,bg=C["hover"],height=8); track.pack(side="left",fill="x",expand=True,padx=6); tk.Frame(track,bg=C["violet"] if i==top else C["dim"],height=8).place(relwidth=max(.01,float(score)),relheight=1)
            tk.Label(row,text=f"{float(score):.1%}",bg=C["card"],fg=C["text"],width=6,anchor="e",font=("Segoe UI",9,"bold")).pack(side="right")

    def make_overlay(self) -> None:
        if self.attention: self.overlay=Image.blend(self.original,self.attention,float(self.opacity_var.get()))

    def set_mode(self,mode:str) -> None: self.mode=mode; self.render_views()
    def current_base(self) -> Image.Image | None:
        if not self.viewing_result: return self.original
        if self.mode=="AI Attention": return self.attention or self.original
        if self.mode=="Overlay": return self.overlay or self.original
        return self.original
    def set_tab(self,mode:str) -> None:
        if mode=="AI Attention" and self.attention is None: return
        if mode=="Overlay" and self.overlay is None: return
        self.set_mode(mode); self._refresh_tabs()
    def _refresh_tabs(self) -> None:
        if not hasattr(self,"tab_labels"): return
        for name,lbl in self.tab_labels.items():
            active=name==self.mode; lbl.configure(bg=C["violet"] if active else C["hover"],fg=C["text"] if active else C["muted"])
    def render_views(self) -> None:
        base_image=self.current_base()
        if not base_image or not self.active_canvas: return
        base=ImageEnhance.Contrast(ImageEnhance.Brightness(base_image).enhance(self.bright_var.get())).enhance(self.contrast_var.get())
        self._draw(self.active_canvas,base)
    def _draw(self,canvas:tk.Canvas,image) -> None:
        w,h=image.size; resized=image.resize((max(1,int(w*self.zoom)),max(1,int(h*self.zoom)))); photo=ImageTk.PhotoImage(resized)
        canvas.delete("all"); cw,ch=max(1,canvas.winfo_width()),max(1,canvas.winfo_height()); canvas.create_image(cw//2+self.pan_x,ch//2+self.pan_y,image=photo); canvas.image=photo
    def fit_image(self) -> None:
        self.pan_x=self.pan_y=0; base_image=self.current_base()
        if base_image and self.active_canvas:
            cw,ch=self.active_canvas.winfo_width(),self.active_canvas.winfo_height(); iw,ih=base_image.size
            if cw>2 and ch>2: self.zoom=min((cw-28)/iw,(ch-28)/ih,1.0)
            else: self.zoom=1.0
        else: self.zoom=1.0
        self.render_views()
    def toggle_fullscreen(self) -> None:
        enabled = not bool(self.root.attributes("-fullscreen"))
        self.root.attributes("-fullscreen", enabled)
        if enabled: self.root.bind("<Escape>", lambda _event: self.root.attributes("-fullscreen", False))
    def reset_view(self) -> None:
        self.bright_var.set(1); self.contrast_var.set(1); self.fit_image()
    def nudge_brightness(self) -> None:
        value=self.bright_var.get()+.1; self.bright_var.set(.8 if value>1.4 else value); self.render_views()
    def nudge_contrast(self) -> None:
        value=self.contrast_var.get()+.1; self.contrast_var.set(.8 if value>1.5 else value); self.render_views()
    def change_zoom(self,factor:float) -> None: self.zoom=min(5,max(.2,self.zoom*factor)); self.render_views()
    def wheel_zoom(self,event) -> None: self.change_zoom(1.12 if event.delta>0 else .89)
    def pan_start(self,event) -> None: self.drag=(event.x,event.y)
    def pan_move(self,event) -> None:
        if self.drag: self.pan_x+=event.x-self.drag[0]; self.pan_y+=event.y-self.drag[1]; self.drag=(event.x,event.y); self.render_views()
    def make_summary(self,label:str,separation:float)->str:
        parts = [
            "NeuroScan — AI-Assisted MRI Analysis Report",
            f"Analysis ID: NS-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            f"Date: {datetime.now():%Y-%m-%d %H:%M:%S}",
            f"File: {self.path.name}",
            f"Model: {MODEL_ARCHITECTURE} {MODEL_VERSION}",
            f"Classification: {display_name(label)}",
            f"Top probability: {float(self.scores.max()):.1%}",
            f"Top-two separation: {separation:.1%}",
            f"Image quality: {self.quality[0]}",
            "Class probabilities:",
        ]
        parts.extend(f"- {display_name(name)}: {float(score):.1%}" for name,score in zip(self.labels,self.scores))
        parts.extend([DISCLAIMER,"AI attention maps show prediction influence only; they do not identify tumour boundaries."])
        return "\n".join(parts)

    def _encode_report_image(self,image:Image.Image) -> str:
        buffer=io.BytesIO()
        image.save(buffer,format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def _report_html(self,summary_html:str,mri:str,overlay:str) -> str:
        return f"<!doctype html><html><head><meta charset='utf-8'><title>NeuroScan AI-Assisted MRI Analysis Report</title><style>body{{font:16px Segoe UI;background:#111315;color:#f5f5f4;max-width:950px;margin:auto;padding:35px}}h1{{color:#a78bfa}}.card{{background:#22262a;padding:20px;margin:18px 0}}img{{max-width:46%;margin-right:2%;background:#000}}</style></head><body><h1>NeuroScan — AI-Assisted MRI Analysis Report</h1><div class='card'>{summary_html}</div><div class='card'><h2>MRI preview and AI attention overlay</h2><img src='data:image/png;base64,{mri}'><img src='data:image/png;base64,{overlay}'><p>{DISCLAIMER}</p></div></body></html>"

    def generate_report(self) -> None:
        report_dir=ROOT/"reports"
        report_dir.mkdir(exist_ok=True)
        target=report_dir/f"neuroscan_report_{datetime.now():%Y%m%d_%H%M%S}.html"
        mri=self._encode_report_image(self.original)
        overlay=self._encode_report_image(self.overlay or self.original)
        summary_html="<br>".join(self.summary.replace("&","&amp;").replace("<","&lt;").splitlines())
        target.write_text(self._report_html(summary_html,mri,overlay),encoding="utf-8")
        messagebox.showinfo("Report generated",f"Saved local report:\n{target}\n\nIt has not been uploaded or sent anywhere.")
    def model_performance(self) -> None:
        self.show_model_info()

    def show_model_info(self) -> None:
        self._clear_root(); self._header(None,"Model information","Architecture, dataset, and evaluation metrics")
        data = load_metrics()
        short={"glioma":"Glioma","meningioma":"Meningioma","pituitary":"Pituitary","notumor":"No tumour"}
        main=tk.Frame(self.root,bg=C["bg"],padx=16,pady=16); main.pack(fill="both",expand=True)
        for col in (0,1,2): main.columnconfigure(col,weight=1)
        main.rowconfigure(1,weight=1)

        def card(col,title):
            pad=(0 if col==0 else 7, 0 if col==2 else 7)
            outer,inner=self._panel(main); outer.grid(row=0,column=col,sticky="nsew",padx=pad,pady=(0,14))
            self._label(inner,title,9,C["violet2"]).pack(anchor="w",padx=16,pady=(14,8))
            return inner

        def stat_row(parent,label,value,fg=None):
            row=tk.Frame(parent,bg=C["card"]); row.pack(fill="x",padx=16,pady=3)
            self._label(row,label,10,C["muted"]).pack(side="left")
            self._label(row,value,10,fg or C["text"]).pack(side="right")

        overview=card(0,"OVERVIEW")
        stat_row(overview,"Architecture",MODEL_ARCHITECTURE)
        stat_row(overview,"Version",MODEL_VERSION)
        stat_row(overview,"Input size",str((data or {}).get("input_size","224 × 224")))
        stat_row(overview,"Classes",str((data or {}).get("classes",len(CLASS_NAMES))))
        stat_row(overview,"Epochs",str((data or {}).get("epochs","20 + up to 25 fine-tune")))
        tk.Frame(overview,bg=C["card"],height=14).pack()

        dataset=card(1,"DATASET")
        if data and any(k in data for k in ("training_samples","validation_samples","testing_samples")):
            for key,label in (("training_samples","Training"),("validation_samples","Validation"),("testing_samples","Testing")):
                if key in data: stat_row(dataset,label,f"{data[key]:,}")
            tk.Frame(dataset,bg=C["card"],height=14).pack()
        else:
            self._label(dataset,"No metrics recorded yet.\nRun the app with --train to generate model_metrics.json.",10,C["muted"],wraplength=260,justify="left").pack(anchor="w",padx=16,pady=(0,16))

        perf=card(2,"PERFORMANCE")
        if data:
            for key,label in (("accuracy","Accuracy"),("precision_macro","Precision (macro)"),("recall_macro","Recall (macro)"),("f1_macro","F1 (macro)")):
                if key in data and isinstance(data[key],(int,float)): stat_row(perf,label,f"{data[key]:.1%}",C["teal"])
            roc=data.get("roc_auc")
            if roc: stat_row(perf,"ROC AUC",str(roc),C["dim"])
            tk.Frame(perf,bg=C["card"],height=14).pack()
        else:
            self._label(perf,"Not available.",10,C["muted"]).pack(anchor="w",padx=16,pady=(0,16))

        bottom_outer,bottom=self._panel(main); bottom_outer.grid(row=1,column=0,columnspan=3,sticky="nsew")
        self._label(bottom,"CONFUSION MATRIX · PER-CLASS ACCURACY",9,C["violet2"]).pack(anchor="w",padx=18,pady=(16,4))
        cm=(data or {}).get("confusion_matrix")
        if cm and len(cm)==len(self.labels):
            body=tk.Frame(bottom,bg=C["card"]); body.pack(fill="both",expand=True,padx=18,pady=(4,4))
            grid=tk.Frame(body,bg=C["card"]); grid.pack(side="left",padx=(0,32))
            tk.Label(grid,text="",bg=C["card"],width=11).grid(row=0,column=0)
            for j,name in enumerate(self.labels):
                tk.Label(grid,text=short.get(name,name),bg=C["card"],fg=C["muted"],font=("Segoe UI",8,"bold"),width=9).grid(row=0,column=j+1,pady=(0,4))
            for i,row_name in enumerate(self.labels):
                tk.Label(grid,text=short.get(row_name,row_name),bg=C["card"],fg=C["muted"],font=("Segoe UI",8,"bold"),anchor="w",width=11).grid(row=i+1,column=0,sticky="w")
                for j,value in enumerate(cm[i]):
                    correct=i==j
                    tk.Label(grid,text=str(value),bg=C["violet"] if correct else C["hover"],fg=C["text"] if correct else C["muted"],font=("Segoe UI",9,"bold" if correct else "normal"),width=9,pady=6).grid(row=i+1,column=j+1,padx=2,pady=2)
            tk.Label(grid,text="Rows = true class · Columns = predicted class",bg=C["card"],fg=C["dim"],font=("Segoe UI",8)).grid(row=len(self.labels)+1,column=0,columnspan=len(self.labels)+1,sticky="w",pady=(8,0))
            perclass=tk.Frame(body,bg=C["card"]); perclass.pack(side="left",anchor="n")
            self._label(perclass,"Per-class accuracy",9,C["muted"]).pack(anchor="w",pady=(0,6))
            for i,name in enumerate(self.labels):
                row_total=max(sum(cm[i]),1); acc=cm[i][i]/row_total
                acc_colour=C["teal"] if acc>=.9 else C["amber"] if acc>=.8 else C["coral"]
                row=tk.Frame(perclass,bg=C["card"]); row.pack(fill="x",pady=3,anchor="w")
                self._label(row,short.get(name,name),9,C["text"]).pack(side="left")
                self._label(row,f"{acc:.1%}",9,acc_colour).pack(side="left",padx=(10,0))
        else:
            self._label(bottom,"Confusion matrix not available yet. Run the app with --train to generate model_metrics.json after held-out testing.",10,C["muted"],wraplength=1000,justify="left").pack(anchor="w",padx=18,pady=(0,4))
        self._label(bottom,"These are self-reported test-set metrics from the model's last training run. "+DISCLAIMER,8,C["dim"],wraplength=1100,justify="left").pack(anchor="w",padx=18,pady=(14,18))

    def show_help(self) -> None:
        messagebox.showinfo("About NeuroScan",f"NeuroScan is a local research-use image classifier, not a medical device.\n\nSupported formats: PNG, JPG, JPEG\nSupported classes: glioma-like, meningioma-like, pituitary tumour-like, no-tumour.\n\nA single 2D image cannot provide a complete MRI volume or clinical context. This app does not segment, locate, or outline tumours.\n\nPrivacy: images are processed locally, never uploaded, and not stored by NeuroScan.\n\nModel: {MODEL_ARCHITECTURE} {MODEL_VERSION}\nFile: {MODEL_FILE.name}")
    def run(self)->None: self.root.mainloop()


def open_app() -> None:
    if not MODEL_FILE.exists() or not LABEL_FILE.exists(): print("No trained model exists. Run: python brain_tumor_detector.py --train"); sys.exit(1)
    model = tf.keras.models.load_model(MODEL_FILE)
    model.predict(np.zeros((1,*IMAGE_SIZE,3),dtype=np.float32),verbose=0)  # warm up on the main thread
    NeuroScanApp(model,json.loads(LABEL_FILE.read_text(encoding="utf-8"))).run()

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--train",action="store_true"); args=parser.parse_args(); train_model() if args.train else open_app()
