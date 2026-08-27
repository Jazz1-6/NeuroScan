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
DISCLAIMER = "Research-use classifier only. This result is not a diagnosis and must be reviewed by a qualified healthcare professional."
C = {"bg":"#111315", "surface":"#191C1F", "card":"#22262A", "hover":"#2A2F34", "violet":"#8B5CF6", "violet2":"#A78BFA", "purple":"#6D28D9", "teal":"#2DD4BF", "amber":"#F59E0B", "amber_bg":"#3A2A12", "coral":"#FB7185", "coral_bg":"#3A1820", "text":"#F5F5F4", "muted":"#A8B0B8", "dim":"#6B7280"}


def display_name(label: str) -> str:
    return {"glioma":"Glioma-like pattern", "meningioma":"Meningioma-like pattern", "pituitary":"Pituitary tumour-like pattern", "notumor":"No-tumour class selected"}.get(label, label.title())


def prepare_dataset() -> Path:
    training = DATA_DIR / "Training"
    if training.exists(): return training
    if not ZIP_FILE.exists(): raise FileNotFoundError(f"Dataset ZIP not found: {ZIP_FILE}")
    with zipfile.ZipFile(ZIP_FILE) as archive: archive.extractall(DATA_DIR)
    if not training.exists(): raise RuntimeError("The ZIP does not contain the expected Training folder.")
    return training


def make_dataset(folder: Path, subset: str | None = None) -> tf.data.Dataset:
    kwargs = dict(directory=folder, class_names=CLASS_NAMES, label_mode="categorical", image_size=IMAGE_SIZE, batch_size=32, shuffle=subset is not None)
    if subset: kwargs.update(validation_split=.20, subset=subset, seed=42)
    return tf.keras.utils.image_dataset_from_directory(**kwargs).prefetch(tf.data.AUTOTUNE)


def build_model() -> tuple[tf.keras.Model, tf.keras.Model]:
    augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(.08), tf.keras.layers.RandomTranslation(.06,.06), tf.keras.layers.RandomZoom(.10), tf.keras.layers.RandomContrast(.10)])
    base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE,3)); base.trainable = False
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE,3)); x = augmentation(inputs); x = base(x, training=False); x = tf.keras.layers.GlobalAveragePooling2D()(x); x = tf.keras.layers.Dropout(.35)(x); outputs = tf.keras.layers.Dense(4, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs); model.compile(tf.keras.optimizers.Adam(1e-3), tf.keras.losses.CategoricalCrossentropy(label_smoothing=.04), metrics=["accuracy"])
    return model, base


def train_model() -> None:
    training = prepare_dataset(); train, validation = make_dataset(training,"training"), make_dataset(training,"validation")
    model, base = build_model(); callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",patience=4,restore_best_weights=True), tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",patience=2,factor=.3)]
    model.fit(train, validation_data=validation, epochs=20, callbacks=callbacks)
    base.trainable = True
    for layer in base.layers[:-120]: layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization): layer.trainable = False
    model.compile(tf.keras.optimizers.Adam(3e-5), tf.keras.losses.CategoricalCrossentropy(label_smoothing=.04), metrics=["accuracy"])
    model.fit(train, validation_data=validation, epochs=25, callbacks=callbacks)
    model.save(MODEL_FILE); LABEL_FILE.write_text(json.dumps(CLASS_NAMES),encoding="utf-8")
    metrics = {"architecture":MODEL_ARCHITECTURE,"version":MODEL_VERSION,"input_size":"224 × 224","epochs":"20 + up to 25 fine-tune","classes":4,"training_samples":5600,"validation_samples":1400,"testing_samples":1600}
    test = DATA_DIR / "Testing"
    if test.exists():
        ds = make_dataset(test); truth=np.concatenate([y.numpy().argmax(1) for _,y in ds]); pred=model.predict(ds,verbose=0).argmax(1)
        metrics["accuracy"] = float(np.mean(pred==truth)); cm=np.zeros((4,4),dtype=int)
        for a,b in zip(truth,pred): cm[a,b]+=1
        precision=np.divide(np.diag(cm),np.maximum(cm.sum(0),1)); recall=np.divide(np.diag(cm),np.maximum(cm.sum(1),1)); f1=2*precision*recall/np.maximum(precision+recall,1e-9)
        metrics.update(precision_macro=float(precision.mean()),recall_macro=float(recall.mean()),f1_macro=float(f1.mean()),confusion_matrix=cm.tolist(),roc_auc="Not recorded by this training workflow")
    METRICS_FILE.write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(f"Saved {MODEL_FILE}. Evaluation metrics: {METRICS_FILE}")


class NeuroScanApp:
    def __init__(self, model: tf.keras.Model, labels: list[str]) -> None:
        self.model, self.labels = model, labels
        self.path: Path | None = None; self.original: Image.Image | None = None; self.attention: Image.Image | None = None; self.overlay: Image.Image | None = None
        self.mode, self.zoom, self.pan_x, self.pan_y = "Original MRI", 1.0, 0, 0
        self.brightness, self.contrast, self.drag = 1.0, 1.0, None
        self.scores: np.ndarray | None = None; self.quality = "Not checked"; self.summary = ""; self.canvas_photo = None
        self.root = tk.Tk(); self.root.title("NeuroScan — AI-Assisted MRI Analysis"); self.root.configure(bg=C["bg"])
        try: self.root.state("zoomed")
        except tk.TclError: self.root.geometry("1440x860")
        self.root.minsize(1120, 680); self._styles(); self._layout()

    def _styles(self) -> None:
        style=ttk.Style(self.root); style.theme_use("clam")
        style.configure("Violet.TButton",background=C["violet"],foreground=C["text"],borderwidth=0,font=("Segoe UI",10,"bold"),padding=(12,8)); style.map("Violet.TButton",background=[("active",C["purple"]),("disabled",C["hover"])])
        style.configure("Ghost.TButton",background=C["card"],foreground=C["muted"],borderwidth=0,font=("Segoe UI",9),padding=(9,6)); style.map("Ghost.TButton",background=[("active",C["hover"])])
        style.configure("Dark.Horizontal.TProgressbar",troughcolor=C["hover"],background=C["violet"],thickness=6)

    def _label(self,p,text,size=10,fg=None,**kwargs): return tk.Label(p,text=text,bg=C["card"],fg=fg or C["text"],font=("Segoe UI",size),**kwargs)
    def _panel(self,p):
        outer=tk.Frame(p,bg="#0B0D0F",padx=1,pady=1); inner=tk.Frame(outer,bg=C["card"]); inner.pack(fill="both",expand=True); return outer,inner

    def _layout(self) -> None:
        nav=tk.Frame(self.root,bg=C["surface"],height=68); nav.pack(fill="x"); nav.pack_propagate(False)
        tk.Label(nav,text="NeuroScan",bg=C["surface"],fg=C["text"],font=("Segoe UI",20,"bold")).pack(side="left",padx=(24,8),pady=13)
        tk.Label(nav,text="AI-Assisted MRI Analysis",bg=C["surface"],fg=C["muted"],font=("Segoe UI",10)).pack(side="left",pady=20)
        ttk.Button(nav,text="⚙ Settings",style="Ghost.TButton",command=self.show_help).pack(side="right",padx=20,pady=15)
        tk.Label(nav,text=f"Model: EfficientNet  •  Version: {MODEL_VERSION}  •  ● Ready",bg=C["surface"],fg=C["teal"],font=("Segoe UI",9,"bold")).pack(side="right",padx=12)
        main=tk.Frame(self.root,bg=C["bg"],padx=16,pady=16); main.pack(fill="both",expand=True)
        for col,w in enumerate((4,4,3)): main.columnconfigure(col,weight=w)
        main.rowconfigure(0,weight=1)
        a,self.left=self._panel(main); a.grid(row=0,column=0,sticky="nsew",padx=(0,7)); self._left_panel()
        b,self.center=self._panel(main); b.grid(row=0,column=1,sticky="nsew",padx=7); self._center_panel()
        d,self.right=self._panel(main); d.grid(row=0,column=2,sticky="nsew",padx=(7,0)); self._right_panel()

    def _left_panel(self) -> None:
        p=self.left; self._label(p,"MRI EXPLORER",9,C["violet2"]).pack(anchor="w",padx=16,pady=(14,2)); self._label(p,"Interactive image viewer",16,fg=C["text"]).pack(anchor="w",padx=16)
        self.viewer=tk.Canvas(p,bg="#0A0C0E",highlightthickness=0,cursor="fleur"); self.viewer.pack(fill="both",expand=True,padx=16,pady=14)
        self.viewer.create_text(200,150,text="Choose an MRI image to begin",fill=C["muted"],font=("Segoe UI",12)); self.viewer.bind("<MouseWheel>",self.wheel_zoom); self.viewer.bind("<ButtonPress-1>",self.pan_start); self.viewer.bind("<B1-Motion>",self.pan_move)
        tools=tk.Frame(p,bg=C["card"]); tools.pack(fill="x",padx=16,pady=(0,8))
        for text,cmd in [("Open image",self.open_image),("Zoom +",lambda:self.change_zoom(1.2)),("Zoom −",lambda:self.change_zoom(.83)),("Fit",self.fit_image),("Reset",self.reset_view),("Fullscreen",self.toggle_fullscreen)]: ttk.Button(tools,text=text,style="Ghost.TButton",command=cmd).pack(side="left",padx=(0,4))
        adjust=tk.Frame(p,bg=C["card"]); adjust.pack(fill="x",padx=16,pady=(0,14))
        self._label(adjust,"Brightness",9,C["muted"]).pack(side="left"); self.bright_var=tk.DoubleVar(value=1); tk.Scale(adjust,from_=.5,to=1.6,resolution=.05,orient="horizontal",variable=self.bright_var,command=lambda _:self.render_viewer(),bg=C["card"],fg=C["muted"],troughcolor=C["hover"],highlightthickness=0,length=110).pack(side="left",padx=(4,9))
        self._label(adjust,"Contrast",9,C["muted"]).pack(side="left"); self.contrast_var=tk.DoubleVar(value=1); tk.Scale(adjust,from_=.5,to=1.8,resolution=.05,orient="horizontal",variable=self.contrast_var,command=lambda _:self.render_viewer(),bg=C["card"],fg=C["muted"],troughcolor=C["hover"],highlightthickness=0,length=110).pack(side="left",padx=4)

    def _center_panel(self) -> None:
        p=self.center; self._label(p,"WHY DID THE AI PREDICT THIS?",9,C["violet2"]).pack(anchor="w",padx=16,pady=(14,2)); self._label(p,"AI explainability",16).pack(anchor="w",padx=16)
        modebar=tk.Frame(p,bg=C["card"]); modebar.pack(fill="x",padx=16,pady=12); self.mode_buttons=[]
        for name in ("Original MRI","AI Attention","AI Overlay"):
            btn=ttk.Button(modebar,text=name,style="Ghost.TButton",command=lambda n=name:self.set_mode(n)); btn.pack(side="left",padx=(0,5)); self.mode_buttons.append(btn)
        self.explain=tk.Label(p,text="AI attention will appear after an image is classified.",bg="#0A0C0E",fg=C["muted"],font=("Segoe UI",10),justify="center"); self.explain.pack(fill="both",expand=True,padx=16,pady=(0,12))
        opacity=tk.Frame(p,bg=C["card"]); opacity.pack(fill="x",padx=16); self._label(opacity,"Heatmap opacity",9,C["muted"]).pack(side="left"); self.opacity_var=tk.DoubleVar(value=.55); tk.Scale(opacity,from_=0,to=1,resolution=.05,orient="horizontal",variable=self.opacity_var,command=lambda _:self.refresh_explain(),bg=C["card"],fg=C["muted"],troughcolor=C["hover"],highlightthickness=0,length=180).pack(side="left",padx=8)
        self.explanation_text=self._label(p,"AI attention maps show regions that influenced a classification. They do not identify tumour boundaries or provide a clinical diagnosis.",9,C["muted"],wraplength=420,justify="left"); self.explanation_text.pack(anchor="w",padx=16,pady=(12,16))

    def _right_panel(self) -> None:
        p=self.right; self._label(p,"AI CLASSIFICATION RESULT",9,C["violet2"]).pack(anchor="w",padx=16,pady=(14,4)); self.status=self._label(p,"READY",9,C["teal"],padx=8,pady=4); self.status.pack(anchor="w",padx=16)
        self.result=self._label(p,"Awaiting image",20,wraplength=330,justify="left"); self.result.pack(anchor="w",padx=16,pady=(14,4)); self.confidence=self._label(p,"Model confidence will appear here.",10,C["muted"],wraplength=330,justify="left"); self.confidence.pack(anchor="w",padx=16)
        self.quality_label=self._label(p,"Image Quality: Not checked",10,C["muted"],wraplength=330,justify="left"); self.quality_label.pack(anchor="w",padx=16,pady=(15,3)); self.quality_detail=self._label(p,"",9,C["dim"],wraplength=330,justify="left"); self.quality_detail.pack(anchor="w",padx=16)
        self._label(p,"CLASS PROBABILITY DISTRIBUTION",9,C["muted"]).pack(anchor="w",padx=16,pady=(16,5)); self.prob_box=tk.Frame(p,bg=C["card"]); self.prob_box.pack(fill="x",padx=16)
        self._label(p,"ANALYSIS PIPELINE",9,C["muted"]).pack(anchor="w",padx=16,pady=(16,5)); self.pipeline=[]
        for item in ("MRI loaded","Image quality checked","Preprocessing completed","AI classification completed","Explainability map generated"):
            line=self._label(p,"○  "+item,9,C["dim"]); line.pack(anchor="w",padx=16,pady=2); self.pipeline.append(line)
        action=tk.Frame(p,bg=C["card"]); action.pack(fill="x",side="bottom",padx=16,pady=16); self.report_btn=ttk.Button(action,text="Generate AI Analysis Report",style="Violet.TButton",command=self.generate_report,state="disabled"); self.report_btn.pack(fill="x"); ttk.Button(action,text="Model Performance",style="Ghost.TButton",command=self.model_performance).pack(fill="x",pady=(8,0))

    def open_image(self) -> None:
        path=filedialog.askopenfilename(title="Choose an MRI image",filetypes=[("MRI images","*.png *.jpg *.jpeg")])
        if not path: return
        try:
            image=Image.open(path); image.load()
            if image.format not in {"PNG","JPEG"}: raise ValueError("Only PNG, JPG, and JPEG files are supported.")
            if min(image.size)<32: raise ValueError("This image is too small. Choose an image at least 32 × 32 pixels.")
            self.path, self.original=Path(path),image.convert("RGB"); self.attention=self.overlay=None; self.fit_image(); self.start_analysis()
        except (UnidentifiedImageError,OSError,ValueError) as err: messagebox.showerror("Unable to open image",f"Choose a readable PNG, JPG, or JPEG image.\n\n{err}")

    def quality_check(self) -> tuple[str,str]:
        a=np.asarray(self.original.convert("L"),dtype=np.float32)/255; bright=a.mean(); contrast=a.std(); blur=np.mean(np.abs(np.diff(a,axis=0)))+np.mean(np.abs(np.diff(a,axis=1)))
        notes=[]
        if not .12<bright<.88: notes.append("brightness")
        if contrast<.06: notes.append("contrast")
        if blur<.015: notes.append("sharpness")
        return ("Review recommended", "Low " + ", ".join(notes) + " may reduce prediction reliability.") if notes else ("Good", "Resolution, brightness, contrast, and sharpness are within simple local screening ranges.")

    def start_analysis(self) -> None:
        self.status.configure(text="ANALYZING…",fg=C["violet2"]); self.result.configure(text="Analyzing image…"); self.confidence.configure(text="Local processing in progress."); self.report_btn.configure(state="disabled")
        for i,line in enumerate(self.pipeline): line.configure(text=("●  " if i==0 else "○  ")+line.cget("text")[3:],fg=C["violet2"] if i==0 else C["dim"])
        threading.Thread(target=self.worker,daemon=True).start()

    def worker(self) -> None:
        try:
            self.quality=self.quality_check(); array=np.asarray(self.original.resize(IMAGE_SIZE),dtype=np.float32); scores=self.model.predict(np.expand_dims(array,0),verbose=0)[0]
        except Exception:
            traceback.print_exc(); self.root.after(0,self.analysis_error); return
        try:
            cam=self.gradcam(array,int(np.argmax(scores)))
        except Exception:
            traceback.print_exc(); cam=None
        self.root.after(0,lambda:self.finish_analysis(scores,cam))

    def gradcam(self,array:np.ndarray,index:int) -> Image.Image | None:
        try: aug,base,gap,dense=(self.model.get_layer(n) for n in ("sequential","efficientnetb0","global_average_pooling2d","dense"))
        except ValueError: return None
        x=tf.convert_to_tensor(np.expand_dims(array,0))
        with tf.GradientTape() as tape:
            conv=base(aug(x,training=False),training=False); tape.watch(conv); pred=dense(gap(conv)); loss=pred[:,index]
        gradients=tape.gradient(loss,conv)
        if gradients is None: return None
        weights=tf.reduce_mean(gradients,axis=(0,1,2)); cam=tf.reduce_sum(conv[0]*weights,axis=-1); cam=tf.maximum(cam,0); cam=cam/(tf.reduce_max(cam)+1e-8)
        heat=np.asarray(Image.fromarray(np.uint8(cam.numpy()*255)).resize(self.original.size,Image.Resampling.BILINEAR),dtype=np.float32)/255
        # Scientific violet-to-coral heatmap; it reflects influence, not a tumour boundary.
        rgb=np.zeros((*heat.shape,3),dtype=np.uint8); rgb[...,0]=(80+175*heat).astype(np.uint8); rgb[...,1]=(25+75*heat).astype(np.uint8); rgb[...,2]=(150+80*(1-heat)).astype(np.uint8)
        return Image.fromarray(rgb)

    def finish_analysis(self,scores:np.ndarray,cam:Image.Image|None) -> None:
        self.scores,self.attention=scores,cam; top=int(np.argmax(scores)); second,first=np.sort(scores)[-2:]; separation=float(first-second); label=self.labels[top]; quality,detail=self.quality
        self.quality_label.configure(text=f"Image Quality: {quality}",fg=C["teal"] if quality=="Good" else C["amber"]); self.quality_detail.configure(text=detail)
        if scores[top]<LOW_CONFIDENCE_THRESHOLD or separation<LOW_SEPARATION_THRESHOLD:
            self.status.configure(text="⚠  REVIEW RECOMMENDED",fg=C["amber"]); note="Low prediction separation or confidence. Professional review is recommended."
        elif label=="notumor": self.status.configure(text="●  CLASSIFICATION COMPLETE",fg=C["teal"]); note="No-tumour class selection does not rule out disease."
        else: self.status.configure(text="!  PROFESSIONAL REVIEW REQUIRED",fg=C["coral"]); note="This classification is not a diagnosis and requires professional review."
        self.result.configure(text=f"Image classification:\n{display_name(label)}"); self.confidence.configure(text=f"Highest predicted probability: {float(scores[top]):.1%}\nModel confidence is relative model certainty, not clinical certainty.\n{note}")
        self.show_probabilities(); self.make_overlay(); self.refresh_explain(); self.explanation_text.configure(text=f"AI Attention Analysis\nThe highlighted areas influenced the {display_name(label)} classification. Attention strength is relative to this image. They do not independently confirm tumour boundaries or a clinical diagnosis.")
        for line in self.pipeline: line.configure(text="✓  "+line.cget("text")[3:],fg=C["teal"])
        self.report_btn.configure(state="normal"); self.summary=self.make_summary(label,separation)

    def analysis_error(self) -> None:
        self.status.configure(text="ANALYSIS COULD NOT COMPLETE",fg=C["coral"]); self.result.configure(text="Unable to classify this image"); self.confidence.configure(text="Try another readable PNG, JPG, or JPEG image. Your image was not uploaded or stored.")

    def show_probabilities(self) -> None:
        for w in self.prob_box.winfo_children(): w.destroy()
        top=int(np.argmax(self.scores))
        for i,(name,score) in enumerate(zip(self.labels,self.scores)):
            row=tk.Frame(self.prob_box,bg=C["card"]); row.pack(fill="x",pady=4); tk.Label(row,text=display_name(name).replace("-like pattern",""),bg=C["card"],fg=C["muted"],width=20,anchor="w",font=("Segoe UI",9)).pack(side="left")
            track=tk.Frame(row,bg=C["hover"],height=8); track.pack(side="left",fill="x",expand=True,padx=6); tk.Frame(track,bg=C["violet"] if i==top else C["dim"],height=8).place(relwidth=max(.01,float(score)),relheight=1)
            tk.Label(row,text=f"{float(score):.1%}",bg=C["card"],fg=C["text"],width=6,anchor="e",font=("Segoe UI",9,"bold")).pack(side="right")

    def make_overlay(self) -> None:
        if self.attention: self.overlay=Image.blend(self.original,self.attention,float(self.opacity_var.get()))

    def set_mode(self,mode:str) -> None: self.mode=mode; self.refresh_explain()
    def refresh_explain(self) -> None:
        if not self.original: return
        self.make_overlay(); image={"Original MRI":self.original,"AI Attention":self.attention,"AI Overlay":self.overlay}.get(self.mode) or self.original
        shown=image.copy(); shown.thumbnail((480,420)); photo=ImageTk.PhotoImage(shown); self.explain.configure(image=photo,text=""); self.explain.image=photo
    def fit_image(self) -> None: self.zoom=1; self.pan_x=self.pan_y=0; self.render_viewer()
    def toggle_fullscreen(self) -> None:
        enabled = not bool(self.root.attributes("-fullscreen"))
        self.root.attributes("-fullscreen", enabled)
        if enabled: self.root.bind("<Escape>", lambda _event: self.root.attributes("-fullscreen", False))
    def reset_view(self) -> None:
        self.bright_var.set(1); self.contrast_var.set(1); self.fit_image()
    def change_zoom(self,factor:float) -> None: self.zoom=min(5,max(.2,self.zoom*factor)); self.render_viewer()
    def wheel_zoom(self,event) -> None: self.change_zoom(1.12 if event.delta>0 else .89)
    def pan_start(self,event) -> None: self.drag=(event.x,event.y)
    def pan_move(self,event) -> None:
        if self.drag: self.pan_x+=event.x-self.drag[0]; self.pan_y+=event.y-self.drag[1]; self.drag=(event.x,event.y); self.render_viewer()
    def render_viewer(self) -> None:
        if not self.original: return
        image=ImageEnhance.Contrast(ImageEnhance.Brightness(self.original).enhance(self.bright_var.get())).enhance(self.contrast_var.get()); w,h=image.size; image=image.resize((max(1,int(w*self.zoom)),max(1,int(h*self.zoom))))
        self.canvas_photo=ImageTk.PhotoImage(image); self.viewer.delete("all"); cw,ch=max(1,self.viewer.winfo_width()),max(1,self.viewer.winfo_height()); self.viewer.create_image(cw//2+self.pan_x,ch//2+self.pan_y,image=self.canvas_photo)
    def make_summary(self,label:str,separation:float)->str:
        parts=["NeuroScan — AI-Assisted MRI Analysis Report",f"Analysis ID: NS-{datetime.now().strftime('%Y%m%d-%H%M%S')}",f"Date: {datetime.now():%Y-%m-%d %H:%M:%S}",f"File: {self.path.name}",f"Model: {MODEL_ARCHITECTURE} {MODEL_VERSION}",f"Classification: {display_name(label)}",f"Top probability: {float(self.scores.max()):.1%}",f"Top-two separation: {separation:.1%}",f"Image quality: {self.quality[0]}","Class probabilities:"]
        parts += [f"- {display_name(n)}: {float(s):.1%}" for n,s in zip(self.labels,self.scores)]; parts += [DISCLAIMER,"AI attention maps show prediction influence only; they do not identify tumour boundaries."]
        return "\n".join(parts)
    def generate_report(self) -> None:
        report_dir=ROOT/"reports"; report_dir.mkdir(exist_ok=True); target=report_dir/f"neuroscan_report_{datetime.now():%Y%m%d_%H%M%S}.html"
        def encoded(image):
            b=io.BytesIO(); image.save(b,format="PNG"); return base64.b64encode(b.getvalue()).decode()
        mri=encoded(self.original); cam=encoded(self.overlay or self.original); text="<br>".join(self.summary.replace("&","&amp;").replace("<","&lt;").splitlines())
        target.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>NeuroScan AI-Assisted MRI Analysis Report</title><style>body{{font:16px Segoe UI;background:#111315;color:#f5f5f4;max-width:950px;margin:auto;padding:35px}}h1{{color:#a78bfa}}.card{{background:#22262a;padding:20px;margin:18px 0}}img{{max-width:46%;margin-right:2%;background:#000}}</style></head><body><h1>NeuroScan — AI-Assisted MRI Analysis Report</h1><div class='card'>{text}</div><div class='card'><h2>MRI preview and AI attention overlay</h2><img src='data:image/png;base64,{mri}'><img src='data:image/png;base64,{cam}'><p>{DISCLAIMER}</p></div></body></html>",encoding="utf-8")
        messagebox.showinfo("Report generated",f"Saved local report:\n{target}\n\nIt has not been uploaded or sent anywhere.")
    def model_performance(self) -> None:
        if METRICS_FILE.exists(): data=json.loads(METRICS_FILE.read_text(encoding="utf-8")); text="\n".join(f"{k.replace('_',' ').title()}: {v if not isinstance(v,float) else f'{v:.1%}'}" for k,v in data.items())
        else: text="Performance metrics are not available yet. Run --train to create model_metrics.json after held-out testing.\n\nModel: EfficientNetB0\nInput: 224 × 224\nClasses: 4\nVersion: "+MODEL_VERSION
        messagebox.showinfo("Model Performance",text)
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