"""
Point-and-click app for deepfin.explore: pick a folder of images, a network, a map method and a clustering method.

    deepfin-gui            (or: python -m deepfin.gui)

Uses Tkinter, which comes with Python from python.org (Homebrew's Python needs `brew install python-tk`).
"""
from __future__ import annotations

import os, queue, subprocess, sys, threading, traceback
from pathlib import Path

from .explore import MODELS, find_images

NOTES = {"MobileNetV2": "fast; among the best on our clownfish", "VGG16": "older, strong on patterns", "DenseNet121": "accurate",
         "ResNet50": "general purpose", "EfficientNetB0": "fast", "InceptionV3": "general purpose",
         "EfficientNetV2L": "large and slow (the paper's default)"}


class App:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.root, self.q, self.out = tk, root, queue.Queue(), None
        root.title("DEEPFIN: sort images into groups")
        root.minsize(720, 640)
        f = ttk.Frame(root, padding=14)
        f.pack(fill="both", expand=True)
        f.columnconfigure(0, weight=1)
        V = lambda v, kind=tk.StringVar: kind(value=v)
        self.v = dict(folder=V(""), remove_bg=V(False, tk.BooleanVar), model=V(f"MobileNetV2: {NOTES['MobileNetV2']}"),
                      reduce=V("umap"), n_neighbors=V(15, tk.IntVar), min_dist=V(0.1, tk.DoubleVar), seed=V(24, tk.IntVar),
                      cluster=V("hdbscan"), min_cluster_size=V(5, tk.IntVar), min_samples=V(0, tk.IntVar), k=V(2, tk.IntVar),
                      out=V(""), copies=V(True, tk.BooleanVar))

        s1 = ttk.LabelFrame(f, text="1. Images", padding=10); s1.grid(sticky="ew", pady=4); s1.columnconfigure(0, weight=1)
        ttk.Entry(s1, textvariable=self.v["folder"]).grid(row=0, column=0, sticky="ew")
        ttk.Button(s1, text="Choose folder…", command=self.pick_folder).grid(row=0, column=1, padx=(8, 0))
        self.count = ttk.Label(s1, text="Choose a folder of jpg, png, tif, bmp or webp images.")
        self.count.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Checkbutton(s1, text="Remove backgrounds first (rembg; slower, helps with photos of single objects)",
                        variable=self.v["remove_bg"]).grid(row=2, column=0, columnspan=2, sticky="w")
        self.v["folder"].trace_add("write", lambda *_: self.folder_changed())

        s2 = ttk.LabelFrame(f, text="2. Neural network that fingerprints each image", padding=10); s2.grid(sticky="ew", pady=4)
        ttk.Combobox(s2, textvariable=self.v["model"], state="readonly", width=52,
                     values=[f"{m}: {NOTES[m]}" for m in MODELS]).grid(row=0, column=0, sticky="w")

        s3 = ttk.LabelFrame(f, text="3. Flatten the fingerprints onto a 2-D map", padding=10); s3.grid(sticky="ew", pady=4)
        ttk.Radiobutton(s3, text="UMAP (keeps similar images together)", value="umap", variable=self.v["reduce"],
                        command=self.toggle).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(s3, text="PCA (fast, straight-line)", value="pca", variable=self.v["reduce"],
                        command=self.toggle).grid(row=0, column=1, sticky="w", padx=(16, 0))
        self.umap_opts = self.options(s3, 1, [("neighbours", "n_neighbors", 2, 200, 1), ("min. distance", "min_dist", 0, 0.99, 0.05)])
        self.options(s3, 2, [("random seed", "seed", 0, 99999, 1)])

        s4 = ttk.LabelFrame(f, text="4. Find groups on the map", padding=10); s4.grid(sticky="ew", pady=4)
        ttk.Radiobutton(s4, text="HDBSCAN (finds the number of groups itself)", value="hdbscan", variable=self.v["cluster"],
                        command=self.toggle).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(s4, text="k-means (you set the number of groups)", value="kmeans", variable=self.v["cluster"],
                        command=self.toggle).grid(row=0, column=1, sticky="w", padx=(16, 0))
        self.hdb_opts = self.options(s4, 1, [("smallest group", "min_cluster_size", 2, 1000, 1),
                                             ("min. samples (0 = same)", "min_samples", 0, 1000, 1)])
        self.km_opts = self.options(s4, 2, [("number of groups", "k", 2, 100, 1)])

        s5 = ttk.LabelFrame(f, text="Results", padding=10); s5.grid(sticky="ew", pady=4); s5.columnconfigure(0, weight=1)
        ttk.Entry(s5, textvariable=self.v["out"]).grid(row=0, column=0, sticky="ew")
        ttk.Button(s5, text="Choose folder…", command=self.pick_out).grid(row=0, column=1, padx=(8, 0))
        ttk.Checkbutton(s5, text="Also copy the images into one folder per group", variable=self.v["copies"]).grid(row=1, column=0, sticky="w", pady=(4, 0))

        b = ttk.Frame(f); b.grid(sticky="ew", pady=(8, 4)); b.columnconfigure(0, weight=1)
        self.run_btn = ttk.Button(b, text="Sort the images", command=self.start)
        self.run_btn.grid(row=0, column=0, sticky="ew", ipady=4)
        self.open_btn = ttk.Button(b, text="Open results folder", command=self.open_out, state="disabled")
        self.open_btn.grid(row=0, column=1, padx=(8, 0), ipady=4)
        self.bar = ttk.Progressbar(f, mode="indeterminate"); self.bar.grid(sticky="ew")
        self.log = tk.Text(f, height=10, wrap="word", state="disabled"); self.log.grid(sticky="nsew", pady=(6, 0))
        f.rowconfigure(self.log.grid_info()["row"], weight=1)
        self.toggle()
        root.after(100, self.poll)

    def options(self, parent, row, specs):
        from tkinter import ttk
        frame = ttk.Frame(parent); frame.grid(row=row, column=0, columnspan=2, sticky="w", pady=(6, 0))
        widgets = []
        for i, (label, key, lo, hi, step) in enumerate(specs):
            ttk.Label(frame, text=label).grid(row=0, column=2 * i, padx=(0 if i == 0 else 16, 4))
            w = ttk.Spinbox(frame, from_=lo, to=hi, increment=step, width=7, textvariable=self.v[key])
            w.grid(row=0, column=2 * i + 1); widgets.append(w)
        return widgets

    def toggle(self):
        for w in self.umap_opts:
            w.state(["!disabled"] if self.v["reduce"].get() == "umap" else ["disabled"])
        for w in self.hdb_opts:
            w.state(["!disabled"] if self.v["cluster"].get() == "hdbscan" else ["disabled"])
        for w in self.km_opts:
            w.state(["!disabled"] if self.v["cluster"].get() == "kmeans" else ["disabled"])

    def pick_folder(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(title="Folder with the images")
        if d:
            self.v["folder"].set(d)

    def pick_out(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(title="Folder for the results")
        if d:
            self.v["out"].set(d)

    def folder_changed(self):
        d = Path(self.v["folder"].get()).expanduser()
        try:
            n = len(find_images(d)) if d.is_dir() else 0
        except OSError:
            n = 0
        self.count.config(text=f"{n} images found" + ("" if n >= 3 else " (need at least 3)"))
        if d.is_dir():
            self.v["out"].set(str(d.parent / f"{d.name}_deepfin"))

    def say(self, line):
        self.log.config(state="normal"); self.log.insert("end", line + "\n"); self.log.see("end"); self.log.config(state="disabled")

    def start(self):
        from tkinter import messagebox
        try:
            g = {k: v.get() for k, v in self.v.items()}
        except self.tk.TclError:
            return messagebox.showerror("DEEPFIN", "Please check the numbers in the settings.")
        if not g["folder"].strip() or not Path(g["folder"]).expanduser().is_dir():  # empty would mean the current folder
            return messagebox.showerror("DEEPFIN", "Please choose a folder of images first.")
        kwargs = dict(folder=g["folder"], model=g["model"].split(":")[0], reduce=g["reduce"], n_neighbors=g["n_neighbors"],
                      min_dist=g["min_dist"], cluster=g["cluster"], min_cluster_size=g["min_cluster_size"],
                      min_samples=g["min_samples"] or None, k=g["k"], seed=g["seed"], remove_background=g["remove_bg"],
                      out=g["out"] or None, sort_copies=g["copies"])
        self.run_btn.state(["disabled"]); self.open_btn.state(["disabled"]); self.bar.start(12)
        self.log.config(state="normal"); self.log.delete("1.0", "end"); self.log.config(state="disabled")
        threading.Thread(target=self.work, args=(kwargs,), daemon=True).start()

    def work(self, kwargs):
        try:
            from .explore import run
            self.q.put(("done", run(log=lambda s: self.q.put(("log", s)), **kwargs)))
        except Exception as e:  # show any problem in the window instead of a silent failure
            traceback.print_exc()
            self.q.put(("error", str(e)))

    def poll(self):
        from tkinter import messagebox
        while not self.q.empty():
            kind, value = self.q.get()
            if kind == "log":
                self.say(value)
            else:
                self.bar.stop(); self.run_btn.state(["!disabled"])
                if kind == "done":
                    self.out = value; self.open_btn.state(["!disabled"])
                else:
                    self.say("Problem: " + value); messagebox.showerror("DEEPFIN", value)
        self.root.after(100, self.poll)

    def open_out(self):
        if not self.out:
            return
        if sys.platform == "darwin":
            subprocess.run(["open", str(self.out)])
        elif os.name == "nt":
            os.startfile(self.out)
        else:
            subprocess.run(["xdg-open", str(self.out)])


def main():
    try:
        import tkinter as tk
    except ImportError:
        sys.exit("The app needs Tkinter. Install Python from python.org, or on Homebrew run: brew install python-tk")
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
