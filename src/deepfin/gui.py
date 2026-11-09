"""
Point-and-click app for deepfin.explore: pick a folder of images, a network, a map method and a grouping method,
look at the map with a thumbnail of every image, change the settings until the groups look right, then save.

    deepfin-gui            (or: python -m deepfin.gui)

The fingerprints (steps 1-2, the slow part) are kept: changing only steps 3 or 4 redraws the map in seconds.
Uses Tkinter, which comes with Python from python.org (Homebrew's Python needs `brew install python-tk`).
"""
from __future__ import annotations

import os, queue, subprocess, sys, threading, traceback
from pathlib import Path

from .explore import MODELS, find_images

NOTES = {"MobileNetV2": "fast; among the best on our clownfish", "VGG16": "older, strong on patterns", "DenseNet121": "accurate",
         "ResNet50": "general purpose", "EfficientNetB0": "fast", "InceptionV3": "general purpose",
         "EfficientNetV2L": "large and slow (the paper's default)"}
SAVE = [("save_fig", "The map (map.png and map.pdf)"),
        ("save_features", "The fingerprints: raw feature vectors (features.csv)"),
        ("save_coords", "Groups and coordinates in every dimension (clusters.csv)"),
        ("save_copies", "A copy of each image in one folder per group (sorted/)")]


class App:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        self.tk, self.root, self.q, self.out = tk, root, queue.Queue(), None
        self.feat = None    # last fingerprints, dict(key, paths, thumbs, X): reused while only steps 3-4 change
        self.result = None  # the map on screen, dict(feat, emb, labels, variance, settings): what Save writes
        root.title("DEEPFIN: sort images into groups")
        root.minsize(1000, 650)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)
        f = ttk.Frame(root, padding=(14, 10))
        f.grid(row=0, column=0, sticky="nsew")
        f.columnconfigure(0, weight=1)
        V = lambda v, kind=tk.StringVar: kind(value=v)
        self.v = dict(folder=V(""), remove_bg=V(False, tk.BooleanVar), model=V(f"MobileNetV2: {NOTES['MobileNetV2']}"),
                      reduce=V("umap"), dims=V(2, tk.IntVar), n_neighbors=V(15, tk.IntVar), min_dist=V(0.1, tk.DoubleVar),
                      seed=V(24, tk.IntVar), cluster=V("hdbscan"), min_cluster_size=V(5, tk.IntVar),
                      min_samples=V(0, tk.IntVar), k=V(2, tk.IntVar), out=V(""), thumb=V(1.0, tk.DoubleVar),
                      **{key: V(True, tk.BooleanVar) for key, _ in SAVE})

        s1 = ttk.LabelFrame(f, text="1. Images", padding=(8, 4)); s1.grid(sticky="ew", pady=2); s1.columnconfigure(0, weight=1)
        ttk.Entry(s1, textvariable=self.v["folder"]).grid(row=0, column=0, sticky="ew")
        ttk.Button(s1, text="Choose folder…", command=self.pick_folder).grid(row=0, column=1, padx=(8, 0))
        self.count = ttk.Label(s1, text="Choose a folder of jpg, png, tif, bmp or webp images.")
        self.count.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Checkbutton(s1, text="Remove backgrounds first (rembg; slower, helps with photos of single objects)",
                        variable=self.v["remove_bg"]).grid(row=2, column=0, columnspan=2, sticky="w")
        self.v["folder"].trace_add("write", lambda *_: self.folder_changed())

        s2 = ttk.LabelFrame(f, text="2. Neural network that fingerprints each image", padding=(8, 4)); s2.grid(sticky="ew", pady=2)
        ttk.Combobox(s2, textvariable=self.v["model"], state="readonly", width=52,
                     values=[f"{m}: {NOTES[m]}" for m in MODELS]).grid(row=0, column=0, sticky="w")

        s3 = ttk.LabelFrame(f, text="3. Squeeze the fingerprints into a few dimensions", padding=(8, 4)); s3.grid(sticky="ew", pady=2)
        ttk.Radiobutton(s3, text="UMAP (keeps similar images together)", value="umap", variable=self.v["reduce"],
                        command=self.toggle).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(s3, text="PCA (fast, straight-line)", value="pca", variable=self.v["reduce"],
                        command=self.toggle).grid(row=0, column=1, sticky="w", padx=(16, 0))
        self.umap_opts = self.options(s3, 1, [("neighbours", "n_neighbors", 2, 200, 1), ("min. distance", "min_dist", 0, 0.99, 0.05)])
        self.options(s3, 2, [("dimensions", "dims", 2, 100, 1), ("random seed", "seed", 0, 99999, 1)])
        ttk.Label(s3, text="Step 4 uses all the dimensions; the map shows the first two.", foreground="gray"
                  ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))

        s4 = ttk.LabelFrame(f, text="4. Find groups", padding=(8, 4)); s4.grid(sticky="ew", pady=2)
        ttk.Radiobutton(s4, text="HDBSCAN (finds the number of groups itself)", value="hdbscan", variable=self.v["cluster"],
                        command=self.toggle).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(s4, text="k-means (you set the number)", value="kmeans", variable=self.v["cluster"],
                        command=self.toggle).grid(row=0, column=1, sticky="w", padx=(16, 0))
        self.hdb_opts = self.options(s4, 1, [("smallest group", "min_cluster_size", 2, 1000, 1),
                                             ("min. samples (0 = same)", "min_samples", 0, 1000, 1)])
        self.km_opts = self.options(s4, 2, [("number of groups", "k", 2, 100, 1)])

        self.run_btn = ttk.Button(f, text="Make the map", command=self.start)
        self.run_btn.grid(sticky="ew", pady=(8, 2), ipady=4)
        ttk.Label(f, text="Changed only step 3 or 4? The fingerprints are kept, so the new map takes seconds.",
                  foreground="gray").grid(sticky="w")
        self.bar = ttk.Progressbar(f, mode="indeterminate"); self.bar.grid(sticky="ew", pady=(4, 0))
        self.log = tk.Text(f, height=4, width=60, wrap="word", state="disabled"); self.log.grid(sticky="nsew", pady=(4, 0))
        f.rowconfigure(self.log.grid_info()["row"], weight=1)

        s5 = ttk.LabelFrame(f, text="5. Happy with the groups? Save the results", padding=(8, 4))
        s5.grid(sticky="ew", pady=(6, 0)); s5.columnconfigure(0, weight=1)
        for i, (key, text) in enumerate(SAVE):
            ttk.Checkbutton(s5, text=text, variable=self.v[key]).grid(row=i, column=0, columnspan=2, sticky="w")
        ttk.Entry(s5, textvariable=self.v["out"]).grid(row=4, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(s5, text="Choose folder…", command=self.pick_out).grid(row=4, column=1, padx=(8, 0), pady=(6, 0))
        self.save_btn = ttk.Button(s5, text="Save", command=self.save, state="disabled")
        self.save_btn.grid(row=5, column=0, sticky="ew", pady=(6, 0), ipady=2)
        self.open_btn = ttk.Button(s5, text="Open folder", command=self.open_out, state="disabled")
        self.open_btn.grid(row=5, column=1, padx=(8, 0), pady=(6, 0), ipady=2)

        right = ttk.Frame(root, padding=(0, 10, 14, 10))
        right.grid(row=0, column=1, sticky="nsew")
        size = ttk.Frame(right); size.pack(side="top", fill="x")
        ttk.Label(size, text="Thumbnail size").pack(side="left")
        ttk.Scale(size, from_=0.5, to=3, variable=self.v["thumb"], length=220,
                  command=lambda _: self.thumb_changed()).pack(side="left", padx=8)
        self.thumb_label = ttk.Label(size, text="100%", width=5); self.thumb_label.pack(side="left")
        self.redraw_job = None
        self.fig = Figure(figsize=(7, 6), layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False).pack(side="bottom", fill="x")  # zoom and pan
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.fig.text(0.5, 0.5, "The map appears here.\nChoose a folder of images and press Make the map.",
                      ha="center", va="center", color="gray", fontsize=13)
        self.canvas.draw()
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

    def busy(self, on):
        for b in (self.run_btn, self.save_btn):
            b.state(["disabled"] if on or (b is self.save_btn and not self.result) else ["!disabled"])
        self.bar.start(12) if on else self.bar.stop()

    def start(self):
        from tkinter import messagebox
        try:
            g = {k: v.get() for k, v in self.v.items()}
        except self.tk.TclError:
            return messagebox.showerror("DEEPFIN", "Please check the numbers in the settings.")
        if not g["folder"].strip() or not Path(g["folder"]).expanduser().is_dir():  # empty would mean the current folder
            return messagebox.showerror("DEEPFIN", "Please choose a folder of images first.")
        choices = dict(folder=str(Path(g["folder"]).expanduser()), model=g["model"].split(":")[0],
                       remove_background=g["remove_bg"], reduce=g["reduce"], dims=g["dims"], n_neighbors=g["n_neighbors"],
                       min_dist=g["min_dist"], seed=g["seed"], cluster=g["cluster"], min_cluster_size=g["min_cluster_size"],
                       min_samples=g["min_samples"] or None, k=g["k"])
        self.busy(True)
        self.log.config(state="normal"); self.log.delete("1.0", "end"); self.log.config(state="disabled")
        threading.Thread(target=self.work, args=(choices,), daemon=True).start()

    def work(self, c):
        try:
            from .explore import check, embed, fingerprint, group, settings_for
            say = lambda s: self.q.put(("log", s))
            paths = find_images(c["folder"])
            check(len(paths), c["reduce"], c["dims"], c["cluster"], c["k"], c["min_cluster_size"])
            key = (c["folder"], c["model"], c["remove_background"], tuple(p.name for p in paths))
            if self.feat and self.feat["key"] == key:
                say(f"1-2/4  Keeping the {c['model']} fingerprints of the {len(paths)} images")
            else:
                self.feat = dict(zip(("paths", "thumbs", "X"), fingerprint(paths, c["model"], c["remove_background"], say)), key=key)
            feat = self.feat
            emb, variance = embed(feat["X"], c["reduce"], c["dims"], c["n_neighbors"], c["min_dist"], c["seed"], say)
            labels = group(emb, c["cluster"], c["min_cluster_size"], c["min_samples"], c["k"], c["seed"], say)
            self.q.put(("done", dict(feat=feat, emb=emb, labels=labels, variance=variance,
                                     settings=settings_for(images=len(paths), **c))))
        except Exception as e:  # show any problem in the window instead of a silent failure
            traceback.print_exc()
            self.q.put(("error", str(e)))

    def poll(self):
        from tkinter import messagebox
        from .explore import summary
        while not self.q.empty():
            kind, value = self.q.get()
            if kind == "log":
                self.say(value)
                continue
            if kind == "done":
                self.result = r = value
                self.redraw()
                self.say("Map ready: " + summary(r["labels"]) + ". Not separated well? Change the settings and press "
                         "Make the map again.")
            else:
                self.say("Problem: " + value); messagebox.showerror("DEEPFIN", value)
            self.busy(False)
        self.root.after(100, self.poll)

    def thumb_changed(self):
        self.thumb_label.config(text=f"{self.v['thumb'].get():.0%}")
        if self.redraw_job:  # redraw once the slider stops, not at every step of a drag
            self.root.after_cancel(self.redraw_job)
        self.redraw_job = self.root.after(250, self.redraw)

    def redraw(self):
        from .explore import draw_map, title_for
        self.redraw_job, r = None, self.result
        if r:
            draw_map(self.fig, r["emb"], r["labels"], r["feat"]["thumbs"], title_for(r["settings"]), r["variance"],
                     thumb_scale=self.v["thumb"].get())
            self.canvas.draw_idle()

    def save(self):
        from tkinter import messagebox
        from .explore import save
        r, folder = self.result, Path(self.result["settings"]["folder"])
        out = self.v["out"].get().strip() or str(folder.parent / f"{folder.name}_deepfin")
        self.root.config(cursor="watch"); self.root.update()
        try:
            self.out = save(out, r["feat"]["paths"], r["feat"]["X"], r["emb"], r["labels"], r["settings"],
                            fig=self.fig if self.v["save_fig"].get() else None, features=self.v["save_features"].get(),
                            coords=self.v["save_coords"].get(), copies=self.v["save_copies"].get(), variance=r["variance"])
        except OSError as e:
            return messagebox.showerror("DEEPFIN", f"Could not save: {e}")
        finally:
            self.root.config(cursor="")
        self.say(f"Saved to {self.out}")
        self.open_btn.state(["!disabled"])

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
