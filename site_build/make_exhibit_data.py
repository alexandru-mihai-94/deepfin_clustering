"""
Data for the "How does an AI see a fish?" exhibit page (deepfin_clustering/docs/how_ai_sees.html).

MobileNetV2 with its ImageNet classifier, run on the 68 fish of the species page
(processed_rembg_v3 cut-outs on white, padded to a square) and the two wild iNaturalist photos
(centre-cropped to a square, as ImageNet photos are). For every image it keeps
  - the 1,000 ImageNet class scores (logits): "how much does it look like a zebra, a bicycle, ..."
  - the 1,280-number fingerprint one layer earlier (global average pool), which DEEPFIN uses
  - for the demo fish: feature maps from five depths of the network, drawn as small heat maps.
It also scores the 144 A. ocellaris stage fish of the stages page (cropped to the fish, padded with
black), so the "build a classifier" screen can switch between species and growth stages. Each set
gets its own 12 everyday objects, the best score any pair of them reaches, and the score of
DEEPFIN's own map (MobileNetV2 UMAP from the species/stages pages) under the same rule.

Run once with OBJECTS/DEMO empty to print the predictions and the classes that separate the
species best, pick the objects and demo fish, then run again to write docs/ai_eye/.

    site_build/venv/bin/python -u site_build/make_exhibit_data.py
"""
import csv, json, re
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image
import keras

REPO = Path(__file__).resolve().parent.parent  # deepfin_clustering/
ROOT = REPO.parent  # the analysis folder next to the repo: flank_umap/, rembg_FishImages/, inat_wild/ (not in git)
CUTOUTS = REPO / "docs" / "percula_ocellaris" / "processed_rembg_v3"
OUT = REPO / "docs" / "ai_eye"
CACHE = REPO / "site_build" / "cache" / "mobilenetv2_imagenet.npz"
STAGE_CACHE = REPO / "site_build" / "cache" / "mobilenetv2_imagenet_stages.npz"
NAMES = [v[1].replace("_", " ") for _, v in sorted(json.load(open(Path.home() / ".keras/models/imagenet_class_index.json")).items(), key=lambda kv: int(kv[0]))]
WILD = {"wild_ocellaris": ("Ocellaris", "wild_ocellaris_raw.jpg"), "wild_percula": ("Percula", "wild_percula_raw.jpg")}
# (layer name, caption) for the five depths shown on the page
# captions describe what the maps on the page actually show (checked by eye)
LAYERS = [("Conv1_relu", "colour and brightness"), ("block_2_add", "edges and outlines"), ("block_5_add", "textures"),
          ("block_12_add", "patches and patterns"), ("out_relu", "the whole fish, coarsely")]

# ImageNet class index -> (emoji, name shown on the page). A mix of objects that split the species
# well (wig, flamingo, penguin, basketball, zebra), partly (orange, butterfly, tiger) and not at all
# (pizza, anemone fish, bicycle, koala); listed in the order they appear on the page.
OBJECTS = {671: ("🚲", "bicycle"), 145: ("🐧", "penguin"), 963: ("🍕", "pizza"), 903: ("👩‍🦰", "wig"),
           105: ("🐨", "koala"), 340: ("🦓", "zebra"), 393: ("🐠", "anemone fish"), 430: ("🏀", "basketball"),
           292: ("🐅", "tiger"), 130: ("🦩", "flamingo"), 950: ("🍊", "orange"), 323: ("🦋", "butterfly")}
# the same idea for the growth stages: scores that rise as the fish grow (anemone fish, traffic light,
# microphone, hermit crab, teapot), fall (frog, hare, chain, vase), or do not help (pufferfish, which is
# the network's favourite answer for larvae, butterfly, pizza)
STAGE_OBJECTS = {397: ("🐡", "pufferfish"), 920: ("🚦", "traffic light"), 963: ("🍕", "pizza"), 32: ("🐸", "frog"),
                 650: ("🎤", "microphone"), 323: ("🦋", "butterfly"), 331: ("🐇", "hare"), 393: ("🐠", "anemone fish"),
                 488: ("⛓️", "chain"), 849: ("🫖", "teapot"), 883: ("🏺", "vase"), 125: ("🦀", "hermit crab")}
# fish for the "pick a fish" screen (checked by eye on demo_fish_sheet.jpg; ocellaris_sample_36 swapped for 30,
# whose "hair slide" answer is not caused by a wooden stick at the head)
DEMO = ["ocellaris_sample_03", "ocellaris_sample_30", "percula_sample_31", "percula_sample_17", "wild_ocellaris", "wild_percula"]


def square_pad(im, bg="white"):
    s = max(im.size); c = Image.new("RGB", (s, s), bg); c.paste(im, ((s - im.width) // 2, (s - im.height) // 2)); return c


def centre_crop(im):
    s = min(im.size); l, t = (im.width - s) // 2, (im.height - s) // 2; return im.crop((l, t, l + s, t + s))


def load_images():
    rows = list(csv.DictReader(open(ROOT / "flank_umap" / "spot_species_dataset.csv")))
    ids, species, imgs = [], [], []
    for r in rows:
        p = Image.open(CUTOUTS / r["processed_file"]).convert("RGBA")
        imgs.append(square_pad(Image.alpha_composite(Image.new("RGBA", p.size, "white"), p).convert("RGB")))
        ids.append(r["individual"]); species.append(r["species"])
    for i, (sp, f) in WILD.items():
        imgs.append(centre_crop(Image.open(REPO / "docs" / "spot_species" / f).convert("RGB")))
        ids.append(i); species.append(sp)
    return ids, np.array(species), imgs


A = keras.applications
model = A.MobileNetV2(weights="imagenet", include_top=True, classifier_activation=None)
gap = next(l for l in model.layers if isinstance(l, keras.layers.GlobalAveragePooling2D))
ids, species, imgs = load_images()
batch = A.mobilenet_v2.preprocess_input(np.stack([np.asarray(im.resize((224, 224), Image.BICUBIC), np.float32) for im in imgs]))

if CACHE.exists():
    d = np.load(CACHE); logits, fp = d["logits"], d["fingerprint"]
else:
    logits, fp = keras.Model(model.input, [model.output, gap.output]).predict(batch, verbose=0)
    np.savez_compressed(CACHE, logits=logits, fingerprint=fp, ids=np.array(ids), species=species)
prob = np.exp(logits - logits.max(1, keepdims=True)); prob /= prob.sum(1, keepdims=True)
lab = np.array([not i.startswith("wild") for i in ids])  # the 68 lab fish

# --- what the classifier says, and which classes separate the species -------------------
top1 = prob.argmax(1)
print("top-1 ImageNet answer for the 68 lab fish:")
for c, n in zip(*np.unique(top1[lab], return_counts=True)):
    print(f"  {n:2d} x {NAMES[c]}")
for i in np.where(~lab)[0]:
    print(f"{ids[i]}: " + ", ".join(f"{NAMES[c]} {prob[i, c]:.0%}" for c in prob[i].argsort()[::-1][:3]))

L, y = logits[lab], species[lab]
auc = np.array([np.mean(L[y == "Percula", c][:, None] > L[y == "Ocellaris", c][None, :]) for c in range(1000)])
sep = np.abs(auc - 0.5) * 2  # 0 = no help, 1 = splits the species perfectly
order = sep.argsort()[::-1]
print("\nclasses that best separate the species (AUC; >0.5 means Percula scores higher):")
for c in order[:40]:
    print(f"  {c:4d} {NAMES[c]:28s} AUC {auc[c]:.2f}")
for c in (671, 444, 870, 340, 292, 393, 1):  # bicycles, zebra, tiger, anemone fish, goldfish
    print(f"  ref {c:4d} {NAMES[c]:28s} AUC {auc[c]:.2f}  rank {int(np.where(order == c)[0][0]) + 1}")

if not OBJECTS or not DEMO:
    raise SystemExit("\nfill in OBJECTS and DEMO, then run again")

# --- assets for the page ---------------------------------------------------------------------
OUT.mkdir(parents=True, exist_ok=True)
for f in OUT.glob("*"):
    f.unlink()
ramp = np.array([[13, 8, 135], [126, 3, 168], [204, 71, 120], [248, 149, 64], [240, 249, 33]], float)  # plasma-like


def heat(a, size=110):
    a = (a - a.min()) / (np.ptp(a) + 1e-9) * (len(ramp) - 1)
    lo = np.floor(a).astype(int).clip(0, len(ramp) - 2); t = (a - lo)[..., None]
    rgb = (ramp[lo] * (1 - t) + ramp[lo + 1] * t).astype(np.uint8)
    return Image.fromarray(rgb).resize((size, size), Image.NEAREST)  # keep the grid visible: 7x7 stays blocky


# fingerprint numbers lined up from "higher in Ocellaris" to "higher in Percula", so strips can show the difference
F = fp[lab]; Fn = (F - F.min(0)) / (np.ptp(F, 0) + 1e-9)
fp_order = np.argsort(Fn[species[lab] == "Ocellaris"].mean(0) - Fn[species[lab] == "Percula"].mean(0))[::-1]
maps_model = keras.Model(model.input, [model.get_layer(n).output for n, _ in LAYERS])
di = [ids.index(i) for i in DEMO]
fmaps = maps_model.predict(batch[di], verbose=0)
demo = []
for k, i in enumerate(di):
    tiles = []
    for (name, _), fm in zip(LAYERS, fmaps):
        a = fm[k]
        ch = np.argsort(a.reshape(-1, a.shape[-1]).std(0))[::-1][:4]  # the 4 most varied channels for this fish
        sheet = Image.new("RGB", (226, 226), (11, 17, 24))
        for j, c in enumerate(ch):
            sheet.paste(heat(a[..., c]), ((j % 2) * 116, (j // 2) * 116))
        sheet.save(OUT / f"{ids[i]}_{name}.png", optimize=True)
        tiles.append(f"ai_eye/{ids[i]}_{name}.png")
    imgs[i].resize((448, 448), Image.BICUBIC).save(OUT / f"{ids[i]}_in.jpg", quality=85)
    q = np.round((fp[i] - F.min(0)) / (np.ptp(F, 0) + 1e-9) * 99).clip(0, 99).astype(int)[fp_order]
    demo.append({"id": ids[i], "species": species[i], "wild": ids[i].startswith("wild"), "img": f"ai_eye/{ids[i]}_in.jpg",
                 "maps": tiles, "top": [[NAMES[c], round(float(prob[i, c]), 3)] for c in prob[i].argsort()[::-1][:3]],
                 "fp": "".join(f"{v:02d}" for v in q)})

def knn_good(pts, labels):  # fish with >= 3 of their 5 nearest neighbours in the same group: the page's rule
    d = ((pts[:, None] - pts[None]) ** 2).sum(-1); np.fill_diagonal(d, np.inf)
    return int(np.sum((labels[np.argsort(d, 1)[:, :5]] == labels[:, None]).sum(1) >= 3))


def make_set(Ls, labels, set_ids, objects, thumb, deepfin_xy):
    objs = list(objects)
    r = np.argsort(np.argsort(Ls[:, objs], 0), 0) / (len(Ls) - 1)  # 0..1 within the set, per object
    best = max((knn_good(r[:, [a, b]], labels), a, b) for a, b in combinations(range(len(objs)), 2))
    return {"thumb": thumb, "n": len(set_ids),
            "objects": [{"name": objects[c][1], "emoji": objects[c][0], "imagenet": NAMES[c]} for c in objs],
            "fish": [[i, (l.item() if hasattr(l, "item") else l), [round(float(v), 3) for v in rr]] for i, l, rr in zip(set_ids, labels, r)],
            "best": {"n": best[0], "pair": [best[1], best[2]]},
            "deepfin": knn_good(np.array([deepfin_xy[i] for i in set_ids]), labels)}


def crop_to_fish(f):  # as in deepfin_clustering/stage_demo_umap.py
    im = Image.open(f).convert("RGB"); im.thumbnail((1600, 1600))
    x0, y0, x1, y1 = im.convert("L").point(lambda v: 255 if v > 25 else 0).getbbox(); p = int(0.04 * (x1 - x0))
    return im.crop((max(0, x0 - p), max(0, y0 - p), min(im.width, x1 + p), min(im.height, y1 + p)))


stages_page = (REPO / "docs" / "stages.html").read_text()
blocks = dict(re.findall(r'(\d): "([^"]+)"', stages_page))
st_ids = [i for k in sorted(blocks) for i in blocks[k].split()]
st_lab = np.array([int(k) for k in sorted(blocks) for _ in blocks[k].split()])
if STAGE_CACHE.exists():
    SL = np.load(STAGE_CACHE)["logits"]
else:
    st_imgs = [square_pad(crop_to_fish(next((ROOT / "rembg_FishImages").glob(f"{i}_*rembg.jpg"))), "black") for i in st_ids]
    SL = model.predict(A.mobilenet_v2.preprocess_input(np.stack([np.asarray(im.resize((224, 224), Image.BICUBIC), np.float32)
                                                                   for im in st_imgs])), verbose=0)
    np.savez_compressed(STAGE_CACHE, logits=SL, ids=np.array(st_ids), stage=st_lab)
# DEEPFIN's own maps (MobileNetV2 features, UMAP) as shown on the species and stages pages
spot = json.loads(re.search(r"window\.SPOT = (.*);", (REPO / "docs" / "spot_species" / "data.js").read_text(), re.S).group(1))
stage_umap = json.loads(re.search(r"^const UMAP = (.*);$", stages_page, re.M).group(1))
lab_ids = [i for i, keep in zip(ids, lab) if keep]
data = {
    "layers": [{"name": n, "caption": c, "shape": "×".join(map(str, o.shape[1:]))} for (n, c), o in zip(LAYERS, maps_model.outputs)],
    "top1_anemone": int(np.sum(top1[lab] == 393)), "top1_hairslide": int(np.sum(top1[lab] == 584)), "n_fish": int(lab.sum()),
    "demo": demo,
    "sets": {"species": make_set(L, y, lab_ids, OBJECTS, "spot_species/%s_t.jpg", spot["models"]["MobileNetV2"]["xy"]),
             "stages": make_set(SL, st_lab, st_ids, STAGE_OBJECTS, "stages/%s_t.jpg", stage_umap["MobileNetV2"]["xy"])},
    "top1_puffer": int(np.sum(np.exp(SL).argmax(1) == 397)), "n_stage_fish": len(st_ids),
}
for k, v in data["sets"].items():
    o = v["objects"]; b = v["best"]
    print(f"{k}: best pair {o[b['pair'][0]]['name']} + {o[b['pair'][1]]['name']} = {b['n']}/{v['n']}; DEEPFIN map {v['deepfin']}/{v['n']}")
(OUT / "data.js").write_text("// generated by site_build/make_exhibit_data.py\nwindow.AIE = " + json.dumps(data, separators=(",", ":")) + ";\n")
print("wrote", OUT, f"({len(list(OUT.iterdir()))} files)")
