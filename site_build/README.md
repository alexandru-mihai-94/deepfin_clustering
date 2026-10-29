# site_build: how the website in `docs/` is made

The pages in `docs/` are plain HTML. The images and `data.js` files they load are made by the scripts here.
Most of them read photographs and analysis results that are not in git, because of their size. Those live in
the analysis folder that holds this repository (`DATA` in the scripts): `flank_umap/`, `rembg_FishImages/` and
`inat_wild/`, plus `docs/percula_ocellaris/processed_rembg_v3/` (git-ignored). So the scripts record how the site was
built; they do not run from a fresh clone.

| Script | Makes | Reads |
|---|---|---|
| `flank_representatives.py` | `DATA/flank_umap/representatives.csv`: one photo per fish and flank | `docs/percula_ocellaris/processed_rembg_v3`, `originals_all` |
| `flank_umap_sweep.py` | `DATA/flank_umap/features/` and the model × UMAP comparison figures | `representatives.csv` |
| `spot_species_data.py` | `docs/spot_species/`: species quiz images, thumbnails and maps | `representatives.csv`, features, `DATA/inat_wild/credits.csv` |
| `stage_demo_umap.py` | the three maps inside `docs/stages.html` | `DATA/rembg_FishImages/`, the 144 fish listed in `stages.html` |
| `make_exhibit_data.py` | `docs/ai_eye/`: the classifier page (feature maps, ImageNet scores, fingerprints) | all of the above |
| `make_offline.py` | `docs/sw.js`: the offline cache for the iPad app | every file the pages use |
| `make_standalone.py` | `../deepfin_exhibit_standalone/`: the exhibit as one HTML file (site look, and a white OIST-coloured version) plus `images/`, for embedding elsewhere | `docs/` |

Made once by hand and not by a script: the stage images in `docs/stages/` (cropped to the fish from
`DATA/rembg_FishImages`, 240 px thumbnails and 1200 px quiz images) and the placeholder app icons in `docs/icons/`
(from `docs/stages/C2_28_Final_L9.jpg`; replace them freely, keeping the file names and sizes).

## Running

From the repository root, with a plain venv in this folder (the network outputs are kept in `site_build/cache/`,
so reruns are quick):

```bash
python3.12 -m venv site_build/venv
site_build/venv/bin/pip install -r site_build/requirements.txt   # pinned: UMAP layouts depend on the version
site_build/venv/bin/python site_build/spot_species_data.py
site_build/venv/bin/python site_build/stage_demo_umap.py
site_build/venv/bin/python site_build/make_exhibit_data.py
site_build/venv/bin/python site_build/make_offline.py   # always last, after any change in docs/
```

To preview the site locally: `python3 -m http.server 8000 -d docs`, then open http://localhost:8000.
The offline cache only switches on over https (the published site), so local previews always show the latest edits.
