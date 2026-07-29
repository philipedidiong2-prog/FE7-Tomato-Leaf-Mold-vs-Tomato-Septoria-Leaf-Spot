# FE7-Tomato-Leaf-Mold-vs-Tomato-Septoria-Leaf-Spot

# Tomato Leaf Disease Classifier: Leaf Mold vs Septoria Leaf Spot

# Contributors:
## Robinson Ekemini Kenneth - 23/EG/FE/656
## Wilson, Deborah Ezekiel - 22/EG/FE/646
## Philip, Edidiong Linus - 23/EG/FE/036
## Blessing Anietie James - 23/EG/FE/046
## Uko, Augustine Stanislaus - 23/EG/FE/006
## Umoh, Uyai Marttins - 22/EG/FE/636

Binary image classifier that distinguishes Tomato Leaf Mold from Tomato
Septoria Leaf Spot, built with transfer learning on EfficientNetB0.

## Dataset

Source: [naveedgull/tomato-leaf-disease](https://www.kaggle.com/datasets/naveedgull/tomato-leaf-disease)
on Kaggle (PlantVillage-derived, lab-condition images: uniform background,
controlled lighting).

**Known limitation:** this dataset is not field imagery. A model trained on
it learns to separate these two classes under lab conditions and should not
be assumed to generalize to real field photos (soil, overlapping leaves,
variable lighting) without further validation on out-of-domain data.

## Project structure

```
.
├── notebook.ipynb          # data prep, training, evaluation
├── app.py                  # Streamlit inference app
├── requirements.txt
├── models/
│   └── efficientnet_transfer_best.keras   # produced by training, not included here
└── results/
    └── tomato_leaf_lab/     # saved plots: samples, learning curves, confusion matrix
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

Run the notebook top to bottom. Order matters — later cells depend on
variables (`SEED`, `class_dirs`, `class_weight_dict`) defined earlier. Steps:

1. Downloads the dataset via `kagglehub`.
2. Matches the two class folders by keyword (`leaf_mold` / `mold`,
   `septoria`) — **before trusting this**, print the actual folder names
   from the `os.walk` output and confirm the match is correct.
3. Checks for near-duplicate images (perceptual hashing) — flag and dedupe
   before splitting if the dataset contains augmented copies of the same
   source leaf. Skipping this risks the same source image appearing in both
   train and test, which inflates test accuracy without you knowing it.
4. Splits into train/validation/test (70/15/15), rebuilding the output
   directory from scratch on every run (stale folders from a previous run
   will otherwise get merged into the new split — this has already caused
   one wrong-class-count bug in this project, don't reintroduce it).
5. Builds an EfficientNetB0 transfer-learning model, computes class weights,
   trains, evaluates (accuracy, precision, recall, F1, confusion matrix).

Trained model is saved to `models/efficientnet_transfer_best.keras` by the
`ModelCheckpoint` callback.

## Running the app

```bash
streamlit run app.py
```

Requires `models/efficientnet_transfer_best.keras` to exist at that relative
path (or edit `MODEL_PATH` in `app.py`). Upload a leaf image; the app
returns the predicted class, a confidence score, and flags predictions
below a confidence threshold as unreliable.

## Evaluation notes

Accuracy alone is not sufficient for this problem — the two classes are
visually confusable, and a high overall accuracy can hide a lopsided
false-positive/false-negative split. Check the confusion matrix and
per-class precision/recall before drawing conclusions, and treat any single
number reported without those alongside it as incomplete.
