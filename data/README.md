# Dataset Information

## 📊 Dataset

EmoLens is planned to use **FERPlus** as the preferred dataset for facial emotion recognition.

FERPlus provides improved emotion annotations compared with the original FER2013 dataset by using multiple annotator labels to produce more reliable emotion categories.

## 🎯 Selected Emotion Classes

EmoLens will use exactly five emotion classes:

* Angry
* Happy
* Neutral
* Sad
* Surprise

Other available emotion categories will not be used in the final classifier.

## 🔄 Fallback Dataset

If FERPlus cannot be accessed or used because of availability, licensing, labeling, or distribution constraints, **FER2013** will be evaluated as the fallback dataset.

The final dataset choice will be confirmed after verifying:

* Dataset availability
* Emotion labels
* Licensing and usage conditions
* Distribution/access requirements
* Suitability for the five-class problem

## 📁 Dataset Storage

Downloaded datasets will be stored locally under:

```text
data/
├── raw/
└── processed/
```

### `raw/`

Contains the original downloaded dataset without modification.

### `processed/`

Contains data prepared for model training, validation, and testing.

## 🔒 GitHub Policy

Dataset files will **not** be committed to the GitHub repository.

The `.gitignore` file excludes:

```text
data/raw/
data/processed/
```

This keeps the repository lightweight and avoids redistributing dataset files without confirming the applicable terms.

## 📝 Dataset Source

The exact dataset source, download method, version, license, and access requirements will be recorded here after the dataset has been verified and selected.

## ⚠️ Important Note

The dataset is intended for research and educational development of the EmoLens emotion recognition system. Facial expressions should not be treated as a definitive measurement of a person's actual emotional or mental state.
