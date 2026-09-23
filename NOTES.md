# EmoLens — Project Notes

Short record of important decisions, progress, and final results.

---

## Project Status

**Status:** Completed

**Project:** EmoLens  
**Input:** JPG, JPEG, PNG images  
**Classes:** Angry, Happy, Neutral, Sad, Surprise  
**Model:** Custom CNN  
**Dataset:** FER2013 + official FERPlus annotations  
**Webcam:** Not included

---

## Phase 1 — Dataset Pipeline

**Status:** Complete

- FER2013 and FERPlus data aligned.
- 35,887 samples checked.
- 34,039 samples retained.
- 1,848 ambiguous samples discarded.
- Five emotion classes prepared.
- Train, validation, and test sets created.
- Dataset integrity check passed with 0 corrupted images.
- Images converted to 48 × 48 grayscale.

---

## Phase 2 — CNN Training & Evaluation

**Status:** Complete

- 30 epochs completed.
- Best validation accuracy: **82.00%**
- Final validation accuracy: **81.82%**
- Test samples: **3,414**
- Test accuracy: **81.17%**
- Macro F1: **0.7855**
- Weighted F1: **0.8121**

### Class F1 Scores

| Emotion | F1 Score |
|---|---:|
| Angry | 71.99% |
| Happy | 90.18% |
| Neutral | 82.25% |
| Sad | 62.19% |
| Surprise | 86.14% |

The trained model is stored locally as:

`models/emolens_cnn.keras`

The model file is excluded from GitHub.

---

## Phase 3 — Prediction & Application

**Status:** Complete

Implemented:

- OpenCV face detection.
- Single-face validation.
- Face cropping and preprocessing.
- CNN inference.
- Emotion and confidence prediction.
- Emotion probability display.
- Emoji-based result.
- No-face handling.
- Multiple-face handling.
- Streamlit web interface.
- Light/dark mode.

### Face Handling

- **0 faces:** Reject image.
- **1 face:** Analyze the face.
- **2+ faces:** Reject image.

---

## Final Verification

The application was tested successfully with:

- Single-face image.
- No-face image.
- Multiple-face image.
- Image upload validation.
- Emotion prediction.
- Probability display.
- Light/dark interface.

---

## Final Run Command

```powershell
conda activate emo_lens
streamlit run frontend/app.py