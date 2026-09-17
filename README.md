# EmoLens 🎭

AI-powered facial emotion recognition system using deep learning.

## 📌 Project Overview

EmoLens is a computer vision application that detects facial expressions from an uploaded image and classifies the detected emotion into one of five categories:

* Happy
* Sad
* Angry
* Surprise
* Neutral

The project uses a Convolutional Neural Network (CNN) trained on a facial emotion dataset and provides the prediction through a simple web interface.

## 🎯 Problem Statement

Human emotions are often expressed through facial expressions. Automatically recognizing these expressions from images is a computer vision problem with applications in human-computer interaction, education, entertainment, accessibility, and other AI systems.

EmoLens aims to build a simple end-to-end emotion recognition system that can detect and classify facial emotions from an image.

## ✨ Objectives

* Detect faces from uploaded images.
* Classify facial expressions into five emotion categories.
* Build and evaluate a CNN-based deep learning model.
* Provide an easy-to-use web interface.
* Create a clean, reproducible, and well-documented AI project.

## 🧠 Emotion Classes

| Class | Emotion  |
| ----- | -------- |
| 0     | Angry    |
| 1     | Happy    |
| 2     | Neutral  |
| 3     | Sad      |
| 4     | Surprise |

## 🏗️ Project Structure

```text
EmoLens/
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── models/
├── src/
├── tests/
├── frontend/
├── .gitignore
├── README.md
├── NOTES.md
└── requirements.txt
```

## 🔄 Planned Workflow

```text
Input Image
     ↓
Face Detection
     ↓
Image Preprocessing
     ↓
CNN Emotion Classifier
     ↓
Emotion Prediction
     ↓
Web Interface
```

## 🛠️ Planned Technology Stack

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Git & GitHub
* Web frontend

## 📊 Dataset

EmoLens uses the original FER2013 image data together with the official
Microsoft FERPlus annotations.

The FER2013 and FERPlus files contain 35,887 aligned samples. FERPlus
annotator votes are used to assign labels rather than relying directly
on the original FER2013 emotion labels.

Only five emotion classes are used:

- Angry
- Happy
- Sad
- Surprise
- Neutral

A sample is retained only when one of these five emotions has a unique
highest vote count greater than zero. Samples with tied highest votes
are discarded to reduce label ambiguity.

After filtering:

- Original samples: 35,887
- Retained samples: 34,039
- Discarded ambiguous samples: 1,848

The processed dataset is divided into training, validation, and test
sets.

Dataset files are excluded from GitHub through `.gitignore`.
See `data/README.md` for dataset preparation and source details.

## ⚠️ Limitations

* The system is designed for five emotion classes only.
* Facial emotion recognition is inherently imperfect because facial expressions do not always represent a person's actual emotional state.
* Model performance depends on image quality, lighting, face visibility, dataset quality, and training.
* FER-based datasets may have class imbalance and labeling uncertainty.
* The initial version will work with uploaded images and will not use a live webcam.

## 🚀 Future Scope

Possible future improvements include:

* Improved model architectures.
* Better face detection and preprocessing.
* Model performance optimization.
* Support for additional emotion classes.
* Real-time webcam-based recognition.
* Deployment as a cloud/web application.

## 📌 Project Status

## 📌 Project Status

**Current Phase:** Phase 1- dataset pipeline

Next: Phase 2 — Baseline CNN development


## 👩‍💻 Author

**Shalini A P**

B.Sc. (Hons) Data Science & Artificial Intelligence

