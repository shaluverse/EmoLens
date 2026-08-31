# EmoLens — Project Notes

This file records important project decisions, development notes, and changes made during the development of EmoLens.

## 📌 Current Status

**Phase:** Phase -1 — Repository Initialization

**Status:** In progress

## ✅ Decisions Made

### Project Name

The official project name is **EmoLens**.

### Emotion Classes

The final classifier will recognize exactly five emotions:

1. Angry
2. Happy
3. Neutral
4. Sad
5. Surprise

### Input

The initial version will accept:

* JPG
* JPEG
* PNG

The first version will use uploaded images rather than a live webcam.

### Dataset

**FERPlus** is the preferred dataset.

**FER2013** will be considered as a fallback after verifying dataset availability, labels, licensing, and distribution requirements.

### Model

The initial model will be a **Convolutional Neural Network (CNN)**.

The baseline model will be kept relatively small so that development and experimentation remain manageable.

### GitHub

GitHub will be used from the beginning of the project.

The repository will be public.

The following will not be committed:

* Raw datasets
* Processed datasets
* API keys or secrets
* Virtual environments
* Trained `.keras` model files

### Development Approach

The project will be developed in phases.

Each phase should be completed and tested before moving to the next phase.

Git commits will be made at meaningful milestones.

## 🗺️ Planned Phases

### Phase -1 — Repository Initialization

* Create GitHub repository
* Connect local repository
* Add `.gitignore`
* Create project structure
* Create initial documentation
* Make first commit and push

### Phase 0 — Environment & Project Setup

* Create Python environment
* Install and verify dependencies
* Add configuration
* Finalize project structure
* Create initial tests

### Phase 1 — Dataset

* Verify FERPlus availability and terms
* Download and inspect dataset
* Analyze class distribution
* Select the five required classes
* Prepare training, validation, and test data

### Phase 2 — Baseline CNN

* Build the initial CNN
* Train the model
* Evaluate performance
* Save the trained model locally
* Analyze errors and class performance

### Phase 3 — Face Detection & Prediction Pipeline

* Detect faces in uploaded images
* Preprocess detected faces
* Load the trained model
* Generate emotion predictions
* Handle invalid or unsupported images

### Phase 4 — Frontend

* Build the user interface
* Add image upload
* Display the uploaded image
* Display detected face
* Show predicted emotion
* Add confidence information where appropriate

### Phase 5 — Integration & Testing

* Connect frontend and backend
* Test the complete workflow
* Add edge-case handling
* Improve error messages
* Add automated tests

### Phase 6 — Finalization

* Improve UI/UX
* Clean project code
* Update documentation
* Verify `.gitignore`
* Prepare GitHub repository
* Add final project screenshots
* Prepare the project for portfolio/resume use

## 📝 Development Log

### 2026-08-31

* Created public GitHub repository.
* Connected local repository to GitHub.
* Added `.gitignore`.
* Created initial project folder structure.
* Created `README.md`.
* Created `data/README.md`.
* Created `NOTES.md`.
* Dataset has not yet been downloaded.
* Model development has not yet started.

### Phase 0 Completion — 2026-08-31

* Created the dedicated `emo_lens` Conda environment.
* Configured Python 3.11.16.
* Installed and verified TensorFlow 2.15.1 and core dependencies.
* Added pinned dependencies to `requirements.txt`.
* Added `src/config.py` for centralized project configuration.
* Added `src/__init__.py` to make `src` a Python package.
* Added the initial `tests/test_config.py` test suite.
* Verified the configuration using pytest.
* All 4 configuration tests pass.
* Phase 0 environment setup is complete.
* Next phase: Phase 1 — Dataset verification and preparation.
