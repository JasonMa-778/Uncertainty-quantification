# Project: Uncertainty Quantification with Ensembles and Transformers

This repository contains the code used for the experiments described in the following paper:

👉 [Overleaf Paper](https://www.overleaf.com/read/qxtrwqmtstvs#ce76dd)

> **Note**: The code may not run directly due to the absence of some necessary files (e.g., stored results), but it provides a valuable reference for data processing, modeling, and visualization.

---

## 📁 Project Structure

### 1. Simulation Experiments

#### • `data/`
Contains the code for generating synthetic datasets used in simulations.

#### • `models/`
Implements three types of models for uncertainty quantification.

#### • `train/` and `utils/`
Includes training scripts and helper functions.

#### • `visualization/`
Code for visualizing the results of simulation experiments.

#### • `epi.ipynb` & `OOD.ipynb`
- `epi.ipynb`: Trains and evaluates models for **epistemic uncertainty quantification**.
- `OOD.ipynb`: Evaluates models on **out-of-distribution (OOD) detection**.

#### • `vis/`
Supplementary scripts for visualizing model behavior.

---

### 2. Real-World Experiments

This part includes experiments on **six public datasets**:

- **IMDB**
- **Yelp**
- **RCT**
- **MIMIC-III (mortality prediction)**
- **Wafer**
- **ECG5000**

These datasets are publicly available and can be downloaded online.

#### • IMDB
- `IMDB.ipynb`: Uses BERT + Ensemble to quantify uncertainty.

#### • Yelp
- `yelp.ipynb`: Uses BERT + Ensemble for uncertainty quantification.

#### • RCT
- `RCT.ipynb`: Uses BERT + Ensemble to model uncertainty in RCT classification.

#### • MIMIC-III
- `models/`: Contains transformer-based ensemble model code.
- See `mimic3.md` for detailed instructions.

#### • Time Series (Wafer & ECG5000)
- Scripts to apply ensemble methods for uncertainty quantification in time-series data.

---

## 📊 Visualization

- `Final_visl.ipynb` and `Trun_visl.ipynb`: Notebooks for visualizing experimental results.

---

## 🔧 Notes

- Some necessary results or preprocessed files may be missing.
- Please refer to the code structure and logic for implementation references.

---

## 📬 Contact
For questions, please refer to the paper link or contact me: zm91@duke.edu.
