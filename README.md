# Detecting Overconfident Errors in Deep Learning Models Using Calibration and Transformer-Based Approaches

**Author: Bhavesh Uttarkar**

## Overview

This project focuses on identifying overconfident errors in deep learning models. In many cases, models produce predictions with very high confidence even when they are incorrect. This project builds a system to detect such cases and improve the reliability of model predictions.

The workflow includes training a classification model, extracting prediction probabilities, and using additional models to detect whether a prediction is overconfident and incorrect. The project also analyzes calibration using Expected Calibration Error and visualizations.

---

## Setup Instructions

Clone the repository

```
git clone https://github.com/username/overconfidence-detection.git
cd overconfidence-detection
```

Install dependencies

```
pip install -r requirements.txt
```

---

## Dependencies

The project uses the following main libraries

* torch
* numpy
* matplotlib

All required packages are listed in the requirements.txt file.

---

## Run Instructions

Execute the main script

```
python main.py
```

---

## Outputs

After running the project, the following outputs will be generated in the outputs/plots folder

* confidence_hist.png
* accuracy.png
* conf_vs_correct.png
* error_distribution.png
* reliability.png

The console will also display key metrics such as accuracy and calibration error.

---

## Project Structure

```
overconfidence-detection/
│
├── main.py
├── config.py
├── requirements.txt
│
├── data/
│   └── data_loader.py
│
├── models/
│   ├── cnn.py
│   ├── mlp.py
│   └── transformer.py
│
├── training/
│   └── train.py
│
├── evaluation/
│   └── evaluate.py
│
├── utils/
│   ├── metrics.py
│   └── plots.py
│
└── outputs/
    ├── plots/
    └── results/
```

---

## Description

The project follows a two-stage pipeline

First, a convolutional neural network is trained for image classification and produces prediction probabilities

Second, these probabilities are analyzed to detect overconfident errors using different approaches such as thresholding, neural networks, and transformer-based models

The system also evaluates calibration and generates visualizations to understand model behavior

---

## Notes

The project is designed to run quickly and generate all outputs in a single execution

It demonstrates how confidence-based analysis can improve trust in deep learning systems
