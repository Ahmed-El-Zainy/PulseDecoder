#!/usr/bin/env python
"""Challenge harness — do not edit. Loads the model and classifies each record."""

import os
import sys

import numpy as np
from scipy.io import loadmat

from run_12ECG_classifier import load_12ECG_model, run_12ECG_classifier


def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x["val"], dtype=np.float64)
    with open(filename.replace(".mat", ".hea"), "r") as f:
        header_data = f.readlines()
    return data, header_data


def save_challenge_predictions(output_directory, filename, scores, labels, classes):
    recording = os.path.splitext(filename)[0]
    output_file = os.path.join(output_directory, filename.replace(".mat", ".csv"))
    with open(output_file, "w") as f:
        f.write("#{}\n".format(recording))
        f.write(",".join(classes) + "\n")
        f.write(",".join(str(int(i)) for i in labels) + "\n")
        f.write(",".join("{:.6f}".format(float(i)) for i in scores) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("Usage: python driver.py <model_dir> <input_dir> <output_dir>")

    model_input, input_directory, output_directory = sys.argv[1:4]

    input_files = [
        f for f in os.listdir(input_directory)
        if f.lower().endswith(".mat") and not f.startswith(".")
    ]
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    print("Loading 12ECG model...")
    model = load_12ECG_model(model_input)

    print("Classifying recordings...")
    for i, f in enumerate(input_files):
        print("    {}/{}...".format(i + 1, len(input_files)))
        data, header_data = load_challenge_data(os.path.join(input_directory, f))
        labels, scores, classes = run_12ECG_classifier(data, header_data, model)
        save_challenge_predictions(output_directory, f, scores, labels, classes)

    print("Done.")
