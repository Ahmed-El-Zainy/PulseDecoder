#!/usr/bin/env python
"""Challenge harness — do not edit. Trains the model and saves it to disk."""

import os
import sys

from train_12ECG_classifier import train_12ECG_classifier

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python train_model.py <training_data> <model_dir>")

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    print("Running training code...")
    train_12ECG_classifier(input_directory, output_directory)
    print("Done.")
