"""Adapted from https://github.com/annahung31/EMOPIA/blob/main/workspace/transformer/generate.ipynb"""

import os
import pickle

import torch
from emopia_transformer.model import TransformerModel
from emopia_transformer.utils import write_midi

PRETRAINED_MODEL_PATH = "emopia_transformer/pretrained_transformer/loss_25_params.pt"
DICTIONARY_PATH = "emopia_transformer/co-representation/dictionary.pkl"

try:
    THIS_DIR = os.path.dirname(os.path.realpath(__file__))
except:
    THIS_DIR = os.getcwd()

OUTPUT_DIR = os.path.join(THIS_DIR, "../output")
EMOPIA_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "emopia-output")


if __name__ == "__main__":

    # Prepare the dictionary
    dictionary = pickle.load(open(DICTIONARY_PATH, "rb"))
    event2word, word2event = dictionary

    n_class = []  # num classes for each token
    for key in event2word.keys():
        n_class.append(len(dictionary[0][key]))
    n_token = len(n_class)

    # Initialize model
    net = TransformerModel(n_class, is_training=False)
    net.eval()

    # Load the pre-trained model state
    net.load_state_dict(
        torch.load(PRETRAINED_MODEL_PATH, map_location=torch.device("cpu"))
    )

    # Generate 10 examples for each emotion class
    for emotion_class in range(1, 5):
        for i in range(10):
            res, _ = net.inference_from_scratch(
                dictionary, emotion_class, n_token=8, display=False
            )

            filepath = os.path.join(EMOPIA_OUTPUT_DIR, f"{emotion_class}_{i}.mid")
            write_midi(res, filepath, word2event)
            print(f"Saved {filepath}")
