import argparse
import pickle

import torch

from emopia_transformer.model import TransformerModel, network_paras
from emopia_transformer.utils import write_midi

PRETRAINED_MODEL_PATH = "emopia_transformer/pretrained_transformer/loss_25_params.pt"
DICTIONARY_PATH = "emopia_transformer/co-representation/dictionary.pkl"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--emotion_class",
        type=int,
        default=4,
        help="the target emotion class you want. It should belongs to [1,2,3,4].",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="test",
        help="Name for the output files",
    )
    args = parser.parse_args()

    output_name = args.output_name
    emotion_class = args.emotion_class

    dictionary = pickle.load(open(DICTIONARY_PATH, "rb"))
    event2word, word2event = dictionary

    # config
    n_class = []  # num of classes for each token
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

    # Generate
    midi_path = f"{output_name}.mid"
    audio_path = f"{output_name}.mp3"

    res, _ = net.inference_from_scratch(
        dictionary, emotion_class, n_token=8, display=False
    )
    write_midi(res, midi_path, word2event)
