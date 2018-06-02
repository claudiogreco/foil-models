import csv
import json

import numpy as np

from utils import pad_sequences


def load_foil_dataset(filename, token2id, label2id):
    labels = []
    padded_sentences = []
    image_names = []
    original_sentences = []

    with open(filename) as in_file:
        reader = csv.reader(in_file, delimiter="\t")

        for row in reader:
            label = row[0].strip()
            sentence_tokens = row[1].strip().split()
            image = row[2].strip()
            labels.append(label2id[label])
            padded_sentences.append([token2id.get(token, token2id["#unk#"]) for token in sentence_tokens])
            image_names.append(image)
            original_sentences.append(sentence)

        padded_sentences = pad_sequences(padded_sentences, padding="post", value=token2id["#pad#"], dtype=np.long)
        labels = np.array(labels)

    return labels, padded_sentences, image_names, original_sentences


class ImageReader:
    def __init__(self, img_names_filename, img_features_filename):
        self._img_names_filename = img_names_filename
        self._img_features_filename = img_features_filename

        with open(img_names_filename) as in_file:
            img_names = json.load(in_file)

        with open(img_features_filename, mode="rb") as in_file:
            img_features = np.load(in_file)

        self._img_names_features = {filename: features for filename, features in zip(img_names, img_features)}

    def get_features(self, images_names):
        return np.array([self._img_names_features[image_name] for image_name in images_names])
