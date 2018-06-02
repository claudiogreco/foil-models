import csv
import json
from argparse import ArgumentParser

import spacy

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--foil_split_filename", type=str, required=True)
    parser.add_argument("--preprocessed_foil_split_filename", type=str, required=True)
    args = parser.parse_args()

    images = {}

    with open(args.foil_split_filename) as in_file:
        dataset = json.load(in_file)
        nlp = spacy.load("en_core_web_sm")

        for image in dataset["images"]:
            images[image["id"]] = image["file_name"]

        with open(args.preprocessed_foil_split_filename, mode="w") as out_file:
            writer = csv.writer(out_file, delimiter="\t")

            for i, annotation in enumerate(dataset["annotations"]):
                print("Processing sentence [{}/{}]".format(i, len(dataset["annotations"])))
                caption = [token.lower_ for token in nlp(annotation["caption"])]
                image = images[annotation["image_id"]]
                label = "yes" if annotation["foil_word"] == "ORIG" else "no"
                writer.writerow([label, " ".join(caption), image])
