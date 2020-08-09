import argparse

import pandas as pd
from flask import Flask, jsonify, request

from cian_similarity import Model
from cian_similarity.utils import (_get_features, category_dummies,
                                   features_index)

app = Flask(__name__)
model = Model()


@app.route("/predict", methods=["POST"])
def predict():
    # TODO: Improve speed using batch processing, not one-by-one
    data = request.get_json(force=True)
    result = []

    for sample in data:
        left, right = [_get_features(el) for el in process_request(sample)]
        left = left.reindex(features_index)
        right = right.reindex(features_index)

        x = model.get_residual_inference(left, right)
        proba = model.clf.predict_proba(x.values.reshape(1, -1))
        result += proba.tolist()

    return jsonify(result)

@app.route("/save", methods=["GET"])
def save_model():
    model.save('saved_model.pkl')


def process_request(r: str):
    row = pd.Series(r)
    left = row[row.index.str.endswith('_x')]
    right = row[row.index.str.endswith('_y')]
    left.index = left.index.map(lambda x: x[:-2])  # _x
    right.index = right.index.map(lambda x: x[:-2])  # _y

    # inplace dummy fitter, may be generalized via defining additional function
    dummies_category = pd.Series(0, index=category_dummies)    

    left = pd.concat([left, dummies_category])
    right = pd.concat([right, dummies_category])

    left["category_" + left['category']] = 1
    left["category_" + right['category']] = 1
    # --

    assert left.shape == right.shape, 'sanity check'
    return left, right


def parse_args():
    description = ('Description')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.train is None:
        model = model.load('lgb.pkl')
    else:
        model.train()

    app.run()
