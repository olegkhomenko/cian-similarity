import argparse
import json

import requests


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default='request-example.json')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    with open(args.json_path, 'r') as fin:
        request = json.load(fin)

    res = requests.post('http://127.0.0.1:5000/predict', json=request)
    print(eval(res.text))
