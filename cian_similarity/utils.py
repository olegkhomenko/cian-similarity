import json
from pprint import pprint
from typing import Optional

import pandas as pd
import psycopg2
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import \
    train_test_split  # no kfold here, only train-val-test

db_config_default = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "test-task",
    "host": "localhost",
    "port": 8213,
}


request_keys = ('bargainterms',
                'building',
                'category',
                'description',
                'flattype',
                'floornumber',
                'geo',
                'offer_id',
                'publisheduserid',
                'roomscount',
                'totalarea',
                'userid',
                )


category_dummies = [  # We don't use any dummy encoder w/ fit-predict interface, thus we store column names
    'category_cottageRent', 'category_cottageSale',
    'category_dailyFlatRent', 'category_dailyHouseRent',
    'category_dailyRoomRent', 'category_flatRent', 'category_flatSale',
    'category_flatShareSale', 'category_houseRent',
    'category_houseSale', 'category_houseShareRent',
    'category_houseShareSale', 'category_landSale',
    'category_newBuildingFlatSale', 'category_roomRent',
    'category_roomSale', 'category_townhouseRent',
    'category_townhouseSale'
]

features_index = [  # Used to fill null values if some value are missed
    'category_cottageRent', 'category_cottageSale',
    'category_dailyFlatRent', 'category_dailyHouseRent',
    'category_dailyRoomRent', 'category_flatRent', 'category_flatSale',
    'category_flatShareSale', 'category_houseRent', 'category_houseSale',
    'category_houseShareRent', 'category_houseShareSale',
    'category_landSale', 'category_newBuildingFlatSale',
    'category_roomRent', 'category_roomSale', 'category_townhouseRent',
    'category_townhouseSale', 'house', 'lat', 'lng', 'street', 'totalarea',
    'totalarea_diff'
]


def get_connection(db_config: Optional[dict] = None):
    if db_config is None:
        db_config = db_config_default

    return psycopg2.connect(**db_config)


def drop_tables(conn: psycopg2.extensions.connection) -> None:
    psql_cursor = conn.cursor()
    table_names = ["offers", "pairs"]
    for t in table_names:
        psql_cursor.execute(f"drop table {t};")
    psql_cursor.close()


def get_offers(conn: psycopg2.extensions.connection) -> pd.DataFrame:
    offers = pd.read_sql_query('select * from offers', con=conn)
    offers = pd.get_dummies(offers, columns=['category'])
    return offers


def get_pairs(conn: psycopg2.extensions.connection) -> pd.DataFrame:
    return pd.read_sql_query('select * from pairs', con=conn)


def _get_features(row: pd.Series) -> pd.Series:
    # WARN: This is a dangerous type of parsing JSON-like strings in production environment.
    # Be aware of python-code injection
    geo = eval(row['geo'])
    # ---
    result = {}
    result['offer_id'] = row['offer_id']
    result['lat'] = geo['coordinates']['lat']
    result['lng'] = geo['coordinates']['lng']

    result = {  # Category {category_roomRent, category_landSale, ...} 
        **result,
        **row[row.index.str.startswith('category_')].to_dict()}

    result['totalarea'] = row['totalarea']

    # we need only street and house types
    result = {**result, **{el['type']: el['id'] for el in geo['address'] if el['type'] in ('street', 'house')}}
    return pd.Series(result)


def get_features(offers: pd.DataFrame) -> pd.DataFrame:
    feats = offers.apply(_get_features, axis=1)
    feats = feats.set_index('offer_id')
    return feats


def get_residual(row: pd.Series):
    global feats
    left = feats.loc[row.offer_id1]
    right = feats.loc[row.offer_id2]

    residual = abs(left - right)
    residual = residual.fillna(-1)
    residual['totalarea_diff'] = residual['totalarea'] / max(left['totalarea'], right['totalarea'])

    return residual


def get_residual_inference(left: pd.Series, right: pd.Series):
    residual = abs(left - right)
    residual = residual.fillna(-1)
    residual['totalarea_diff'] = residual['totalarea'] / max(left['totalarea'], right['totalarea'])

    return residual


def calc_metrics(y_true, y_pred) -> dict:
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'mean_pred': y_pred.mean(),
        'mean_true': y_true.mean(),
    }

    return metrics
