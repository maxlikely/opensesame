import argparse
import logging
import random

import pandas as pd
from sklearn import cross_validation
from sklearn import ensemble
from sklearn import metrics


logger = logging.getLogger('part1')


def extract_features(x):
    """Basic feature extraction."""

    def explode_pool(pool):
        """One-hot encoding for 'Pool' variable."""
        pools = {}
        if 'community' in pool.lower():
            pools['pool:community'] = True
        if 'private' in pool.lower():
            pools['pool:private'] = True
        return pools

    features = explode_pool(x.Pool)
    features['living_area'] = x.LivingArea
    features['num_bedrooms'] = x.NumBedrooms
    features['num_full_baths'] = x.NumBaths // 1
    features['num_half_baths'] = x.NumBaths % 1
    features['num_stories'] = x.ExteriorStories
    features['dwelling_type:%s' % x.DwellingType] = True
    features['days_on_market'] = (x.CloseDate - x.ListDate).days
    features['close_date'] = x.CloseDate.toordinal()
    features['close_year'] = x.CloseDate.year
    features['close_month'] = x.CloseDate.month
    features['geo_lat'] = x.GeoLat
    features['geo_lon'] = x.GeoLon
    features['list_price'] = x.ListPrice

    return pd.Series(features)


def main(args):
    random.seed(args.random_seed)

    logger.info('loading data')
    converters = {'CloseDate': pd.to_datetime, 'ListDate': pd.to_datetime}
    nrows = None
    if args.debug:
        logger.debug('working with a smaller dataset for debugging purposes')
        nrows = 50
    df = pd.read_csv(args.input, converters=converters, nrows=nrows)

    # ignore any rows with null values for expediency
    df = df.dropna()

    logger.info('extracting features')
    y = df.ClosePrice.values
    X = df.apply(extract_features, axis=1).fillna(False)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y)
    clf = ensemble.RandomForestRegressor()

    # mean value baseline
    y_baseline = [y_train.mean()] * len(y_test)

    # with ListPrice
    logger.info('training models')
    clf.fit(X_train.values, y_train)
    y_pred_with_list_price = clf.predict(X_test.values)

    # without ListPrice
    cols = [col for col in X_train.columns if col != 'list_price']
    clf.fit(X_train[cols].values, y_train)
    y_pred_sans_list_price = clf.predict(X_test[cols].values)

    mae_baseline = metrics.mean_absolute_error(y_test, y_baseline)
    mae_with_list_price = metrics.mean_absolute_error(y_test, y_pred_with_list_price)
    mae_sans_list_price = metrics.mean_absolute_error(y_test, y_pred_sans_list_price)

    pd.options.display.float_format = '${:,.2f}'.format
    results = pd.Series([mae_baseline, mae_with_list_price, mae_sans_list_price],
                        index=['baseline', 'with list price', 'without list price'])
                
    logger.info('Mean Average Error:\n' + str(results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model evaluation script')
    parser.add_argument('input', help='input dataset.csv')
    parser.add_argument('--random-seed', type=int, default=random.randint(0, 1000))
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    log_format = '%(asctime)s : %(levelname)s : %(name)s : %(message)s'
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format=log_format, level=log_level)
    logger.info(args)

    main(args)
