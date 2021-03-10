import pandas as pd

import sys 
import pickle

config = {
    'path': '.',
}

def prepare(data, path_to_data):
    tfidf = pickle.load(open(path_to_data + 'tfidf', 'rb'))
    item_name_tfidf = tfidf.transform(data['item_name'])

    X_train = item_name_tfidf

    pickle.dump(X_train, open(path_to_data + 'prepared_data', 'wb'))


def predict(path_to_data):
    X_train = pickle.load(open(path_to_data + 'prepared_data', 'rb'))

    model = pickle.load(open(path_to_data + 'clf_model', 'rb'))
    pred = model.predict(X_train)

    res = pd.DataFrame(pred, columns=['pred'])
    res['id'] = X_train['id']

    res[['id', 'pred']].to_csv('answers.csv', index=None)


if __name__=='__main__':
    if sys.argv[1] == 'prepare':
        #Берутся первые 1000, чтобы не грузить ноутбук
        data = pd.read_parquet(config['path'] + 'data_fusion_train.parquet')[:1000]

        prepare(data, config['path'])

    elif sys.argv[1] == 'predict':
        predict(config['path'])