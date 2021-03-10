import pandas as pd

import pickle

config = {
    'path': './',
}

def prepare(data, path_to_data):

    tfidf = pickle.load(open(path_to_data + 'tfidf', 'rb'))
    item_name_tfidf = tfidf.transform(data['item_name'])

    X_train = item_name_tfidf

    return X_train


def predict(path_to_data):
    #Берутся первые 1000, чтобы не грузить ноутбук
    data = pd.read_parquet(path_to_data + 'test_pipeline.parquet')[:1000]

    X_train = prepare(data, path_to_data)

    model = pickle.load(open(path_to_data + 'clf_model', 'rb'))
    pred = model.predict(X_train)

    res = pd.DataFrame(pred, columns=['pred'])
    res['id'] = data.index.values

    res[['id', 'pred']].to_csv('answers.csv', index=None)
    print(res)
    
if __name__=='__main__':
    predict(config['path'])
