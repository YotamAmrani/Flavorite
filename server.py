from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
import json
import re
import pickle
from sklearn.decomposition import NMF as SK_NMF
from sklearn.decomposition import PCA

FILE_NAME = "results_en - Form Responses 2.csv"


def parse(file):
    """
    the function receives a CSV file and parse it into pandas df
    :param file:
    :return: parsed file
    """
    # read relevant columns
    df = pd.read_csv(file, nrows=1)  # read just first line for columns
    columns = df.columns.tolist()  # get the columns
    cols_to_use = columns[3:len(columns) - 1]  # drop the last one
    df = pd.read_csv(file, usecols=cols_to_use)

    # edit columns names
    cols_to_use = [re.search(r"[\w\d\s]*", c).group(0) for c in cols_to_use]
    df.columns = [c.rstrip() for c in cols_to_use]
    # replace the headlines

    # count values:

    # create new table and sort by values
    df_values = df.count().transpose()
    df_values = df_values.sort_values()
    # print(df_values[-10:])
    #
    values = df_values.head(df.shape[1] - df.shape[0]).to_frame().transpose()
    values = list(values.columns.values)

    df = df.drop(columns=values)
    # print(df.shape)

    return df


def parse_users():
    print()



# help functions for prediction
def create_vector(rate, rest, columns):
    rate = np.array(rate).astype(float)
    new_rate = []
    for r in rate:
        if (r / 20 < 1) and (r / 20 > 0):
            new_rate.append(1)
        elif r / 20 >= 1:
            new_rate.append(round(r / 20))
    m = np.mean(new_rate)

    vector = pd.DataFrame(np.array(list([m for i in range(len(columns))])).reshape(1, len(columns)))
    vector.columns = columns
    for i in range(len(rest)):
        vector.at[0, rest[i]] = new_rate[i]

    return vector


def train_model(df, n=50, filling='U', model='SVD'):
    train_full = 0
    if filling == 'U':
        train_full = fill_with_ave_per_user(df).dropna()
    elif filling == 'R':
        train_full = fill_with_ave_per_rest(df).dropna()
    elif filling == 'A':
        train_full = fill_with_ave(df)
        # train_full = df - df.dropna().to_numpy().mean()

    if model == 'SVD':
        pickle.dump(model, open('svd_model.sav', 'wb'))
        return PCA(n_components=n).fit(train_full)
    elif model == 'NMF':
        pickle.dump(model, open('nmf_model.sav', 'wb'))
        return SK_NMF().fit(train_full)


def get_prediction(user_input, rest, model_type='NMF'):

    model = train_model(df)

    # data = fill_with_ave_per_user(user_input).fillna(user_input.to_numpy().mean())
    data = model.inverse_transform(model.transform(user_input))
    data = pd.DataFrame(data.reshape(1, len(df.columns.values)))
    data.columns = df.columns.values
    data = data.sort_values(by=0, axis=1, ascending=False)

    result = [d for d in data if d not in rest]

    return result


def fill_with_ave_per_user(df):
    return df.T.fillna(df.mean(axis=1)).T


def fill_with_ave(df):
    temp = df.fillna(0).to_numpy().mean()
    return df.fillna(temp)

def fill_with_ave_per_rest(df):
    return df.fillna(df.mean())


# server properties
app = Flask(__name__)
cors = CORS(app)


@app.route('/get_rating')
def get_rating():
    ratings = request.args.get('rating', '')
    rest, rate, algo = json.loads(ratings)
    rate = np.array(list(map(int, rate)))
    new_rest = []
    new_rate = []
    for i in range(len(rest)):
        if rate[i] != 0:
            new_rate.append(rate[i])
            new_rest.append(rest[i])

    user_sample = create_vector(new_rate, new_rest, df.columns.values)
    p = get_prediction(user_sample, new_rest, algo[0])
    # p = user_sample
    # result = list(p.columns.values)
    jsn = json.dumps(p[:5])
    return jsn


if __name__ == '__main__':
    df = parse(FILE_NAME)
    app.run(port=5005)
