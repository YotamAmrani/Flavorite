import pandas as pd
import numpy as np
from sklearn.decomposition import NMF as SK_NMF
from sklearn.decomposition import PCA
from sklearn import metrics, preprocessing
from surprise import SVD, NMF
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
from server import parse, FILE_NAME


# ------ Surprise Model ------ #
def sup_model(df, algorithm='SVD'):
    # prepare data
    idx = [('user_' + str(i)) for i in range(df.shape[0])]
    df['user'] = idx
    df = df.melt(id_vars='user')
    df = shuffle(df)
    reader = Reader(rating_scale=(1.0, 5.0))
    data = Dataset.load_from_df(df.dropna(), reader)

    # run evaluation
    if algorithm == 'SVD':
        algo = SVD()
    else:
        algo = NMF()

    cross_validate(algo, data, measures=['RMSE'], cv=10, verbose=True)


def fill_with_average_per_rest(df):
    return df.fillna(df.mean())


def fill_with_average_per_user(df):
    return df.T.fillna(df.mean(axis=1)).T


def fill_with_average(df):
    temp = df.fillna(0).to_numpy().mean()
    return df.fillna(temp)


def split_data(df):
    # df = df.fillna(0)
    msk = np.random.rand(len(df)) < 0.9
    train = df[msk]
    test = df[~msk]
    # print(test.isna().sum()/ train.isna().sum())
    # print(len(test), len(train), len(df))
    return train, test, msk


def get_score(model, data, scorer=metrics.explained_variance_score):
    """ Estimate performance of the model on the data """
    prediction = model.inverse_transform(model.transform(data))
    return scorer(data, prediction)


def pca_chhosing_n(df):
    train, test, msk = split_data(df)
    train_full = fill_with_average_per_user(train).dropna()

    pca = PCA()
    pca.fit(train_full)

    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)')  # for each component
    plt.title('Pulsar Dataset Explained Variance')
    plt.show()


def nmf_choosing_n(df):
    train, test, msk = split_data(df)
    train_full = fill_with_average_per_user(train).dropna()
    test = fill_with_average_per_user(test).dropna()
    ks = [i for i in range(1, 80)]

    perfs_train = []
    perfs_test = []
    for k in ks:
        nmf = SK_NMF(n_components=k).fit(train_full)

        perfs_train.append(get_score(nmf, train_full))
        perfs_test.append(get_score(nmf, test))
        print(k)

    plt.plot(ks, perfs_train, ks, perfs_test)
    plt.show()


def train_model(df, n=50, filling='R', model='SVD'):
    if filling == 'U':
        train_full = fill_with_average(fill_with_average_per_user(df))
    elif filling == 'R':
        train_full = fill_with_average(fill_with_average_per_rest(df))
    elif filling == 'A':
        train_full = fill_with_average(df)
        # train_full = df - df.dropna().to_numpy().mean()

    if model == 'SVD':
        # pickle.dump(model, open('svd_model.sav', 'wb'))
        return PCA(n_components=n).fit(train_full), train_full
    elif model == 'NMF':
        # pickle.dump(model, open('nmf_model.sav', 'wb'))
        return SK_NMF(n_components=n).fit(train_full), train_full


def get_prediction(model, data):
    """ Estimate performance of the model on the data """
    data = fill_with_average_per_user(data).fillna(data.to_numpy().mean())

    return model.inverse_transform(model.transform(data))


def accuracy_rate(model, test_set):
    p = get_prediction(model, test_set)
    p = p.round()
    correct_ones = (p - test_set).fillna(0).astype(bool).to_numpy().sum()
    return correct_ones / test_set.fillna(0).astype(bool).to_numpy().sum()


def new_split(df):
    # prepare df for spliting
    df = df.copy()
    idx = [i for i in range(df.shape[0])]
    df['index'] = idx
    melted_df = df.melt(id_vars='index').dropna()
    melted_df = shuffle(melted_df)

    # split
    msk = np.random.rand(len(melted_df)) < 0.9
    test = melted_df[~msk].copy()

    for index, row in test.iterrows():
        one = test.at[index, 'index']
        two = test.at[index, 'variable']
        df.at[one, two] = np.NaN

    return df.drop(['index'], axis=1), test


def get_prediction_values(df, test):
    prediction = []
    for index, row in test.iterrows():
        one = test.at[index, 'index']
        two = test.at[index, 'variable']
        # print(one,two)
        prediction.append(df.at[one, two])

    return prediction


def new_split_test():
    # prepare data
    df = parse(FILE_NAME)
    train, test = new_split(df)

    # nmf_choosing_n(df)
    # sup_model(df, 'SVD')
    # train, test, msk = split_data(df)
    model = train_model(train, n=25, filling='R', model='SVD')
    # model = train_model(train, n=10, filling='R', model='NMF')

    matrix = pd.DataFrame(
        model.inverse_transform(model.transform(fill_with_average(fill_with_average_per_user(train)))))
    matrix.columns = df.columns.values
    # matrix[df.columns.values] = m
    p = np.array(get_prediction_values(matrix, test))
    real = test['value'].to_numpy()
    print(metrics.mean_squared_error(real, p))
    temp = np.array(p.round() - real) != 0
    print(temp.sum() / temp.shape[0])
    print()

    # comparing sum of mistakes
    # print(accuracy_rate(model, test), accuracy_rate(model2, test))

    # testing(df)


def finding_n():
    # get data
    df = parse(FILE_NAME)
    perfs_train = []
    perfs_test = []
    acc_rate_test = []
    acc_rate_train = []
    for k in range(1, 80):
        mean_perfs_train = 0
        mean_perfs_test = 0
        mean_acc_rate_test = 0
        mean_acc_rate_train = 0

        for i in range(10):
            train, test = new_split(df)
            model, train_full = train_model(train, k, 'A', 'SVD')

            # get prediction by train
            matrix = pd.DataFrame(
                model.inverse_transform(model.transform(train_full)))
            matrix.columns = df.columns.values

            # get the test values
            p = np.array(get_prediction_values(matrix, test))
            real = test['value'].to_numpy()
            temp = np.array(p.round() - real) != 0
            acc = temp.sum() / temp.shape[0]
            # mse  and accuracy for test
            mean_perfs_test += metrics.mean_squared_error(p, real)
            mean_acc_rate_test += acc

            # get the train values
            # mse = np.array((matrix.as_matrix() - df.as_matrix()) ** 2).mean(ax=None)
            # temp = np.array(matrix.round() - real) != 0
            # acc = temp.sum() / temp.shape[0]
            # mse for train
            # mean_perfs_train += metrics.explained_variance_score(matrix, train_full)
            # mean_acc_rate_train += accuracy_rate(model, train_full)

        perfs_test.append(mean_perfs_test / 10)
        # perfs_train.append(mean_perfs_train / 10)
        acc_rate_test.append(mean_acc_rate_test / 10)
        # acc_rate_train.append(mean_acc_rate_train / 5)
        print(k)


# model = train_model(df, n=50, filling='A', model=model_type)
# df = parse(FILE_NAME)
# train_model(df, model='NMF')
if __name__ == '__main__':
    df = parse(FILE_NAME)

    acc = [0, 0, 0, 0, 0, 0]
    mse = [0, 0, 0, 0, 0, 0]

    for i in range(50):
        train, test = new_split(df)
        model1, train_full = train_model(train, 50, 'A', 'SVD')
        model2, train_full = train_model(train, 50, 'U', 'SVD')
        model3, train_full = train_model(train, 50, 'R', 'SVD')
        model4, train_full = train_model(train, 50, 'A', 'NMF')
        model5, train_full = train_model(train, 50, 'U', 'NMF')
        model6, train_full = train_model(train, 50, 'R', 'NMF')
        # ----------------------------------
        matrix = pd.DataFrame(
            model1.inverse_transform(model1.transform(train_full)))
        matrix.columns = df.columns.values
        # get the test values
        p = np.array(get_prediction_values(matrix, test))
        real = test['value'].to_numpy()
        temp = np.array(p.round() - real) != 0
        acc[0] += temp.sum() / temp.shape[0]
        mse[0] += metrics.mean_squared_error(p, real)
        # acc = temp.sum() / temp.shape[0]
        # mse  and accuracy for test
        # print('m-1')
        # print(acc, metrics.mean_squared_error(p, real))
        # ----------------------------------
        matrix = pd.DataFrame(
            model2.inverse_transform(model2.transform(train_full)))
        matrix.columns = df.columns.values
        # get the test values
        p = np.array(get_prediction_values(matrix, test))
        real = test['value'].to_numpy()
        temp = np.array(p.round() - real) != 0
        acc[1] += temp.sum() / temp.shape[0]
        mse[1] += metrics.mean_squared_error(p, real)
        # acc = temp.sum() / temp.shape[0]
        # mse  and accuracy for test
        # print('m-2')
        # print(acc, metrics.mean_squared_error(p, real))
        # ----------------------------------
        matrix = pd.DataFrame(
            model3.inverse_transform(model3.transform(train_full)))
        matrix.columns = df.columns.values
        # get the test values
        p = np.array(get_prediction_values(matrix, test))
        real = test['value'].to_numpy()
        temp = np.array(p.round() - real) != 0
        # acc = temp.sum() / temp.shape[0]
        acc[2] += temp.sum() / temp.shape[0]
        mse[2] += metrics.mean_squared_error(p, real)
        # mse  and accuracy for test
        # print('m-3')
        # print(acc, metrics.mean_squared_error(p, real))
        # ----------------------------------
        matrix = pd.DataFrame(
            model4.inverse_transform(model4.transform(train_full)))
        matrix.columns = df.columns.values
        # get the test values
        p = np.array(get_prediction_values(matrix, test))
        real = test['value'].to_numpy()
        temp = np.array(p.round() - real) != 0
        acc[3] += temp.sum() / temp.shape[0]
        mse[3] += metrics.mean_squared_error(p, real)
        # acc = temp.sum() / temp.shape[0]
        # mse  and accuracy for test
        # print('m-4')
        # print(acc, metrics.mean_squared_error(p, real))
        # ----------------------------------
        matrix = pd.DataFrame(
            model5.inverse_transform(model5.transform(train_full)))
        matrix.columns = df.columns.values
        # get the test values
        p = np.array(get_prediction_values(matrix, test))
        real = test['value'].to_numpy()
        temp = np.array(p.round() - real) != 0
        # acc = temp.sum() / temp.shape[0]
        acc[4] += temp.sum() / temp.shape[0]
        mse[4] += metrics.mean_squared_error(p, real)
        # mse  and accuracy for test
        # print('m-5')
        # print(acc, metrics.mean_squared_error(p, real))
        # ----------------------------------
        matrix = pd.DataFrame(
            model6.inverse_transform(model6.transform(train_full)))
        matrix.columns = df.columns.values
        # get the test values
        p = np.array(get_prediction_values(matrix, test))
        real = test['value'].to_numpy()
        temp = np.array(p.round() - real) != 0
        acc[5] += temp.sum() / temp.shape[0]
        mse[5] += metrics.mean_squared_error(p, real)
        # mse  and accuracy for test
        # print('m-6')
        # print(acc, metrics.mean_squared_error(p, real))
        print(i)

    print(np.array(acc) / 50)
    print(np.array(mse) / 50)



















    # ks = [i for i in range(1,80)]
    # perfs_train = [0.07188754875892693, 0.27048388439048754, 0.3614107502735212, 0.42303897209038716, 0.4817242028691561, 0.5347816874453726, 0.5793995385017102, 0.625956865158202, 0.6530613200656314, 0.6750433778248361, 0.7037826299260685, 0.7295990100161482, 0.7419919502748542, 0.7653854182110436, 0.7851564962940284, 0.7965183733420504, 0.814400641638955, 0.8224703348882978, 0.8372156344101935, 0.8469571684023502, 0.8560432495794112, 0.8680648872909398, 0.8745868869265777, 0.8850851342904104, 0.8912720674402493, 0.8987918964844285, 0.9052072645481978, 0.9108192074123395, 0.9165227254330756, 0.9229308621728457, 0.9269404774832808, 0.9320411621515378, 0.9363027420985033, 0.940726430649303, 0.9448358234717464, 0.9486670221208268, 0.9512705416439509, 0.9552144467536448, 0.957914428733915, 0.9609092359554307, 0.9640562354593853, 0.9662448671952006, 0.9684568247338525, 0.9711399124302524, 0.9733787001228931, 0.9753333097590693, 0.9773091935897247, 0.978828744492407, 0.9804819173228084, 0.9818939121799526, 0.983295663191376, 0.9846625970064279, 0.9857781489452739, 0.9868982431538653, 0.9883877782328714, 0.9892890661938003, 0.9900137386713517, 0.9909007547761062, 0.9917693421016563, 0.9924576565368515, 0.9931858411790009, 0.9937341822754796, 0.9944351144671106, 0.9949718141497685, 0.9954531728068066, 0.9958675930748008, 0.9963859022089133, 0.9967179116991964, 0.9971148542984636, 0.9975160610977426, 0.9977533313132187, 0.9980625037738955, 0.9982744961684558, 0.9985139362040083, 0.9986303932276401, 0.9988489521706866, 0.999017641968166, 0.9992066488056833, 0.9993301335659979]
    # perfs_test = [0.8102119184446603, 0.7714523816819396, 0.8273095903528995, 0.8261420626694875, 0.8214640304265443, 0.8426647392470688, 0.8296675257319748, 0.8606850103878034, 0.8836864147679787, 0.8104405621487757, 0.837660743147642, 0.8280496965202289, 0.8215599385241534, 0.8547425995613717, 0.8613203383910377, 0.8435063776188407, 0.876836466544377, 0.8775554073879988, 0.8577164111708322, 0.8595440300713072, 0.869185054769279, 0.8759628375777531, 0.8686421794741918, 0.8588111386477971, 0.8556383831833424, 0.8861033325727614, 0.8554803880721662, 0.8858676500541961, 0.8747819318051601, 0.8584745109062281, 0.867958381051874, 0.8666537359001397, 0.8363860019476252, 0.8719505258795005, 0.8608943460676068, 0.8448932325975089, 0.8403141410434442, 0.8690928555607531, 0.8782678183859195, 0.8430765340543838, 0.8574087031196388, 0.8398193718925437, 0.8405560340465706, 0.8283237556495378, 0.8488538508101435, 0.8365706192199166, 0.8473185856296788, 0.8964290680472194, 0.864761644082645, 0.8196111244550629, 0.8191697147765804, 0.8353272138512923, 0.8848515049400911, 0.8404728134243278, 0.8576379514172761, 0.8840234624183818, 0.8666165087763424, 0.8733207133504036, 0.8490914237619054, 0.888259794942724, 0.8249954132861959, 0.8683606294903757, 0.8940925609644385, 0.8566185787889378, 0.8803331385671797, 0.8727105601358076, 0.9029747949763568, 0.8274106201028874, 0.8694313962264564, 0.8819661063461455, 0.8532429105225564, 0.8793285124495108, 0.8683946370875077, 0.847112907030068, 0.8692983322658175, 0.8038420994963621, 0.8573780860786193, 0.853291958022632, 0.8489649417084625]
    # acc = [0.5408419075089664, 0.5209810072296513, 0.5318520048906012, 0.5359990098788033, 0.5368237795459753, 0.5399239600377561, 0.5473410908637448, 0.5398031381810148, 0.5482708867955436, 0.5400010342459821, 0.5521339860974447, 0.548843927464794, 0.5431830924810057, 0.5417263933884449, 0.5523247918186641, 0.5544293430181427, 0.5373654440284871, 0.5478487753055001, 0.547132303957567, 0.540131200035742, 0.551549483088245, 0.5362489084687356, 0.5451908914274751, 0.5428516862857621, 0.5359614030658504, 0.5423730526755716, 0.552060664759853, 0.5586130484349106, 0.5449173833152194, 0.5465766598841271, 0.5444548925249675, 0.5467857768749866, 0.5422076061723363, 0.5575331261685339, 0.5485008243214338, 0.5349593495727396, 0.5344674897745575, 0.5291013257116131, 0.548993969850111, 0.5335056540040832, 0.5500967784149162, 0.5308729756889015, 0.5439979856255106, 0.5417294132057858, 0.5496569136024563, 0.5389638328105166, 0.5374346213201452, 0.5446026371520621, 0.5438625392569195, 0.5364561388662055, 0.5467975983992036, 0.5367837281349644, 0.5427344600832279, 0.5465243753694857, 0.549864592622993, 0.5432291805460203, 0.5400556380975167, 0.5545930231051597, 0.537939235391808, 0.5414270843899466, 0.5299668293167432, 0.5470525909577142, 0.5510320664029067, 0.5534478231292782, 0.5551835139459694, 0.5452522158818599, 0.5524799193019752, 0.5343498595406568, 0.5359088291790013, 0.5418181805258798, 0.5371242291185416, 0.5447822440096084, 0.5376679810018299, 0.5473176682708345, 0.5540073636137037, 0.5266204295623467, 0.55344799217772, 0.5445745405511515, 0.5443385689047177]
    # plt.plot(ks, perfs_train)
    # plt.show()
