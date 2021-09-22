from sklearn.model_selection import train_test_split
from utils.utils import get_user_data, get_not_user_data
from sklearn.preprocessing import StandardScaler


def split_x_y(data):
    '''
    [for regression]
    Returns x and y for a lagged dataset, where 'y' is the column named 'slevel' or 'sclass', that is, the sedentary level of the present
    '''
    data = data.values.astype('float64')
    x = data[:, :-1]
    y = data[:, -1]

    return x, y


def get_train_test_data(model_type,  nb_lags=1, period=1, gran='1h', user=-1, standarize=True):
    '''
    From a specific and already processed dataset, generated x_train, x_test, y_train, y_test.

    If standarize is true, X will be standarized
    '''
    assert (model_type == 'regression' or model_type ==
            'classification'), 'Not a valid model type.'
    data = get_lagged_dataset(model_type, nb_lags, period, gran, user)
    x, y = split_x_y(data, model_type)
    if user != -1:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, shuffle=False, train_size=2 / 3)
    else:
        x_test = get_user_data(x, user)
        x_train = get_not_user_data(x, user)
        y_test = get_user_data(y, user)
        y_train = get_not_user_data(y, user)
    x_train, y_train = x_train.values.astype(
        "float32"), y_train.values.astype("float32")
    x_test, y_test = x_test.values.astype(
        "float32"), y_test.values.astype("float32")

    return x_train, y_train, x_test, y_test



