from abc import ABC, abstractmethod
from preprocessing.datasets import get_lagged_dataset
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, f1_score
from preprocessing.model_ready import split_x_y
from utils.utils import get_user_data, get_not_user_data
from utils.utils import get_granularity_from_minutes
import pickle as pkl
from sklearn.preprocessing import StandardScaler
import time
from sklearn.metrics import mean_squared_error, roc_auc_score
from utils.utils import file_exists
import tensorflow as tf
import numpy as np
from math import sqrt
from keras import backend as K
import numpy as np
from tcn import tcn_full_summary


class Experiment(ABC):
    def __init__(self, model_fn, model_name, task_type, user, nb_lags, period, nb_min, need_3d_input):
        self.lags = nb_lags
        self.task_type = task_type
        self.user = user
        self.nb_lags = nb_lags
        self.period = period
        self.nb_min = nb_min
        self.model_fn = model_fn
        self.need_3d_input = need_3d_input
        self.validation_splits = 5
        self.experiment_data = {}
        self.train_data = None
        self.test_data = None
        self.gran = get_granularity_from_minutes(self.nb_min)
        self.arch = model_name
        self.name = f'_{self.task_type}_gran{self.gran}_period{self.period}_lags{self.nb_lags}_model-{model_name}_user{self.user}'

        if self.task_type == 'classification':
            self.scoring_func = roc_auc_score
        else:
            # def rmse(x, y):
            # return sqrt(mean_squared_error(x, y))
            self.scoring_func = mean_squared_error  # rmse

        self.déjà_fait = False

    @abstractmethod
    def prepare_data(self):
        pass

    def time_series_split(self):
        min_train = self.train_data.index.get_level_values(1).min()
        min_test = self.test_data.index.get_level_values(1).min()
        max_train = self.train_data.index.get_level_values(1).max()
        max_test = self.test_data.index.get_level_values(1).max()
        min_date = max([min_train, min_test])
        max_date = min([max_train, max_test])
        diff = max_date-min_date
        n_folds = self.validation_splits + 1
        time_per_fold = diff / n_folds
        split_date = min_date
        for split_nb in range(self.validation_splits):
            split_date = split_date + time_per_fold
            train_index = (
                self.train_data.index.get_level_values(1) <= split_date)
            train_data_split = self.train_data[train_index]
            if split_nb != self.validation_splits-1:
                test_index_may = (
                    self.test_data.index.get_level_values(1) > split_date)
                test_index_inf = (self.test_data.index.get_level_values(
                    1) < (split_date+time_per_fold))
                test_index = test_index_may & test_index_inf
                test_data_split = self.test_data[test_index]
            else:
                test_index = (
                    self.test_data.index.get_level_values(1) > split_date)
                test_data_split = self.test_data[test_index]
            X_train, y_train = split_x_y(train_data_split)
            X_test, y_test = split_x_y(test_data_split)
            yield X_train, y_train, X_test, y_test

    def reserve_file(self):
        experiment_file = open(self.filename, 'wb')
        pkl.dump(['almost empty'], experiment_file)
        experiment_file.close()
        print(f'File reserved {self.filename}')

    def save(self):
        experiment_file = open(self.filename, 'wb')
        pkl.dump(self.experiment_data, experiment_file)
        experiment_file.close()

    def normalize(self, X_train, X_test):
        ss = StandardScaler()
        X_train = ss.fit_transform(X_train)
        X_test = ss.transform(X_test)
        del ss
        return X_train, X_test

    # def set_model_imput(self):

    def load(self):
        if self.déjà_fait:
            self.experiment_data = pkl.load(open(self.filename, 'rb'))

    def run(self, nb_epochs=64, batch_size=64, with_class_weights=False, experiment_verbose=0, fit_verbose=0, model_verbose=0):
        print('*** ' * 10)
        print('Experiment is: ')
        print(self.name)
        self.filename = f'pkl/experiments/{self.name}.pkl'

        if not file_exists(self.filename):
            print(f'Beginning experiment: ')
            # avoids running an experiment if another precess is running it
            self.reserve_file()
            self.prepare_data()
            self.experiment_data['scores'] = []
            self.experiment_data['time_to_train'] = []
            self.experiment_data['y_test_pred'] = []
            i = 1
            for split_data in self.time_series_split():

                print(f'Starting iteration nb {i} of {self.validation_splits}')
                K.clear_session()  # clears tf state like model graph parameters
                i += 1
                X_train, y_train, X_test, y_test = split_data
                if X_train.shape[0] == 0 or X_test.shape[0] == 0:
                    self.experiment_data['scores'].append(np.NaN)
                    self.experiment_data['y_test_pred'].append(np.NaN)
                    self.experiment_data['time_to_train'].append(0)
                else:
                    X_train, X_test = self.normalize(X_train, X_test)

                    if self.need_3d_input:
                        nb_features = int(X_train.shape[1]/self.nb_lags)
                        nb_lags = self.nb_lags
                        nb_train_samples = X_train.shape[0]
                        nb_test_samples = X_test.shape[0]
                        X_train = X_train.reshape(
                            nb_train_samples, nb_lags, nb_features)
                        X_test = X_test.reshape(
                            nb_test_samples, nb_lags, nb_features)

                    start = time.time()
                    model = self.model_fn()
                    tf.keras.backend.clear_session()

                    es_callback = tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        min_delta=0,
                        patience=32,
                        verbose=1,
                        mode="min",
                        baseline=None,
                        restore_best_weights=True,
                    )

                    validation_data = None
                    validation_data = (X_test, y_test)
                    model.fit(X_train,
                              y_train,
                              batch_size=batch_size,
                              epochs=nb_epochs,
                              verbose=fit_verbose,
                              validation_data=validation_data,
                              callbacks=[es_callback])
                    end = time.time()
                    total = round((end - start) / 60, 3)
                    y_pred = model.predict(X_test)
                    score = self.scoring_func(y_test, y_pred)

                    self.experiment_data['scores'].append(round(score, 3))
                    self.experiment_data['nb_params'] = model.count_params()
                    self.experiment_data['y_test_pred'].append(
                        (y_test, y_pred))
                    self.experiment_data['time_to_train'].append(total)
                    if experiment_verbose > 1:
                        print('Shapes for this iteration are: ')
                        print(f'X_train: {X_train.shape}')
                        print(f'X_test: {X_test.shape}')
                        print(f'Score: {round(score, 3)}')
                        print(f'Time: {total}')
                        # print(f'Class weights are: {class_weight}')
                    if model_verbose==1:
                        if  self.arch=='tcn':
                            tcn_full_summary(model, expand_residual_blocks=False)
                        else: 
                            model.summary()
                    del model
            if experiment_verbose > 0:
                print(f"scores: {self.experiment_data['scores']}")
                print(f"nb_params: {self.experiment_data['nb_params']}")
                print(
                    f"time per iteration: {self.experiment_data['time_to_train']}")
                print(
                    f"total time: {sum(self.experiment_data['time_to_train'])}")

            print(f'Experiment finished')
        else:
            print(self.name)
            print('Experiment already done...')
            self.déjà_fait = True

    def get_experiment_data(self):
        return self.experiment_data

    def get_results(self):
        return self.experiment_data['scores']

    def get_mean_score(self):
        return np.mean(self.get_results())

    def get_total_time(self):
        return sum(self.experiment_data['time_to_train'])


class PersonalExperiment(Experiment):

    def __init__(self, model_fn, model_name, task_type, user, nb_lags, period, nb_min, need_3d_input):
        super().__init__(model_fn, model_name, task_type,
                         user, nb_lags, period, nb_min, need_3d_input)
        self.name += '_per'

    def prepare_data(self):
        self.dataset = get_lagged_dataset(task_type=self.task_type,
                                          user=self.user,
                                          nb_lags=self.nb_lags,
                                          period=self.period,
                                          nb_min=self.nb_min)
        self.train_data = get_user_data(self.dataset, self.user)
        self.test_data = get_user_data(self.dataset, self.user)


class ImpersonalExperiment(Experiment):
    def __init__(self, model_fn, model_name, task_type, user, nb_lags, period, nb_min, need_3d_input):
        super().__init__(model_fn, model_name, task_type,
                         user, nb_lags, period, nb_min, need_3d_input)
        self.name += '_imp'

    def prepare_data(self):
        self.dataset = get_lagged_dataset(task_type=self.task_type,
                                          user=-1,
                                          nb_lags=self.nb_lags,
                                          period=self.period,
                                          nb_min=self.nb_min)
        self.train_data = get_not_user_data(self.dataset, self.user)
        self.test_data = get_user_data(self.dataset, self.user)


class HybridExperiment(Experiment):
    def __init__(self, model_fn, model_name, task_type, user, nb_lags, period, nb_min, need_3d_input):
        super().__init__(model_fn, model_name, task_type,
                         user, nb_lags, period, nb_min, need_3d_input)
        self.name += '_hyb'

    def prepare_data(self):
        self.dataset = get_lagged_dataset(task_type=self.task_type,
                                          user=self.user,
                                          nb_lags=self.nb_lags,
                                          period=self.period,
                                          nb_min=self.nb_min)
        self.train_data = self.dataset
        self.test_data = get_user_data(self.dataset, self.user)
