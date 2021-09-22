import sys
import pickle
import matplotlib.pyplot as plt

def show_train_prediction(user, architecture):
    info = train_cache[user][architecture]
    model = models[user][architecture]['model']
    plt.close()
    plt.figure(figsize=(15, 4))
    plt.title('Datos de entrenamiento y predicciones para el usuario {0} con la arquitectura {1}'.format(user, architecture))
    y_pred = model.predict(info['x_train'])
    plt.plot(info['y_train'], label='Train')
    plt.plot(y_pred, label='Predicción')
    plt.axhline(y=1.5, color='r', linestyle=':', )
    plt.legend(loc='upper right')
    plt.savefig('Imagenes/{0}lags_user{1}_arch{2}__train.png'.format(time_lags,user, architecture))

def show_test_prediction(user, architecture):
    info = test_cache[user][architecture]
    model = models[user][architecture]['model']
    plt.close()
    plt.figure(figsize=(15, 4))
    #plt.title('Datos de testeo y predicciones para el usuario {0} con la arquitectura {1}'.format(user, architecture))
    y_pred = model.predict(info['x_test'])
    plt.plot(info['y_test'], label='Prueba')
    plt.plot(y_pred, label='Predicción')
    plt.ylabel('MET')
    plt.axhline(y=1.5, color='r', linestyle=':', )
    plt.legend(loc='upper right')
    plt.savefig('Imagenes/{0}lags_user{1}_arch{2}__test.png'.format(time_lags,user, architecture))

def show_history_loss(user, architecture):
    history = models[user][architecture]['history']
    plt.close()
    plt.title('Train loss vs. Test loss of user {0} with architecture {1}'.format(user, architecture))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig('Imagenes/{0}lags_user{1}_arch{2}_loss.png'.format(time_lags,user, architecture))

def generate_prediction_images():
    for user in users:
        for architecture in range(1, number_of_architectures + 1):
            print('*** Generando predicciones para el usuario {0} y la arquitectura {1}... ***'.format(user,
                                                                                                       architecture))
            show_train_prediction(user, architecture)
            show_test_prediction(user, architecture)
            show_history_loss(user, architecture)

time_lags = 2
metric = 'mse'
number_of_architectures = 4
users = [50, 31, 4]
test_cache = pickle.load(open('pkl/test_cache_{0}lags_metric_{1}.pkl'.format(time_lags,metric), 'rb'))
train_cache = pickle.load(open('pkl/train_cache_{0}lags_metric_{1}.pkl'.format(time_lags,metric), 'rb'))
models = pickle.load(open('pkl/models_{0}lags_metric_{1}.pkl'.format(time_lags,metric), 'rb'))

