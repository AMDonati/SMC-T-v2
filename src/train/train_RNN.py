import tensorflow as tf
import os, argparse
import numpy as np
from preprocessing.time_series.df_to_dataset_synthetic import split_synthetic_dataset, data_to_dataset_3D, \
    split_input_target
from preprocessing.time_series.df_to_dataset_weather import df_to_data_regression
from preprocessing.time_series.df_to_dataset_covid import split_covid_data
from utils.utils_train import create_logger
from models.Baselines.RNNs import build_LSTM_for_regression
from train.train_functions import train_LSTM
from data_provider.datasets import Dataset, CovidDataset
from algos.run_SMC_T import algos

def MC_Dropout_LSTM(lstm_model, inp_model, mc_samples):
    '''
    :param LSTM_hparams: shape_input_1, shape_input_2, shape_ouput, num_units, dropout_rate
    :param inp_model: array of shape (B,S,F)
    :param mc_samples:
    :return:
    '''
    list_predictions = []
    for i in range(mc_samples):
        predictions_test = lstm_model(inputs=inp_model)  # (B,S,1)
        list_predictions.append(predictions_test)
    predictions_test_MC_Dropout = tf.stack(list_predictions, axis=1)  # shape (B, N, S, 1)
    print('done')
    return tf.squeeze(predictions_test_MC_Dropout)

def MC_Dropout_LSTM_multistep(lstm_model, inp_model, mc_samples, len_future=20):
    '''
        :param LSTM_hparams: shape_input_1, shape_input_2, shape_ouput, num_units, dropout_rate
        :param inp_model: array of shape (B,S,F)
        :param mc_samples:
        :return:
    '''
    list_predictions = []
    for i in range(mc_samples):
        inp = inp_model
        for t in range(len_future+1):
                preds_test = lstm_model(inputs=inp)  # (B,S,1)
                last_pred = tf.expand_dims(preds_test[:, -1, :], axis=-2)
                inp = tf.concat([inp, last_pred], axis=1)
        list_predictions.append(preds_test)
    preds_test_MC_Dropout = tf.stack(list_predictions, axis=1)
    print('mc dropout LSTM multistep done')
    return tf.squeeze(preds_test_MC_Dropout)


if __name__ == '__main__':

    # trick for boolean parser args.
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    parser = argparse.ArgumentParser()

    parser.add_argument("-rnn_units", type=int, required=True, help="number of rnn units")
    parser.add_argument("-bs", type=int, default=32, help="batch size")
    parser.add_argument("-ep", type=int, default=1, help="number of epochs")
    parser.add_argument("-lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("-p_drop", type=float, required=True, help="dropout on output layer")
    parser.add_argument("-rnn_drop", type=float, default=0.0, help="dropout on rnn layer")
    parser.add_argument("-cv", type=str2bool, default=False, help="running 5 cross-validation")
    parser.add_argument("-data_path", type=str, required=True, help="path for saving data")
    parser.add_argument("-output_path", type=str, required=True, help="path for output folder")
    parser.add_argument("-dataset", type=str, default='covid', help='dataset selection')

    args = parser.parse_args()

    # ------------------- Upload dataset ----------------------------------------------------------------------------------
    BATCH_SIZE = args.bs

    if args.dataset == 'synthetic':
        BUFFER_SIZE = 500
        dataset = Dataset(data_path=args.data_path, BUFFER_SIZE=BUFFER_SIZE, BATCH_SIZE=BATCH_SIZE)

    elif args.dataset == 'covid':
        BUFFER_SIZE = 50
        dataset = CovidDataset(data_path=args.data_path, BUFFER_SIZE=BUFFER_SIZE, BATCH_SIZE=BATCH_SIZE)

    algo = algos["lstm"](dataset=dataset, args=args)

    algo.train()
    algo.test()
 #---------------------------------------

    # data_path = os.path.join(args.data_path, 'covid_test_data.npy')
    # test_data = np.load(data_path)
    # test_data = tf.convert_to_tensor(test_data)
    # inputs, targets = split_input_target(test_data)
    # #mc_samples = MC_Dropout_LSTM(lstm_model=model, inp_model=inputs, mc_samples=1000)
    # mc_samples = MC_Dropout_LSTM_multistep(lstm_model=model, inp_model=inputs[:,:40,:], mc_samples=1000)
    # print(mc_samples.shape)
    # save_path = os.path.join(output_path, 'mc_dropout_samples_test_data_multi.npy')
    # np.save(save_path, mc_samples)