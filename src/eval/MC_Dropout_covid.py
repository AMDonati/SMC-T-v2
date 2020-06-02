import numpy as np
import tensorflow as tf
from models.Baselines.RNNs import build_LSTM_for_regression
import os
from utils.utils_train import restoring_checkpoint
from preprocessing.time_series.df_to_dataset_synthetic import split_input_target


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
    inp = inp_model
    mc_preds = []
    for t in range(len_future+1):
        list_predictions = []
        for i in range(mc_samples):
            preds_test = lstm_model(inputs=inp)  # (B,S,1)
            last_pred = preds_test[:,-1,:]
            list_predictions.append(last_pred)
        all_preds = tf.stack(list_predictions, axis=1)  # shape (B, N, 1)
        mc_preds.append(all_preds)
        mean_pred = tf.reduce_mean(all_preds, axis=1, keepdims=True) # (B,1,1)
        inp = tf.concat([inp, mean_pred], axis=1)
    mc_preds = tf.stack(mc_preds, axis=2)[:,:,1:,:]
    print('mc dropout LSTM multistep done')
    return tf.squeeze(mc_preds)


if __name__ == '__main__':

    # ---------- Load Test Data -----------------------------------------------------------------------------------------------------------------

    data_path = '../../data/covid_test_data.npy'
    test_data = np.load(data_path)

    index = 33
    test_sample = test_data[index]
    print('test_sample', test_sample)
    test_sample = tf.convert_to_tensor(test_sample)
    test_sample = tf.reshape(test_sample, shape=(1, test_sample.shape[-2], test_sample.shape[-1]))

    inputs, targets = split_input_target(test_sample)

    # ---------- Load LSTM with learned params ------------------------------------------------------------------------------------
    # out_path = "../../output/covid_rnn/covid_LSTM_64_pdrop_0.1_rnndrop_0.0"
    # p_drop = 0.1
    # rnn_drop = 0.0

    out_path = '../../output/covid_rnn/covid_LSTM_64_pdrop_0.2_rnndrop_0.2'
    p_drop = 0.2
    rnn_drop = 0.2

    # out_path = '../../output/covid_rnn/covid_LSTM_64_pdrop_0.5_rnndrop_0.0'
    # p_drop = 0.5
    # rnn_drop = 0.5


    learning_rate = 0.001
    rnn_units = 64
    N_est = 1000

    checkpoint_path = os.path.join(out_path, "checkpoints")
    save_path = os.path.join(out_path, 'mc_dropout_samples_sample_{}.npy'.format(index))

    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)

    lstm = build_LSTM_for_regression(shape_input_1=60, shape_input_2=1,
                                     shape_output=1,
                                     rnn_units=rnn_units,
                                     dropout_rate=p_drop,
                                     rnn_drop_rate=rnn_drop)

    # restore 2 LSTM
    LSTM_ckpt_path = os.path.join(checkpoint_path, "RNN_baseline_1")
    LSTM_ckpt = tf.train.Checkpoint(model=lstm, optimizer=optimizer)
    LSTM_ckpt_manager = tf.train.CheckpointManager(LSTM_ckpt, LSTM_ckpt_path, max_to_keep=30)
    _ = restoring_checkpoint(ckpt_manager=LSTM_ckpt_manager,
                             ckpt=LSTM_ckpt,
                             args_load_ckpt=True,
                             logger=None)

    #mc_samples = MC_Dropout_LSTM(lstm_model=lstm, inp_model=inputs, mc_samples=1000)
    mc_samples_multi = MC_Dropout_LSTM_multistep(lstm_model=lstm, inp_model=inputs[:,:40,:], mc_samples=5)
    print(mc_samples_multi.shape)
