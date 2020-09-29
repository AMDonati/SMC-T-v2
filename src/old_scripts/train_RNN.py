import tensorflow as tf
import argparse
from data_provider.datasets import Dataset, CovidDataset
from algos.run_SMC_T import SMCTAlgo
from algos.run_rnn import RNNAlgo
from algos.run_baseline_T import BaselineTAlgo

algos = {"smc_t": SMCTAlgo, "lstm": RNNAlgo, "baseline_t": BaselineTAlgo}

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

    parser.add_argument("-rnn_units", type=int, default=8, help="number of rnn units")
    parser.add_argument("-bs", type=int, default=32, help="batch size")
    parser.add_argument("-ep", type=int, default=3, help="number of epochs")
    parser.add_argument("-lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("-p_drop", type=float, required=True, help="dropout on output layer")
    parser.add_argument("-rnn_drop", type=float, default=0.0, help="dropout on rnn layer")
    parser.add_argument("-cv", type=str2bool, default=False, help="running 5 cross-validation")
    parser.add_argument("-data_path", type=str, required=True, help="path for saving data")
    parser.add_argument("-output_path", type=str, required=True, help="path for output folder")
    parser.add_argument("-dataset", type=str, default='synthetic', help='dataset selection')
    parser.add_argument("-past_len", type=int, default=40, help="number of timesteps for past timesteps at inference")
    parser.add_argument("-inference", type=int, default=0, help="launch inference or not on test data.")
    parser.add_argument("-multistep", type=str2bool, default=False, help="doing multistep inference or not.")
    parser.add_argument("-mc_samples", type=int, default=100, help="number of samples for MC Dropout algo.")
    parser.add_argument("-save_path", type=str, help="path for saved model folder (if loading ckpt)")

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
    if args.ep > 0:
        algo._train()
    algo.test()
    if args.inference:
        algo.launch_inference(multistep=args.multistep)
    print('done')

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