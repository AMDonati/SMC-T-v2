import tensorflow as tf
import os, argparse
import numpy as np
from preprocessing.time_series.df_to_dataset_synthetic import split_synthetic_dataset, data_to_dataset_3D, \
    split_input_target
from preprocessing.time_series.df_to_dataset_weather import df_to_data_regression
from utils.utils_train import create_logger
from models.Baselines.RNNs import build_LSTM_for_regression
from train.train_functions import train_LSTM

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
    parser.add_argument("-bs", type=int, default=256, help="batch size")
    parser.add_argument("-ep", type=int, default=30, help="number of epochs")
    parser.add_argument("-lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("-p_drop", type=float, required=True, help="learning rate")
    parser.add_argument("-cv", type=str2bool, default=False, help="running 5 cross-validation")
    parser.add_argument("-data_path", type=str, required=True, help="path for saving data")
    parser.add_argument("-output_path", type=str, required=True, help="path for output folder")
    parser.add_argument("-dataset", type=str, default='weather', help='dataset selection')

    args = parser.parse_args()

    # ------------------- Upload synthetic dataset ----------------------------------------------------------------------------------

    BATCH_SIZE = args.bs
    TRAIN_SPLIT = 0.7
    if args.dataset == 'synthetic':
        BUFFER_SIZE = 500
        data_path = os.path.join(args.data_path, 'synthetic_dataset_1_feat.npy')
        input_data = np.load(data_path)
        train_data, val_data, test_data = split_synthetic_dataset(x_data=input_data, TRAIN_SPLIT=TRAIN_SPLIT,
                                                                  cv=args.cv)

        val_data_path = os.path.join(args.data_path, 'val_data_synthetic_1_feat.npy')
        train_data_path = os.path.join(args.data_path, 'train_data_synthetic_1_feat.npy')
        test_data_path = os.path.join(args.data_path, 'test_data_synthetic_1_feat.npy')

        np.save(val_data_path, val_data)
        np.save(train_data_path, train_data)
        np.save(test_data_path, test_data)

    elif args.dataset == 'weather':

        BUFFER_SIZE = 5000
        file_path = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip'
        fname = 'jena_climate_2009_2016.csv.zip'
        col_name = ['p (mbar)', 'T (degC)', 'rh (%)', 'rho (g/m**3)']
        index_name = 'Date Time'
        # temperature recorded every 10 minutes.
        history = 6 * 24 * 4 + 6 * 4  # history of 4 days + one more 4 hours interval for splitting target / input.
        step = 6 * 4  # sample a temperature every 4 hours.

        (train_data, val_data, test_data), original_df, stats = df_to_data_regression(file_path=file_path,
                                                                                      fname=fname,
                                                                                      col_name=col_name,
                                                                                      index_name=index_name,
                                                                                      TRAIN_SPLIT=TRAIN_SPLIT,
                                                                                      history=history,
                                                                                      step=step,
                                                                                      cv=args.cv,
                                                                                      max_samples=20000)

    if not args.cv:
        train_dataset, val_dataset, test_dataset = data_to_dataset_3D(train_data=train_data,
                                                                      val_data=val_data,
                                                                      test_data=test_data,
                                                                      split_fn=split_input_target,
                                                                      BUFFER_SIZE=BUFFER_SIZE,
                                                                      BATCH_SIZE=BATCH_SIZE,
                                                                      target_feature=None,
                                                                      cv=args.cv)
        for (inp, tar) in train_dataset.take(1):
            print('input example', inp[0])
            print('target example', tar[0])

    else:
        list_train_dataset, list_val_dataset, test_dataset = data_to_dataset_3D(train_data=train_data,
                                                                                val_data=val_data,
                                                                                test_data=test_data,
                                                                                split_fn=split_input_target,
                                                                                BUFFER_SIZE=BUFFER_SIZE,
                                                                                BATCH_SIZE=BATCH_SIZE,
                                                                                target_feature=None,
                                                                                cv=args.cv)
        for (inp, tar) in list_train_dataset[0].take(1):
            print('input example', inp[0])
            print('target example', tar[0])


    # -------------------- define hyperparameters -----------------------------------------------------------------------------------
    rnn_units = args.rnn_units
    learning_rate = args.lr
    EPOCHS = args.ep
    # define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)
    output_path = args.output_path
    out_file = 'LSTM_units_{}_pdrop_{}_lr_{}_bs_{}_cv_{}'.format(rnn_units, args.p_drop, learning_rate, BATCH_SIZE, args.cv)
    output_path = os.path.join(output_path, out_file)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    if not args.cv:
        for inp, tar in train_dataset.take(1):
            seq_len = tf.shape(inp)[1].numpy()
            num_features = tf.shape(inp)[-1].numpy()
            output_size = tf.shape(tar)[-1].numpy()
    else:
        for inp, tar in list_train_dataset[0].take(1):
            seq_len = tf.shape(inp)[1].numpy()
            num_features = tf.shape(inp)[-1].numpy()
            output_size = tf.shape(tar)[-1].numpy()

    # -------------------- create logger and checkpoint saver ----------------------------------------------------------------------------------------------------

    out_file_log = output_path + '/' + 'training_log.log'
    logger = create_logger(out_file_log=out_file_log)
    #  creating the checkpoint manager:
    checkpoint_path = os.path.join(output_path, "checkpoints")
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    # -------------------- Build the RNN model -----------------------------------------------------------------------------------------
    model = build_LSTM_for_regression(shape_input_1=seq_len,
                                      shape_input_2=num_features,
                                      shape_output=output_size,
                                      rnn_units=rnn_units,
                                      dropout_rate=args.p_drop,
                                      training=True)
    if not args.cv:
        train_LSTM(model=model,
               optimizer=optimizer,
               EPOCHS=EPOCHS,
               train_dataset=train_dataset,
               val_dataset=val_dataset,
               checkpoint_path=checkpoint_path,
               output_path=output_path,
               logger=logger,
               num_train=1)
    else:
        for t, (train_dataset, val_dataset) in enumerate(zip(list_train_dataset, list_val_dataset)):
            logger.info("starting training of train/val split number {}".format(t+1))
            train_LSTM(model=model,
                       optimizer=optimizer,
                       EPOCHS=EPOCHS,
                       train_dataset=train_dataset,
                       val_dataset=val_dataset,
                       checkpoint_path=checkpoint_path,
                       output_path=output_path,
                       logger=logger,
                       num_train=t+1)
            logger.info("training of a LSTM for train/val split number {} done...".format(t + 1))
            logger.info(
                "<---------------------------------------------------------------------------------------------------------------------------------------------------------->")