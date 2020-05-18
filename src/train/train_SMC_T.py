import tensorflow as tf
import numpy as np
import os, argparse
from preprocessing.time_series.df_to_dataset_synthetic import split_input_target, data_to_dataset_4D, \
    split_synthetic_dataset
from preprocessing.time_series.df_to_dataset_weather import df_to_data_regression
from models.SMC_Transformer.SMC_Transformer import SMC_Transformer
from train.train_functions import train_SMC_transformer
from utils.utils_train import create_logger
from train.loss_functions import CustomSchedule

if __name__ == '__main__':

    #  trick for boolean parser args.
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

    parser.add_argument("-d_model", type=int, required=True, help="depth of attention parameters")
    parser.add_argument("-bs", type=int, default=10000, help="batch size")
    parser.add_argument("-ep", type=int, default=3, help="number of epochs")
    parser.add_argument("-full_model", type=str2bool, default=False,
                        help="simple transformer or one with ffn and layer norm")
    parser.add_argument("-dff", type=int, default=0, help="dimension of feed-forward network")
    parser.add_argument("-particles", type=int, default=1, help="number of particles")
    parser.add_argument("-sigmas", type=float, default=0.1, help="values for sigma_k, sigma_q, sigma_v, sigma_z")
    parser.add_argument("-sigma_obs", type=float, default=0.1, help="values for sigma obs")
    parser.add_argument("-smc", type=str2bool, required=True, help="Recurrent Transformer with or without smc algo")
    parser.add_argument("-dataset", type=str, default='weather', help='dataset selection')
    parser.add_argument("-data_path", type=str, required=True, help="path for saving data")
    parser.add_argument("-output_path", type=str, required=True, help="path for output folder")

    args = parser.parse_args()

    if not args.smc:
        assert args.particles == (1 or None)

    # -------------------------------- Upload synthetic dataset ----------------------------------------------------------------------------------

    BATCH_SIZE = args.bs
    TRAIN_SPLIT = 0.7

    if args.dataset == 'synthetic':
        BUFFER_SIZE = 500
        data_path = os.path.join(args.data_path, 'synthetic_dataset_1_feat.npy')
        input_data = np.load(data_path)

        train_data, val_data, test_data = split_synthetic_dataset(x_data=input_data,
                                                                  TRAIN_SPLIT=TRAIN_SPLIT,
                                                                  cv=False)

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
        history = 6 * 24 * 4 + 6 * 4 # history of 4 days + one more 4 hours interval for splitting target / input.
        step = 6 * 4  # sample a temperature every 4 hours.
        cv = False

        (train_data, val_data, test_data), original_df, stats = df_to_data_regression(file_path=file_path,
                                                                                      fname=fname,
                                                                                      col_name=col_name,
                                                                                      index_name=index_name,
                                                                                      TRAIN_SPLIT=TRAIN_SPLIT,
                                                                                      history=history,
                                                                                      step=step,
                                                                                      cv=cv,
                                                                                      max_samples=20000)

    train_dataset, val_dataset, test_dataset = data_to_dataset_4D(train_data=train_data,
                                                                  val_data=val_data,
                                                                  test_data=test_data,
                                                                  split_fn=split_input_target,
                                                                  BUFFER_SIZE=BUFFER_SIZE,
                                                                  BATCH_SIZE=BATCH_SIZE,
                                                                  target_feature=None,
                                                                  cv=False)

    for (inp, tar) in train_dataset.take(1):
        print('input example', inp[0])
        print('target example', tar[0])

    # ------------------------------ Define hyperparameters ------------------------------------------------------------------------------------------

    d_model = args.d_model
    EPOCHS = args.ep
    for (_, tar) in train_dataset.take(1):
        output_size = tf.shape(tar)[-1].numpy()
    seq_len = train_data.shape[1] - 1  # 24.

    # define optimizer
    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)
    output_path = args.output_path
    out_file = 'Recurrent_T_depth_{}_bs_{}_fullmodel_{}'.format(d_model, BATCH_SIZE, args.full_model)
    if args.particles is not None:
        out_file = out_file + '__p_{}'.format(args.particles)
    if args.sigma_obs is not None:
        out_file = out_file + '_SigmaObs_{}'.format(args.sigma_obs)
    if args.sigmas is not None:
        out_file = out_file + '_sigmas_{}'.format(args.sigmas)

    output_path = os.path.join(output_path, out_file)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # -------------------- create logger and checkpoint saver ----------------------------------------------------------------------------------------------------

    out_file_log = os.path.join(output_path, 'training_log.log')
    logger = create_logger(out_file_log=out_file_log)
    #  creating the checkpoint manager:
    checkpoint_path = os.path.join(output_path, "checkpoints")
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    # ------ Training of the recurrent Transformer ---------------------------------------------------------------------------------------------------
    logger.info('hparams...')
    logger.info(
        'd_model: {} - batch size {} - full model? {} - dff: {}'.format(d_model, BATCH_SIZE, args.full_model, args.dff))
    logger.info('num samples in training dataset:{}'.format(train_data.shape[0]))

    smc_transformer = SMC_Transformer(d_model=d_model, output_size=output_size, seq_len=seq_len,
                                      full_model=args.full_model, dff=args.dff)

    if args.smc:
        logger.info("SMC Transformer for {} particles".format(args.particles))
        if args.sigmas is not None:
            dict_sigmas = dict(zip(['k', 'q', 'v', 'z'], [args.sigmas for _ in range(4)])
        else:
            dict_sigmas = None
        smc_transformer.cell.add_SMC_parameters(dict_sigmas=dict_sigmas, sigma_obs=args.sigma_obs,
                                                num_particles=args.particles)
        assert smc_transformer.cell.noise == smc_transformer.cell.attention_smc.noise == True
        logger.info("Sigma_obs init: {}".format(smc_transformer.cell.Sigma_obs))

    train_SMC_transformer(smc_transformer=smc_transformer,
                          optimizer=optimizer,
                          EPOCHS=EPOCHS,
                          train_dataset=train_dataset,
                          val_dataset=val_dataset,
                          checkpoint_path=checkpoint_path,
                          logger=logger,
                          num_train=1)

    logger.info("computing test mse metric at the end of training...")
    # computing loss on test_dataset:
    for (inp, tar) in test_dataset:
        (preds_test, preds_test_resampl), _, _ = smc_transformer(inputs=inp,
                                                                 targets=tar)  # predictions test are the ones not resampled.
        test_metric_avg_pred = tf.keras.losses.MSE(tar, tf.reduce_mean(preds_test, axis=1, keepdims=True))  # (B,1,S)
        test_metric_avg_pred = tf.reduce_mean(test_metric_avg_pred)

    logger.info("test mse metric from avg particle: {}".format(test_metric_avg_pred))
