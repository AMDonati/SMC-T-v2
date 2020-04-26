import os
import csv
import shutil
import pickle as pkl
import logging
import numpy as np

def write_to_csv(output_dir, dic):
  """Write a python dic to csv."""
  with open(output_dir, 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in dic.items():
      writer.writerow([key, value])

def create_logger(out_file_log):
  logging.basicConfig(filename=out_file_log, level=logging.INFO)
  # create logger
  logger = logging.getLogger('training log')
  logger.setLevel(logging.INFO)
  # create console handler and set level to debug
  ch = logging.StreamHandler()
  ch.setLevel(logging.INFO)
  # create formatter
  formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                              "%Y-%m-%d %H:%M:%S")
  # add formatter to ch
  ch.setFormatter(formatter)
  # add ch to logger
  logger.addHandler(ch)
  return logger


def create_run_dir(path_dir, path_name):
  path = os.path.join(path_dir, path_name)
  if os.path.isdir(path):
    ckpt_path = os.path.join(path, "checkpoints")
    if os.path.exists(ckpt_path):
      print("output folder already existing with checkpoints saved. keeping it and restoring checkpoints if allowed.")
    else:
      print('Suppression of old directory with same parameters')
      os.chmod(path, 0o777)
      shutil.rmtree(path, ignore_errors=True)
      os.makedirs(path)
  else:
    os.makedirs(path)
  return path

def restoring_checkpoint(ckpt_manager, ckpt, args_load_ckpt=True, logger=None):
  if ckpt_manager.latest_checkpoint and args_load_ckpt:
    ckpt_restored_path = ckpt_manager.latest_checkpoint
    ckpt_name = os.path.basename(ckpt_restored_path)
    _, ckpt_num = ckpt_name.split('-')
    start_epoch = int(ckpt_num)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("checkpoint restored from {}".format(ckpt_manager.latest_checkpoint))
    if logger is not None:
      logger.info('Latest checkpoint restored!!')
    return start_epoch

def saving_training_history(keys, values, output_path, csv_fname, logger, start_epoch):
  history = dict(zip(keys, values))
  baseline_history_fn = output_path + '/' + csv_fname
  if os.path.isdir(baseline_history_fn):
    logger.info("saving the history from the restored ckpt # {} in a new csv file...".format(start_epoch))
    baseline_history_fn = output_path + '/' + 'baseline_history_from_ckpt{}.csv'.format(start_epoch)
  write_to_csv(baseline_history_fn, history)
  logger.info('saving loss and metrics information...')

def saving_inference_results(keys, values, output_path, csv_fname):
  results = dict(zip(keys, values))
  results_fn = output_path + '/' + csv_fname
  write_to_csv(results_fn, results)

def saving_model_outputs(output_path, predictions, attn_weights, pred_fname, attn_weights_fname, logger):
  model_output_path = os.path.join(output_path, "model_outputs")
  if not os.path.isdir(model_output_path):
    os.makedirs(model_output_path)
  predictions_fn = model_output_path + '/' + pred_fname
  attn_weights_fn = model_output_path + '/' + attn_weights_fname
  np.save(predictions_fn, predictions)  # DO IT FOR A TEST DATASET INSTEAD?
  np.save(attn_weights_fn, attn_weights)  # DO IT FOR A TEST DATASET INSTEAD?
  logger.info("saving model output in .npy files...")
  return model_output_path


def save_to_pickle(file_name, np_array):
  with open(file_name,'wb') as f:
    pkl.dump(np_array, f)

if __name__ == "__main__":
  path_dir='../../output'
  path_name='ckpt-1'
  temp_path=create_run_dir(path_dir=path_dir, path_name=path_name)
  ckpt_name=os.path.basename(temp_path)
  print('checkpt name', ckpt_name)
  _, ckpt_num=ckpt_name.split('-')
  print(int(ckpt_num))

  # file_temp=temp_path+'/temp.pkl'
  # array=np.zeros(shape=(10,10))
  # save_to_pickle(file_temp, array)
  #
  # csv_temp=path_dir+'/temp_table.csv'
  # l1=['key', 'value']
  # l2=[1,2]
  # write_to_csv(csv_temp, dict(zip(l1,l2)))