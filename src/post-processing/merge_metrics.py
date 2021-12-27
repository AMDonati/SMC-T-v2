import argparse
import os
import pandas as pd
import re
import math
import json

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", type=str, required=True,
                        help="data folder containing experiments")
    parser.add_argument("-to_remove", type=str, default=["train_ce", "train_ppl", "var_bleu"])
    #parser.add_argument('-bottom_folder', type=int, default=1)
    #parser.add_argument('-top_folder', type=int, default=1)
    parser.add_argument('-precision', type=int, default=2)
    return parser

def str_to_float_in_csv(df, index="train_loss", sep=","):
    values = df.loc[index].values[-1]
    values = values.split(sep)
    values = [e.replace('[', '') for e in values]
    values = [e.replace(']', '') for e in values]
    values = [e.replace('\n', '') for e in values]
    if len(values) == 1 and '' in values:
        return None
    else:
        values = [float(e) for e in values]
        return values[-1]

def check_train_mse_metric(df):
    if not "train_mse_metric" in df.index:
        return None
    else:
        values = df.loc["train_mse_metric"].values[-1]
        values = values.split(',')
        values = [e.replace('[', '') for e in values]
        values = [e.replace(']', '') for e in values]
        values = [e.replace('\n', '') for e in values]
        if len(values) == 1 and '' in values:
            return None
        else:
            values = [float(e) for e in values]
            return values[-1]

def open_config(config_path):
    with open(config_path, "r") as file:
        config = json.load(file)
    return config

def merge_one_experiment(path="output/temp", precision=4, to_remove="var_bleu"):
    dirs = [f.path for f in os.scandir(path) if f.is_dir()]
    merge_metrics = pd.DataFrame()
    for dir_conf in dirs:
        dirs = [f.path for f in os.scandir(dir_conf) if f.is_dir()]
        stats_all_runs_df = pd.DataFrame()
        all_metrics_all_runs_df = pd.DataFrame()
        for dir_experiment in dirs: # level for multiple runs with same config.
            config = open_config(os.path.join(dir_experiment, "config.json"))
            if config["algo"] == "lstm":
                losses_path = os.path.join(dir_experiment, "rnn_history_1.csv")
            elif config["algo"] == "smc_t":
                losses_path = os.path.join(dir_experiment, "smc_t_history_1.csv")
            if os.path.exists(losses_path):
                losses_exp = pd.read_csv(losses_path, index_col=0, header=None)
                losses_values = []
                value_train_mse = check_train_mse_metric(losses_exp)
                indexes = ["train_loss", "val_loss"] if value_train_mse is None else ["train_mse_metric", "val_mse_metric"]
                for index in indexes:
                    losses_values.append(str_to_float_in_csv(losses_exp, index, sep=','))
                all_metrics_df = pd.Series(losses_values, index=["train_ce", "val_ce"])
                all_metrics_df.loc["train_ppl"] = math.exp(all_metrics_df["train_ce"])
                all_metrics_df.loc["val_ppl"] = math.exp(all_metrics_df["val_ce"])
            test_metrics_path = os.path.join(dir_experiment, "test_metrics_mean_sampling.csv")
            if os.path.exists(test_metrics_path):
                test_metrics_exp = pd.read_csv(test_metrics_path, index_col=0, header=None)
                all_metrics_df = all_metrics_df.append(test_metrics_exp.squeeze())
            dir_name = os.path.split(dir_experiment)[-1]
            all_metrics_all_runs_df[dir_name] = all_metrics_df
        stats_all_runs_df["mean"] = all_metrics_all_runs_df.mean(axis=1)
        stats_all_runs_df["std"] = all_metrics_all_runs_df.std(axis=1)
        stats_all_runs_df = stats_all_runs_df.dropna(axis=1)
        if len(stats_all_runs_df.columns) > 1:
            merge_metrics[os.path.basename(dir_conf)] = stats_all_runs_df["mean"].apply(lambda t: str(round(t, precision)) + '+/-') + stats_all_runs_df["std"].apply(lambda t: str(round(t, 3)))
        else:
            merge_metrics[os.path.basename(dir_conf)] = stats_all_runs_df["mean"].apply(lambda t: str(round(t, precision)))
        stats_all_runs_df.to_csv(os.path.join(dir_conf, "all_metrics.csv"))
        if len(dirs) > 1:
            all_metrics_all_runs_df.to_csv(os.path.join(dir_conf, "all_metrics_per_run.csv"))
    for tr in to_remove:
        if tr in merge_metrics.index:
            merge_metrics = merge_metrics.drop(tr, axis=0)
    merge_metrics_latex = merge_metrics.apply(lambda t: t.replace('+/-', '\pm'))
    merge_metrics_latex.columns = [col.replace('_', '-') for col in merge_metrics_latex.columns]
    merge_metrics_latex.index = [ind.replace('_', '-') for ind in merge_metrics_latex.index]
    merge_metrics_latex = merge_metrics_latex.T
    merge_metrics.to_csv(os.path.join(path, "merge_metrics.csv"))
    merge_metrics_latex.to_latex(os.path.join(path, "merge_metrics.txt"))

# def merge_one_experiment(args):
#     dirs = [f.path for f in os.scandir(args.path) if f.is_dir()]
#
#     for dir_conf in dirs:
#         dirs = [f.path for f in os.scandir(dir_conf) if f.is_dir()]
#         df_with_trunc = pd.DataFrame()
#         df_no_trunc = pd.DataFrame()
#         for dir_experiment in dirs:
#             all_metrics_path = os.path.join(dir_experiment, "all_metrics.csv")
#             if os.path.exists(all_metrics_path):
#                 df_exp = pd.read_csv(all_metrics_path, index_col=0)
#                 df_exp = add_to_metrics(df_exp, dir_experiment)
#                 if "with_trunc" in df_exp.columns:
#                     df_with_trunc = df_with_trunc.append(df_exp["with_trunc"].to_dict(), ignore_index=True)
#                 if "no_trunc" in df_exp.columns:
#                     df_no_trunc = df_no_trunc.append(df_exp["no_trunc"].to_dict(), ignore_index=True)
#
#         str_mean_std = lambda x: str(round(x.mean(), args.precision)) + "+-" + str(round(x.std(), 2))
#         keys = []
#         concat_truncs = []
#         if not df_with_trunc.empty:
#             merged_with_trunc = df_with_trunc.apply(str_mean_std)
#             concat_truncs.append(merged_with_trunc)
#             keys.append("with_trunc")
#         if not df_no_trunc.empty:
#             merged_no_trunc = df_no_trunc.apply(str_mean_std)
#             keys.append("no_trunc")
#             concat_truncs.append(merged_no_trunc)
#         if concat_truncs:
#             all = pd.concat(concat_truncs, axis=1, keys=keys)
#             all = all.transpose()
#             all.to_csv(os.path.join(dir_conf, "merged_metrics.csv"))


# def merge_all_experiments(args):
#     dirs = [f.path for f in os.scandir(args.path) if f.is_dir()]
#     df_with_trunc = pd.DataFrame()
#     df_no_trunc = pd.DataFrame()
#     for dir_conf in dirs:
#         name_experiment = os.path.basename(dir_conf)
#         filename = os.path.join(dir_conf, "merged_metrics.csv")
#         if os.path.exists(filename):
#             df = pd.read_csv(filename, index_col=0)
#             if "with_trunc" in df.index:
#                 df_with_trunc = df_with_trunc.append(pd.Series(df.loc["with_trunc"], name=name_experiment))
#             if "no_trunc" in df.index:
#                 df_no_trunc = df_no_trunc.append(pd.Series(df.loc["no_trunc"], name=name_experiment))
#
#     columns_to_save = [col for col in args.columns_to_save if col in df_with_trunc.columns]
#     if not df_with_trunc.empty:
#         df_with_trunc = df_with_trunc[columns_to_save]
#     if not df_no_trunc.empty:
#         df_no_trunc = df_no_trunc[columns_to_save]
#
#     df_with_trunc.to_csv(os.path.join(args.path, "merged_with_trunc.csv"))
#     df_no_trunc.to_csv(os.path.join(args.path, "merged_no_trunc.csv"))
#     df_with_trunc.to_latex(os.path.join(args.path, "merged_with_trunc.txt"))
#     df_no_trunc.to_latex(os.path.join(args.path, "merged_no_trunc.txt"))
#     print(f"Saved in {args.path}")


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    merge_one_experiment(path=args.path, precision=args.precision, to_remove=args.to_remove)
    # if args.bottom_folder == 1:
    #     merge_one_experiment(args)
    # if args.top_folder == 1:
    #     merge_all_experiments(args)

