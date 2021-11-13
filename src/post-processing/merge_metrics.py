import argparse
import os
import pandas as pd
import re
import math

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", type=str, required=True,
                        help="data folder containing experiments")
    parser.add_argument('-columns_to_save', nargs='+', type=str,
                        default=["oracle_score", "recall_5", "bleu", "meteor", "cider", "ppl_dialog_lm",
                                 "language_score", "selfbleu", "kurtosis", "peakiness"], help="")
    parser.add_argument('-bottom_folder', type=int, default=1)
    parser.add_argument('-top_folder', type=int, default=1)
    parser.add_argument('-precision', type=int, default=4)

    return parser


def add_to_metrics(df, path_exp):
    pattern = "test_(test_images|test_text)_(no_trunc|with_trunc)_(sampling_ranking_lm|sampling|greedy)_vilbert_recall_rewards.csv"
    all = {"no_trunc": {}, "with_trunc": {}}
    if "oracle_score" not in df.index:
        means = {trunc: [] for trunc in list(df.columns)}
        root, _, files = next(os.walk(os.path.join(path_exp, "metrics")))
        f_patterns = {f: re.findall(pattern, f) for f in files}
        for f, patterns in f_patterns.items():
            if "train" not in f and patterns and len(patterns[0]) == 3:
                try:
                    mode, trunc, sampl = patterns[0]
                    id = "_".join([mode, sampl, "oracle_score"])
                    rew_df = pd.read_csv(os.path.join(root, f), header=None)
                    mean_rew = rew_df[0].mean()
                    all[trunc][id] = mean_rew

                except pd.errors.EmptyDataError:
                    print("empty file {}".format(os.path.join(root, f)))
                    continue
        oracle_score = pd.DataFrame.from_dict(all)
        oracle_score.to_csv(os.path.join(path_exp, "stats", "oracle_score.csv"), )
        oracle_score_serie = oracle_score.mean()
        oracle_score_serie.name = "oracle_score"
        df = df.append(oracle_score_serie)
    return df

def str_to_float_in_csv(df, index="train_loss", sep=","):
    values = df.loc[index].values[-1]
    values = values.split(sep)
    values = [e.replace('[', '') for e in values]
    values = [e.replace(']', '') for e in values]
    values = [e.replace('\n', '') for e in values]
    values = [float(e) for e in values]
    return values[-1]

def merge_one_experiment(path="output/NLP/CLEVR/NEW_EXP__OLD"):
    dirs = [f.path for f in os.scandir(path) if f.is_dir()]
    merge_metrics_mean = pd.DataFrame()

    for dir_conf in dirs:
        dirs = [f.path for f in os.scandir(dir_conf) if f.is_dir()]
        stats_all_runs_df = pd.DataFrame()
        all_metrics_all_runs_df = pd.DataFrame()
        for dir_experiment in dirs: # level for multiple runs with same config.
            losses_path = os.path.join(dir_experiment, "smc_t_history_1.csv")
            if os.path.exists(losses_path):
                losses_exp = pd.read_csv(losses_path, index_col=0, header=None)
                losses_values = []
                for index in ["train_loss", "val_loss", "train_mse_metric", "val_mse_metric"]:
                    sep = "," if index in ["train_loss", "val_loss", "train_mse_metric", "val_mse_metric"] else " "
                    losses_values.append(str_to_float_in_csv(losses_exp, index, sep=sep))
                all_metrics_df = pd.Series(losses_values, index=["train_loss", "val_loss", "train_mse_metric", "val_mse_metric"])
                all_metrics_df.loc["train_ppl"] = math.exp(all_metrics_df["train_mse_metric"])
                all_metrics_df.loc["val_ppl"] = math.exp(all_metrics_df["val_mse_metric"])
            test_metrics_path = os.path.join(dir_experiment, "test_metrics_mean_sampling.csv")
            if os.path.exists(test_metrics_path):
                test_metrics_exp = pd.read_csv(test_metrics_path, index_col=0, header=None)
                all_metrics_df.append(test_metrics_exp.squeeze())
            dir_name = os.path.split(dir_experiment)[-1]
            all_metrics_all_runs_df[dir_name] = all_metrics_df
        stats_all_runs_df["mean"] = all_metrics_all_runs_df.mean(axis=1)
        stats_all_runs_df["std"] = all_metrics_all_runs_df.std(axis=1)
        stats_all_runs_df.to_csv(os.path.join(dir_conf, "all_metrics.csv"))
        all_metrics_all_runs_df.to_csv(os.path.join(dir_conf), "all_metrics_per_run.csv")

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
    merge_one_experiment()
    # parser = get_parser()
    # args = parser.parse_args()
    # if args.bottom_folder == 1:
    #     merge_one_experiment(args)
    # if args.top_folder == 1:
    #     merge_all_experiments(args)
