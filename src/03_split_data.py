import os
import pandas as pd
import logging
import argparse

from config import control_ids, privte_ids, public_ids

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="This script Split the raw data for training and validation",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Required parameters
    parser.add_argument(
        "--de_train",
        type=str,
        default=None,
        help=("String of filname for de_train.parquet."),
    )
    parser.add_argument(
        "--sample_submission",
        type=str,
        default=None,
        help=("Filname for sample_submission.csv."),
    )
    parser.add_argument(
        "--id_map",
        type=str,
        default=None,
        help=("Filname for id_map.csv."),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help=("Output directory."),
    )
    return parser.parse_args()


def convert_to_long_df(df):
    df = df.drop(["sm_lincs_id", "SMILES", "control"], axis=1)
    df = pd.melt(
        df, id_vars=["cell_type", "sm_name"], var_name="gene", value_name="target"
    )

    return df


def main():
    args = parse_args()

    cell_type_names = {
        "NK cells": "nk",
        "T cells CD4+": "t_cd4",
        "T cells CD8+": "t_cd8",
        "T regulatory cells": "t_reg",
    }

    df = pd.read_parquet(args.de_train)

    # prepare training and validation dataset
    logging.info("Splitting data for training and validation")
    for key, cell_type in cell_type_names.items():
        # split data for training and validation, here we used the private test compounds for validation
        df_train = df[(df["cell_type"] != key) | ~df["sm_lincs_id"].isin(privte_ids)]
        df_valid = df[(df["cell_type"] == key) & df["sm_lincs_id"].isin(privte_ids)]

        df_train = df_train.sort_values(["cell_type", "sm_name"])
        df_valid = df_valid.sort_values("sm_name")

        df_train = convert_to_long_df(df_train)
        df_valid = convert_to_long_df(df_valid)

        df_train.to_csv(f"{args.out_dir}/train_{cell_type}.csv")
        df_valid.to_csv(f"{args.out_dir}/valid_{cell_type}.csv")

    # prepare test dataset
    logging.info("Preparing data for testing")
    df_sample = pd.read_csv(args.sample_submission, index_col=0)
    df_test = pd.read_csv(args.id_map, index_col=0)

    df_sample["cell_type"] = df_test["cell_type"]
    df_sample["sm_name"] = df_test["sm_name"]

    df_test = pd.melt(
        df_sample,
        id_vars=["cell_type", "sm_name"],
        var_name="gene",
        value_name="predict",
    )

    df_test.to_csv(f"{args.out_dir}/test.csv")


if __name__ == "__main__":
    main()
