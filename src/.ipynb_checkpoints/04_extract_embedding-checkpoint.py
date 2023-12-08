import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import logging
import argparse
from utils import get_cell_type_compound_gene


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="This script extracts the cell type, small molecualr, and gene embedding from trained model",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--de_train",
        type=str,
        default=None,
        help=("String of filname for de_train.parquet."),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=("Filname for model."),
    )
    parser.add_argument(
        "--splited_data_dir",
        type=str,
        default=None,
        help=("Splited data directory."),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help=("Output directory."),
    )
    return parser.parse_args()


def get_data(
    df,
    cell_type_embedding,
    compound_embedding,
    gene_embedding,
    cell_types,
    compounds,
    genes,
    get_target=True,
):
    n_samples = len(df)
    n_features = (
        cell_type_embedding.shape[1]
        + compound_embedding.shape[1]
        + gene_embedding.shape[1]
    )

    # get features
    x = np.zeros((n_samples, n_features))
    cell_type_idxs = np.zeros(n_samples)
    compound_idxs = np.zeros(n_samples)
    gene_idxs = np.zeros(n_samples)
    for i in tqdm(range(n_samples)):
        cell_type_idx = cell_types.index(df["cell_type"][i])
        compound_idx = compounds.index(df["sm_name"][i])
        gene_idx = genes.index(df["gene"][i])

        cell_type_vec = cell_type_embedding[cell_type_idx]
        compound_vec = compound_embedding[compound_idx]
        gene_vec = gene_embedding[gene_idx]

        x[i] = torch.concat([cell_type_vec, compound_vec, gene_vec])
        cell_type_idxs[i] = cell_type_idx
        compound_idxs[i] = compound_idx
        gene_idxs[i] = gene_idx

    if get_target:
        y = np.zeros(n_samples)
        for i in range(n_samples):
            y[i] = df["target"][i]

        return x, y, cell_type_idxs, compound_idxs, gene_idxs
    else:
        return x, cell_type_idxs, compound_idxs, gene_idxs


def main():
    args = parse_args()

    cell_types, compounds, genes = get_cell_type_compound_gene(args.de_train)

    cell_type_names = {
        "NK cells": "nk",
        "T cells CD4+": "t_cd4",
        "T cells CD8+": "t_cd8",
        "T regulatory cells": "t_reg",
    }

    # get embedding for cell type, compound, and gene
    state_dict = torch.load(args.model)
    cell_type_embedding = state_dict["state_dict"]["cell_type_embedding.weight"].cpu()
    compound_embedding = state_dict["state_dict"]["compound_embedding.weight"].cpu()
    gene_embedding = state_dict["state_dict"]["gene_embedding.weight"].cpu()

    df = pd.read_parquet(args.de_train)

    logging.info("Extracting embeddings for training and validation data")
    for key, cell_type in cell_type_names.items():
        print(cell_type)

        df_train = pd.read_csv(f"{args.splited_data_dir}/train_{cell_type}.csv")
        df_valid = pd.read_csv(f"{args.splited_data_dir}/valid_{cell_type}.csv")

        # training data
        x, y, cell_type_idxs, compound_idxs, gene_idxs = get_data(
            df=df_train,
            cell_type_embedding=cell_type_embedding,
            compound_embedding=compound_embedding,
            gene_embedding=gene_embedding,
            cell_types=cell_types,
            compounds=compounds,
            genes=genes,
        )

        np.savez(
            f"{args.out_dir}/train_{cell_type}.npz",
            x=x,
            y=y,
            cell_types=cell_type_idxs,
            compounds=compound_idxs,
            genes=gene_idxs,
        )

        # validation data
        x, y, cell_type_idxs, compound_idxs, gene_idxs = get_data(
            df=df_valid,
            cell_type_embedding=cell_type_embedding,
            compound_embedding=compound_embedding,
            gene_embedding=gene_embedding,
            cell_types=cell_types,
            compounds=compounds,
            genes=genes,
        )
        np.savez(
            f"{args.out_dir}/valid_{cell_type}.npz",
            x=x,
            y=y,
            cell_types=cell_type_idxs,
            compounds=compound_idxs,
            genes=gene_idxs,
        )

    logging.info("Extracting embeddings for test data")
    df_test = pd.read_csv(f"{args.splited_data_dir}/test.csv")

    x, cell_type_idxs, compound_idxs, gene_idxs = get_data(
        df=df_test,
        cell_type_embedding=cell_type_embedding,
        compound_embedding=compound_embedding,
        gene_embedding=gene_embedding,
        cell_types=cell_types,
        compounds=compounds,
        genes=genes,
        get_target=False,
    )

    np.savez(
        f"{args.out_dir}/test.npz",
        x=x,
        cell_types=cell_type_idxs,
        compounds=compound_idxs,
        genes=gene_idxs,
    )


if __name__ == "__main__":
    main()
