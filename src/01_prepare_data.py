import pandas as pd
import argparse
import logging


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="This script prepares training data for learning embeddings",
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
        "--output_filename",
        type=str,
        default=None,
        help=("Output filename."),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.info(f"Reading raw data from {args.de_train}")
    df = pd.read_parquet(args.de_train)

    # convert to long df
    logging.info(f"Converting dataframe to long format")
    df = df.sort_values(["cell_type", "sm_name"])
    df = df.drop(["sm_lincs_id", "SMILES", "control"], axis=1)
    df = pd.melt(
        df, id_vars=["cell_type", "sm_name"], var_name="gene", value_name="target"
    )

    logging.info(f"Saving data")
    df.to_csv(args.output_filename)


if __name__ == "__main__":
    main()
