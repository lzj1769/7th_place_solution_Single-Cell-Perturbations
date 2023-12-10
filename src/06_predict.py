import argparse
import os
import sys
import numpy as np
import pandas as pd
import warnings
import torch
import logging
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from model import PerturbNet
from dataset_prediction import get_dataloader
from utils import set_seed, get_submission

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


if not sys.warnoptions:
    warnings.simplefilter("ignore")


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--input_data_dir",
        type=str,
        default=None,
        help="Directory for training data",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Directory for models",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="Filename for test data",
    )
    parser.add_argument(
        "--submission_dir",
        type=str,
        default=None,
        help="Directory for submission files",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    return parser.parse_args()


def predict(model, dataloader, device):
    model.eval()

    preds = list()
    for x in tqdm(dataloader):
        pred = model(x.to(device)).detach().cpu().view(-1).tolist()
        preds.append(pred)

    preds = np.concatenate(preds)

    return preds


def main():
    args = parse_args()

    # set random seed
    set_seed(args.seed)

    device = torch.device("cuda")

    os.makedirs(args.submission_dir, exist_ok=True)

    df_submission_list = []
    avg_train_loss, avg_train_mrrmse, avg_valid_loss, avg_valid_mrrmse = 0, 0, 0, 0
    for cell_type in ["nk", "t_cd4", "t_cd8", "t_reg"]:
        logging.info(f"Predicting with cell type as validation: {cell_type}")

        train_deep_tf = np.load(f"{args.input_data_dir}/train_{cell_type}.npz")
        valid_deep_tf = np.load(f"{args.input_data_dir}/valid_{cell_type}.npz")
        test_deep_tf = np.load(f"{args.input_data_dir}/test.npz")

        # load data
        train_x, valid_x, test_x = (
            train_deep_tf["x"],
            valid_deep_tf["x"],
            test_deep_tf["x"],
        )

        logging.info("Standarizing the features")
        scaler = StandardScaler()
        scaler.fit(X=np.concatenate([train_x, valid_x, test_x], axis=0))
        test_x = scaler.transform(test_x)

        logging.info(
            f"Number of test samples: {test_x.shape[0]}; number of features: {test_x.shape[1]}"
        )

        test_loader = get_dataloader(
            x=test_x, num_workers=2, drop_last=False, shuffle=False, train=False
        )

        # Setup model
        model = PerturbNet(n_input=test_x.shape[1])
        state_dict = torch.load(f'{args.model_dir}/{cell_type}.pth')
        model.load_state_dict(state_dict["state_dict"])

        train_loss = state_dict["train_loss"]
        valid_loss = state_dict["valid_loss"]
        train_mrrmse = state_dict["train_mrrmse"]
        valid_mrrmse = state_dict["valid_mrrmse"]

        model.to(device)

        # predict target
        df_test = pd.read_csv(args.test_file)
        model.eval()
        preds = list()
        for x in tqdm(test_loader):
            pred = model(x.to(device)).detach().cpu().view(-1).tolist()
            preds.append(pred)

        df_test["predict"] = np.concatenate(preds)
        df_submission = get_submission(df_test)

        filename = f"{cell_type}_valid_mrrmse_{valid_mrrmse:.03f}.csv"

        df_submission.to_csv(f"{args.submission_dir}/{filename}")

        df_submission_list.append(df_submission)

        avg_train_loss += train_loss / 4
        avg_train_mrrmse += train_mrrmse / 4
        avg_valid_loss += valid_loss / 4
        avg_valid_mrrmse += valid_mrrmse / 4

    # get average submission
    df_avg_submission = sum(df_submission_list) / len(df_submission_list)
    df_avg_submission.to_csv(f"{args.submission_dir}/avg_valid_mrrmse_{avg_valid_mrrmse:.03f}.csv")


if __name__ == "__main__":
    main()
