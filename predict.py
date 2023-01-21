import argparse
import time

import numpy as np
import pandas as pd
import streamlit as st
import torch

from models.model import S3RecModel
from models.utils import get_item2attribute_json, get_user_seqs, set_seed


def get_random_rec(top_k):
    top_k = int(top_k)
    df = pd.read_csv("poster.csv", sep="\t")
    df.fillna("", inplace=True)
    return df.sample(top_k)


def get_model_rec(model, input_ids, top_k):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    top_k = int(top_k)
    model.eval()

    pad_len = 50 - len(input_ids)
    input_ids = [0] * pad_len + input_ids
    input_ids = torch.as_tensor([input_ids])
    input_ids = input_ids.to(device)
    recommend_output = model.finetune(input_ids)

    recommend_output = recommend_output[:, -1, :]
    test_item_emb = model.item_embeddings.weight
    rating_pred = torch.matmul(recommend_output, test_item_emb.transpose(0, 1))

    rating_pred = rating_pred.cpu().data.numpy().copy()[0]
    input_ids = input_ids.cpu()

    rating_pred[input_ids[0]] = 0

    pred_ids = np.argsort(rating_pred)[::-1][:top_k]
    df = pd.read_csv("poster.csv", sep="\t")
    df.fillna("", inplace=True)
    return df[df["item"].isin(pred_ids)]


@st.cache
def load_model():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="hidden size of transformer model",
    )
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument(
        "--attention_probs_dropout_prob",
        type=float,
        default=0.5,
        help="attention dropout p",
    )
    parser.add_argument(
        "--hidden_dropout_prob",
        type=float,
        default=0.5,
        help="hidden dropout p",
    )

    parser.add_argument("--max_seq_length", default=50, type=int)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()
    args.cuda_condition = torch.cuda.is_available() 
    # args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    user_seq, max_item, _, _, submission_rating_matrix = get_user_seqs("models/train_ratings.csv")
    item2attribute, attribute_size = get_item2attribute_json("models/Ml_item2attributes.json")
    set_seed(args.seed)
    args.mask_id = max_item + 1
    args.item_size = max_item + 2
    args.attribute_size = attribute_size + 1

    model = S3RecModel(args=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("models/model.pt", map_location=device))
    return model
