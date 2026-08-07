# Copyright (c) the Lab of Intelligent Data Processing, Wakayama University.
# All rights reserved.

# train.py
# Example:
# python train.py --mode train --epochs 256 --cnn_activation maxout --mlp_activation maxout --pe_type rope --loss_type sent_frame

import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error

from Dataset_Loader import split_dataset, collate_fn_pad
from Gesinet_module import GESINet_Strict


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_corr(func, y_true, y_pred):
    try:
        value = func(y_true, y_pred)[0]
        if np.isnan(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def compute_metrics(targets, preds):
    targets = np.asarray(targets).reshape(-1)
    preds = np.asarray(preds).reshape(-1)

    mse = mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)
    lcc = safe_corr(pearsonr, targets, preds)
    srcc = safe_corr(spearmanr, targets, preds)

    try:
        ktau = kendalltau(targets, preds)[0]
        if np.isnan(ktau):
            ktau = 0.0
    except Exception:
        ktau = 0.0

    return {
        "mse": mse,
        "rmse": rmse,
        "lcc": lcc,
        "srcc": srcc,
        "ktau": ktau,
    }


def compute_loss(pred_sent, pred_frames, scores, loss_type="sent_frame", frame_loss_lambda=1.0):
    scores = scores.view(-1, 1)

    loss_sent = F.mse_loss(pred_sent, scores)

    if loss_type == "sent":
        loss_frame = torch.tensor(0.0, device=scores.device)
        loss = loss_sent

    elif loss_type == "sent_frame":
        scores_expand = scores.unsqueeze(1).expand_as(pred_frames)
        loss_frame = F.mse_loss(pred_frames, scores_expand)
        loss = loss_sent + frame_loss_lambda * loss_frame

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss, loss_sent, loss_frame


def run_one_epoch(model, loader, optimizer, device, args, train=True, epoch=0):
    if train:
        model.train()
        desc = f"Epoch {epoch} [Train]"
    else:
        model.eval()
        desc = f"Epoch {epoch} [Val]"

    total_loss = 0.0
    total_sent_loss = 0.0
    total_frame_loss = 0.0

    preds = []
    targets = []

    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for waveforms, scores in tqdm(loader, desc=desc):
            waveforms = waveforms.to(device)
            scores = scores.to(device).view(-1, 1)

            if train:
                optimizer.zero_grad()

            pred_sent, pred_frames = model(waveforms)

            loss, loss_sent, loss_frame = compute_loss(
                pred_sent=pred_sent,
                pred_frames=pred_frames,
                scores=scores,
                loss_type=args.loss_type,
                frame_loss_lambda=args.frame_loss_lambda,
            )

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_sent_loss += loss_sent.item()
            total_frame_loss += loss_frame.item()

            preds.extend(pred_sent.detach().squeeze(1).cpu().numpy())
            targets.extend(scores.detach().squeeze(1).cpu().numpy())

    avg_loss = total_loss / len(loader)
    avg_sent_loss = total_sent_loss / len(loader)
    avg_frame_loss = total_frame_loss / len(loader)

    metrics = compute_metrics(targets, preds)

    return avg_loss, avg_sent_loss, avg_frame_loss, metrics


def evaluate_model(model, loader, device):
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for waveforms, scores in tqdm(loader, desc="Test"):
            waveforms = waveforms.to(device)
            scores = scores.to(device).view(-1, 1)

            pred_sent, _ = model(waveforms)

            preds.extend(pred_sent.squeeze(1).cpu().numpy())
            targets.extend(scores.squeeze(1).cpu().numpy())

    metrics = compute_metrics(targets, preds)

    print("\n========== Test Result ==========")
    print(f"MSE  : {metrics['mse']:.6f}")
    print(f"RMSE : {metrics['rmse']:.6f}")
    print(f"LCC  : {metrics['lcc']:.6f}")
    print(f"SRCC : {metrics['srcc']:.6f}")
    print(f"KTAU : {metrics['ktau']:.6f}")

    return np.asarray(targets), np.asarray(preds), metrics


def plot_loss(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 8))

    epochs_range = range(1, len(train_losses) + 1)

    plt.plot(epochs_range, train_losses, label="Train")
    plt.plot(epochs_range, val_losses, label="Val")

    best_epoch = int(np.argmin(val_losses))
    best_val = val_losses[best_epoch]
    best_train = train_losses[best_epoch]

    plt.scatter(
        best_epoch + 1,
        best_val,
        marker="*",
        s=100,
        label=f"Best Val Epoch {best_epoch + 1}: {best_val:.4f}",
    )

    plt.scatter(
        best_epoch + 1,
        best_train,
        marker="o",
        s=80,
        label=f"Train @ Best: {best_train:.4f}",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=600)

    if save_path.endswith(".png"):
        plt.savefig(save_path.replace(".png", ".eps"), dpi=600, format="eps")

    plt.close()


def plot_scatter(targets, preds, metrics, save_path):
    plt.figure(figsize=(6, 6))

    plt.scatter(targets, preds, s=15, alpha=0.8)
    plt.plot([0, 1], [0, 1], "--", linewidth=1)

    plt.xlabel("True GESI", fontsize=18)
    plt.ylabel("Predicted GESI", fontsize=18)

    plt.title(
    	f"MSE={metrics['mse']:.4f}, "
    	f"LCC={metrics['lcc']:.4f}, "
    	f"SRCC={metrics['srcc']:.4f}",
    	fontsize=14,)

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=600)

    if save_path.endswith(".png"):
        plt.savefig(save_path.replace(".png", ".eps"), dpi=600, format="eps")

    plt.close()


def plot_hist(targets, preds, save_path):
    plt.figure(figsize=(6, 4))

    plt.hist(targets, bins=20, range=(0, 1), alpha=0.6, label="GESI", edgecolor="black")
    plt.hist(preds, bins=20, range=(0, 1), alpha=0.6, label="DeepGESI", edgecolor="black")

    plt.xlabel("GESI Metric")
    plt.ylabel("Count")
    plt.title("Distribution: DeepGESI vs GESI")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=600)

    if save_path.endswith(".png"):
        plt.savefig(save_path.replace(".png", ".eps"), dpi=600, format="eps")

    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])

    parser.add_argument("--json_path", type=str, default="gesi_scores_from_ref_traindataset.json")
    parser.add_argument(
        "--wav_root",
        type=str,
        default="/home/dl-box/RA(luowenyu/clarity_CPC2_data.v1_1/clarity_CPC2_data/clarity_data/HA_outputs/signals/CEC2",
    )
    parser.add_argument("--sample_rate", type=int, default=16000)

    parser.add_argument("--epochs", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=int, default=0)

    parser.add_argument("--use_attention", type=int, default=1, choices=[0, 1])

    parser.add_argument(
        "--pe_type",
        type=str,
        default="rope",
        choices=["none", "sinusoidal", "t5", "rope"],
    )

    parser.add_argument(
        "--cnn_activation",
        type=str,
        default="maxout",
        choices=["maxout", "relu", "prelu", "lrelu", "silu"],
    )

    parser.add_argument(
        "--mlp_activation",
        type=str,
        default="maxout",
        choices=["maxout", "relu", "prelu", "lrelu", "silu"],
    )

    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--flatten_dropout", type=float, default=0.1)
    parser.add_argument("--t5_max_distance", type=int, default=128)

    parser.add_argument(
        "--loss_type",
        type=str,
        default="sent_frame",
        choices=["sent", "sent_frame"],
    )
    parser.add_argument("--frame_loss_lambda", type=float, default=0.5)

    parser.add_argument("--model_save_path", type=str, default="challenge_best_model.pth")
    parser.add_argument("--loss_plot_path", type=str, default="train_val_loss.png")
    parser.add_argument("--scatter_plot_path", type=str, default="scatter_gesi_test.png")
    parser.add_argument("--hist_plot_path", type=str, default="hist_pred_vs_true.png")

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("========== Experiment Setting ==========")
    print(f"mode              : {args.mode}")
    print(f"device            : {device}")
    print(f"json_path         : {args.json_path}")
    print(f"wav_root          : {args.wav_root}")
    print(f"epochs            : {args.epochs}")
    print(f"batch_size        : {args.batch_size}")
    print(f"lr                : {args.lr}")
    print(f"patience          : {args.patience}")
    print(f"use_attention     : {args.use_attention}")
    print(f"pe_type           : {args.pe_type}")
    print(f"cnn_activation    : {args.cnn_activation}")
    print(f"mlp_activation    : {args.mlp_activation}")
    print(f"loss_type         : {args.loss_type}")
    print(f"frame_loss_lambda : {args.frame_loss_lambda}")
    print(f"model_save_path   : {args.model_save_path}")
    print("========================================")

    train_set, val_set, test_set = split_dataset(
        args.json_path,
        args.wav_root,
        target_sr=args.sample_rate,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_pad,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_pad,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_pad,
    )

    model = GESINet_Strict(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        use_attention=bool(args.use_attention),
        pe_type=args.pe_type,
        cnn_activation=args.cnn_activation,
        mlp_activation=args.mlp_activation,
        attn_dropout=args.attn_dropout,
        flatten_dropout=args.flatten_dropout,
        t5_max_distance=args.t5_max_distance,
    ).to(device)

    if args.mode == "train":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        if args.resume == 1 and os.path.exists(args.model_save_path):
            print(f"Loading existing model from {args.model_save_path}")
            model.load_state_dict(torch.load(args.model_save_path, map_location=device))

        best_val_loss = float("inf")
        early_stop_counter = 0

        train_losses = []
        val_losses = []

        for epoch in range(1, args.epochs + 1):
            train_loss, train_sent_loss, train_frame_loss, train_metrics = run_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                args=args,
                train=True,
                epoch=epoch,
            )

            val_loss, val_sent_loss, val_frame_loss, val_metrics = run_one_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                device=device,
                args=args,
                train=False,
                epoch=epoch,
            )

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"\n[Epoch {epoch}/{args.epochs}]")
            print(
                f"Train | Loss={train_loss:.6f} "
                f"Sent={train_sent_loss:.6f} Frame={train_frame_loss:.6f} | "
                f"MSE={train_metrics['mse']:.6f} "
                f"RMSE={train_metrics['rmse']:.6f} "
                f"LCC={train_metrics['lcc']:.6f} "
                f"SRCC={train_metrics['srcc']:.6f} "
                f"KTAU={train_metrics['ktau']:.6f}"
            )
            print(
                f"Val   | Loss={val_loss:.6f} "
                f"Sent={val_sent_loss:.6f} Frame={val_frame_loss:.6f} | "
                f"MSE={val_metrics['mse']:.6f} "
                f"RMSE={val_metrics['rmse']:.6f} "
                f"LCC={val_metrics['lcc']:.6f} "
                f"SRCC={val_metrics['srcc']:.6f} "
                f"KTAU={val_metrics['ktau']:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), args.model_save_path)
                print(f"Best model saved: {args.model_save_path}")
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                print(f"No improvement: {early_stop_counter}/{args.patience}")

            if early_stop_counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        plot_loss(train_losses, val_losses, args.loss_plot_path)
        print(f"Loss curve saved to {args.loss_plot_path}")

    elif args.mode == "test":
        print(f"Loading model from {args.model_save_path}")
        model.load_state_dict(torch.load(args.model_save_path, map_location=device))

        targets, preds, metrics = evaluate_model(model, test_loader, device)

        plot_scatter(targets, preds, metrics, args.scatter_plot_path)
        plot_hist(targets, preds, args.hist_plot_path)

        print(f"Scatter plot saved to {args.scatter_plot_path}")
        print(f"Histogram saved to {args.hist_plot_path}")


if __name__ == "__main__":
    main()
