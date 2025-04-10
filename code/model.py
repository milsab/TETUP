import os
import math
import time
import random
import pickle
import numpy as np
import pandas as pd
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

DATASET = 'movies'


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # For multi-GPU or CUDA:
        torch.cuda.manual_seed_all(seed)
        print(f"Seed set to {seed}")
    else:
        print("No seed provided. Using default randomness.")


# -------------------
# Configuration
# -------------------
class Config:
    # Data paths
    ITEM_EMBEDDINGS_PATH = f"../data/{DATASET}/bert_item_features.pkl"
    USER_SHORT_TERM_PATH = f"../data/{DATASET}/bert_short_term_user_profiles.pkl"
    USER_LONG_TERM_PATH = f"../data/{DATASET}/bert_long_term_user_profiles.pkl"

    TRAIN_PATH = f"../data/{DATASET}/train.csv"
    VAL_PATH = f"../data/{DATASET}/validation.csv"
    TEST_PATH = f"../data/{DATASET}/test.csv"

    BATCH_SIZE = 2048
    NUM_NEG_SAMPLES = 5
    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    EPOCHS = 100
    EARLY_STOP_PATIENCE = 5

    EMBEDDING_DIM = 384
    HIDDEN_DIM = 256
    DROPOUT = 0.2

    NUM_WORKERS = 4
    MULTI_GPU = True


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_interactions_csv(path):
    df = pd.read_csv(path)
    return df


def negative_sampling(positive_item_ids, all_item_ids, num_neg_samples):

    negative_ids = set()
    while len(negative_ids) < num_neg_samples:
        neg_id = random.choice(list(all_item_ids))
        if neg_id not in positive_item_ids:
            negative_ids.add(neg_id)
    return list(negative_ids)


# -------------------
# Dataset
# -------------------
class RecSysDataset(Dataset):
    def __init__(
            self,
            interactions_df,
            user_short_term_dict,
            user_long_term_dict,
            item_dict,
            num_neg_samples=5
    ):

        self.interactions = interactions_df
        self.user_short_term_dict = user_short_term_dict
        self.user_long_term_dict = user_long_term_dict
        self.item_dict = item_dict

        self.num_neg_samples = num_neg_samples

        self.user_pos_item_map = {}
        for row in self.interactions.itertuples(index=False):
            u_id = getattr(row, "user_id")
            i_id = getattr(row, "item_id")
            if u_id not in self.user_pos_item_map:
                self.user_pos_item_map[u_id] = set()
            self.user_pos_item_map[u_id].add(i_id)

        self.users = list(self.interactions["user_id"])
        self.items = list(self.interactions["item_id"])

        self.all_item_ids = list(self.item_dict.keys())
        self.num_items = len(self.all_item_ids)

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        u_id = self.users[idx]
        pos_i_id = self.items[idx]

        user_short_term_emb = self.user_short_term_dict[u_id]
        user_long_term_emb = self.user_long_term_dict[u_id]
        pos_item_emb = self.item_dict[pos_i_id]

        # Positive sample
        data = []
        data.append((
            user_short_term_emb.astype(np.float32),
            user_long_term_emb.astype(np.float32),
            pos_item_emb.astype(np.float32),
            1  # label = 1 for positive
        ))

        neg_item_ids = negative_sampling(
            positive_item_ids=self.user_pos_item_map[u_id],
            all_item_ids=set(self.items),
            num_neg_samples=self.num_neg_samples
        )

        for neg_i in neg_item_ids:
            neg_item_emb = self.item_dict[neg_i]
            data.append((
                user_short_term_emb.astype(np.float32),
                user_long_term_emb.astype(np.float32),
                neg_item_emb.astype(np.float32),
                0  # label = 0 for negative
            ))

        return data


def collate_fn(batch):
    flat_data = []
    for sample in batch:
        flat_data.extend(sample)

    user_short_term_embs = []
    user_long_term_embs = []
    item_embs = []
    labels = []

    for (u_s, u_l, i, lbl) in flat_data:
        user_short_term_embs.append(u_s)
        user_long_term_embs.append(u_l)
        item_embs.append(i)
        labels.append(lbl)

    user_short_term_embs = torch.stack([torch.from_numpy(u_s) for u_s in user_short_term_embs])
    user_long_term_embs = torch.stack([torch.from_numpy(u_l) for u_l in user_long_term_embs])
    item_embs = torch.stack([torch.from_numpy(i) for i in item_embs])
    labels = torch.tensor(labels, dtype=torch.long)

    return user_short_term_embs, user_long_term_embs, item_embs, labels


class AttentionLayer(nn.Module):

    def __init__(self, embed_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self.softmax = nn.Softmax(dim=1)
        self._initialize_weights()

    def forward(self, short_term_emb, long_term_emb):

        combined = torch.stack([short_term_emb, long_term_emb], dim=1)

        B, N, E = combined.size()
        combined_reshape = combined.view(B * N, E)

        attn_logits = self.fc(combined_reshape)

        attn_logits = attn_logits.view(B, N)

        attn_weights = self.softmax(attn_logits)

        attn_weights = attn_weights.unsqueeze(-1)

        user_rep = (combined * attn_weights).sum(dim=1)

        return user_rep

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


class RecSysModel(nn.Module):

    def __init__(self, embed_dim=384, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.attention = AttentionLayer(embed_dim, hidden_dim, dropout)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

        self._initialize_weights()

    def forward(self, user_s_emb, user_l_emb, item_emb):
        user_rep = self.attention(user_s_emb, user_l_emb)

        x = torch.cat([user_rep, item_emb], dim=1)

        logits = self.fc(x)
        return logits

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


def accuracy(preds, labels):
    predicted = torch.argmax(preds, dim=1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)


def precision_recall_ndcg(preds, labels):
    predicted = torch.argmax(preds, dim=1)
    tp = ((predicted == 1) & (labels == 1)).sum().item()
    fp = ((predicted == 1) & (labels == 0)).sum().item()
    fn = ((predicted == 0) & (labels == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    ndcg = precision

    return precision, recall, ndcg


# -------------------
# Ranking Metrics
# -------------------

def dcg_at_k(relevance_list, k):

    r = relevance_list[:k]
    dcg = 0.0
    for i, rel in enumerate(r):
        # log2(i+2) because i is 0-based
        dcg += rel / math.log2(i + 2)
    return dcg


def ndcg_at_k(relevance_list, k):

    # DCG for the actual order
    dcg = dcg_at_k(relevance_list, k)
    # DCG for the ideal ordering (sort by most relevant first)
    ideal = sorted(relevance_list, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranking(
        model,
        user_s_dict,
        user_l_dict,
        item_dict,
        test_df,
        device,
        top_k_list=[10, 20]
):
    model.eval()

    # 1) Build user -> set of ground truth items from test_df
    user_to_test_items = {}
    for row in test_df.itertuples(index=False):
        u_id = getattr(row, "user_id")
        i_id = getattr(row, "item_id")
        if u_id not in user_to_test_items:
            user_to_test_items[u_id] = set()
        user_to_test_items[u_id].add(i_id)

    all_item_ids = list(item_dict.keys())
    item_embs = [item_dict[i].astype("float32") for i in all_item_ids]
    item_embs_torch = torch.tensor(item_embs).to(device)

    metrics_accumulator = {k: {"precision": 0.0, "recall": 0.0, "ndcg": 0.0} for k in top_k_list}
    num_users_evaluated = 0

    for user_id, test_items in user_to_test_items.items():

        # Skip if user embeddings are missing
        if user_id not in user_s_dict or user_id not in user_l_dict:
            continue

        user_s_np = user_s_dict[user_id].astype("float32")
        user_l_np = user_l_dict[user_id].astype("float32")

        user_s_torch = torch.tensor(user_s_np).unsqueeze(0).to(device)  # [1, 384]
        user_l_torch = torch.tensor(user_l_np).unsqueeze(0).to(device)  # [1, 384]

        with torch.no_grad():
            user_rep = model.module.attention(user_s_torch, user_l_torch) \
                if isinstance(model, torch.nn.DataParallel) else \
                model.attention(user_s_torch, user_l_torch)  # [1, 384]

        user_rep_expanded = user_rep.repeat(item_embs_torch.size(0), 1)

        concat_vec = torch.cat([user_rep_expanded, item_embs_torch], dim=1)

        with torch.no_grad():
            logits = model.module.fc(concat_vec) if isinstance(model, torch.nn.DataParallel) else model.fc(concat_vec)

        probs = torch.softmax(logits, dim=1)[:, 1]

        sorted_indices = torch.argsort(probs, descending=True)

        ranked_item_ids = [all_item_ids[idx] for idx in sorted_indices.tolist()]

        relevant_items = test_items
        num_relevant = len(relevant_items)

        for k in top_k_list:
            top_k_items = ranked_item_ids[:k]

            hits = sum((1 for i in top_k_items if i in relevant_items))

            precision_k = hits / k
            recall_k = hits / num_relevant if num_relevant > 0 else 0.0

            relevance_list = [1 if item_id in relevant_items else 0 for item_id in top_k_items]
            ndcg_k = ndcg_at_k(relevance_list, k)

            metrics_accumulator[k]["precision"] += precision_k
            metrics_accumulator[k]["recall"] += recall_k
            metrics_accumulator[k]["ndcg"] += ndcg_k

        num_users_evaluated += 1

    # 4) Average across all users
    results = {}
    for k in top_k_list:
        if num_users_evaluated > 0:
            results[k] = {
                "precision": metrics_accumulator[k]["precision"] / num_users_evaluated,
                "recall": metrics_accumulator[k]["recall"] / num_users_evaluated,
                "ndcg": metrics_accumulator[k]["ndcg"] / num_users_evaluated
            }
        else:
            results[k] = {
                "precision": 0.0,
                "recall": 0.0,
                "ndcg": 0.0
            }
    return results


# -------------------
# Training and Validation
# -------------------
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    print("Starting Training...")
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0

    for batch_idx, (user_s_emb, user_l_emb, item_emb, labels) in enumerate(train_loader):
        print(f'Training Batch: {batch_idx}')
        user_s_emb = user_s_emb.to(device)
        user_l_emb = user_l_emb.to(device)
        item_emb = item_emb.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(user_s_emb, user_l_emb, item_emb)

        loss = criterion(logits, labels)
        loss.backward()

        optimizer.step()

        # Compute metrics
        batch_acc = accuracy(logits, labels)

        epoch_loss += loss.item()
        epoch_acc += batch_acc

    epoch_loss /= len(train_loader)
    epoch_acc /= len(train_loader)

    return epoch_loss, epoch_acc


def evaluate(model, val_loader, criterion, device):
    print('Start Validation ...')
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    val_precision = 0.0
    val_recall = 0.0
    val_ndcg = 0.0

    with torch.no_grad():
        for batch_idx, (user_s_emb, user_l_emb, item_emb, labels) in enumerate(val_loader):
            print(f'Validation Batch: {batch_idx}')
            user_s_emb = user_s_emb.to(device)
            user_l_emb = user_l_emb.to(device)
            item_emb = item_emb.to(device)
            labels = labels.to(device)

            logits = model(user_s_emb, user_l_emb, item_emb)
            loss = criterion(logits, labels)

            val_loss += loss.item()
            val_acc += accuracy(logits, labels)

            # Additional metrics
            prec, rec, ndcg = precision_recall_ndcg(logits, labels)
            val_precision += prec
            val_recall += rec
            val_ndcg += ndcg

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    val_precision /= len(val_loader)
    val_recall /= len(val_loader)
    val_ndcg /= len(val_loader)

    return val_loss, val_acc, val_precision, val_recall, val_ndcg


# -------------------
# Early Stopping
# -------------------
class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=5, verbose=False, delta=1e-4, checkpoint_path="checkpoint.pt"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.checkpoint_path = checkpoint_path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), self.checkpoint_path)
        self.val_loss_min = val_loss


def main(seed):
    cfg = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading data...")
    item_df = load_pickle(cfg.ITEM_EMBEDDINGS_PATH)
    user_s_df = load_pickle(cfg.USER_SHORT_TERM_PATH)
    user_l_df = load_pickle(cfg.USER_LONG_TERM_PATH)

    train_df = load_interactions_csv(cfg.TRAIN_PATH)
    val_df = load_interactions_csv(cfg.VAL_PATH)
    test_df = load_interactions_csv(cfg.TEST_PATH)

    item_dict = {int(row["item_id"]): row["description"] for _, row in item_df.iterrows()}
    user_s_dict = {int(row["user_id"]): row["profile"] for _, row in user_s_df.iterrows()}
    user_l_dict = {int(row["user_id"]): row["profile"] for _, row in user_l_df.iterrows()}

    train_dataset = RecSysDataset(
        interactions_df=train_df,
        user_short_term_dict=user_s_dict,
        user_long_term_dict=user_l_dict,
        item_dict=item_dict,
        num_neg_samples=cfg.NUM_NEG_SAMPLES
    )
    val_dataset = RecSysDataset(
        interactions_df=val_df,
        user_short_term_dict=user_s_dict,
        user_long_term_dict=user_l_dict,
        item_dict=item_dict,
        num_neg_samples=cfg.NUM_NEG_SAMPLES
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn
    )

    model = RecSysModel(
        embed_dim=cfg.EMBEDDING_DIM,
        hidden_dim=cfg.HIDDEN_DIM,
        dropout=cfg.DROPOUT
    )

    if cfg.MULTI_GPU and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for DataParallel.")
        model = nn.DataParallel(model)

    model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss()

    early_stopper = EarlyStopping(
        patience=cfg.EARLY_STOP_PATIENCE,
        verbose=True,
        checkpoint_path=f"best_model_Seed_{seed}.pt"
    )

    for epoch in range(cfg.EPOCHS):
        print(f"Processing Epoch {epoch + 1} ...")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_precision, val_recall, val_ndcg = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch + 1}/{cfg.EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f} | Val NDCG: {val_ndcg:.4f}")

        # Early stopping check
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    # -------------------
    # Load best model
    # -------------------
    print("Loading best model weights...")
    model.load_state_dict(torch.load(f"best_model_Seed_{seed}.pt"))

    # -------------------
    # Testing (batch-based evaluation)
    # -------------------
    test_dataset = RecSysDataset(
        interactions_df=test_df,
        user_short_term_dict=user_s_dict,
        user_long_term_dict=user_l_dict,
        item_dict=item_dict,
        num_neg_samples=cfg.NUM_NEG_SAMPLES
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn
    )

    print("Starting Testing...")
    test_loss, test_acc, test_precision, test_recall, test_ndcg = evaluate(model, test_loader, criterion, device)

    print(f"Test Results -> Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, "
          f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, NDCG: {test_ndcg:.4f}")

    print("Starting Ranking-Based Testing...")

    top_k_list = [10, 20]
    ranking_results = evaluate_ranking(
        model=model,
        user_s_dict=user_s_dict,
        user_l_dict=user_l_dict,
        item_dict=item_dict,
        test_df=test_df,
        device=device,
        top_k_list=top_k_list
    )

    # (3) Print the results
    for k in top_k_list:
        print(f"[Ranking] Top-{k} => "
              f"Precision: {ranking_results[k]['precision']:.4f}, "
              f"Recall: {ranking_results[k]['recall']:.4f}, "
              f"NDCG: {ranking_results[k]['ndcg']:.4f}")

    print("Done.")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Set seed for reproducibility.")
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Seed for random number generators'
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    set_seed(args.seed)
    start_time = time.time()
    main(args.seed)
    print(f'Execution Time: {time.time() - start_time}.')
