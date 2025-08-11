import math
import time
import random
import pickle
import wandb
import numpy as np
import pandas as pd
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


DATASET = 'games'

# -------------------
# Set seeds
# -------------------
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
    # ITEM_EMBEDDINGS_PATH = f"../datasets/{DATASET}/bert_item_features_10K.pkl"
    # USER_EMBEDDINGS_PATH = f"../datasets/{DATASET}/bert_long_term_user_profiles_train_10K.pkl"
    #
    # TRAIN_PATH = f"../datasets/{DATASET}/train_sample10K.csv"
    # VAL_PATH = f"../datasets/{DATASET}/validation_sample10K.csv"
    # TEST_PATH = f"../datasets/{DATASET}/test_sample10K.csv"

    ITEM_EMBEDDINGS_PATH = f"../datasets/{DATASET}/bert_item_features.pkl"
    USER_EMBEDDINGS_PATH = f"../datasets/{DATASET}/bert_short_term_user_profiles_train.pkl"

    TRAIN_PATH = f"../datasets/{DATASET}/train.csv"
    VAL_PATH = f"../datasets/{DATASET}/validation.csv"
    TEST_PATH = f"../datasets/{DATASET}/test.csv"

    # Training hyperparameters
    BATCH_SIZE = 2048
    NUM_NEG_SAMPLES = 5  # Number of negative samples per positive interaction
    LR = 1e-3  # Learning rate
    WEIGHT_DECAY = 1e-5  # Weight decay for regularization
    EPOCHS = 100
    EARLY_STOP_PATIENCE = 5  # Patience for early stopping

    # Model hyperparameters
    EMBEDDING_DIM = 384  # Dimension of user/item embeddings
    HIDDEN_DIM = 128  # Dimension of hidden layers in attention MLP
    DROPOUT = 0.2  # Dropout rate

    # Hardware settings
    NUM_WORKERS = 4  # Number of workers for DataLoader
    MULTI_GPU = True  # Use DataParallel if multiple GPUs are available

    # WANDB project information
    WANDB_PROJECT = "RecSys-Dual_Preferences"
    WANDB_PRE_NAME = f"Proposed-{DATASET}-Only-ShortTerm-NoAtt"
    WANDB_ENTITY = None  # Set your wandb entity if needed; else None


# -------------------
# Utility functions
# -------------------

def load_pickle(path):
    """Load data from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_interactions_csv(path):
    """Load interaction data from CSV."""
    df = pd.read_csv(path)
    return df


def negative_sampling(positive_item_ids, all_item_ids, num_neg_samples):
    """
    Return a set of negative item_ids given the existing positive item_ids.
    Used to sample negative items for a particular user.
    """
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
    """
    Custom dataset for recommendation.
    Performs negative sampling on-the-fly.
    """

    def __init__(
            self,
            interactions_df,
            user_dict,
            item_dict,
            num_neg_samples=5
    ):
        """
        Args:
            interactions_df (DataFrame): user-item interactions (positive samples).
            user_dict (dict): user_id -> 384-dim embedding.
            item_dict (dict): item_id -> 384-dim embedding.
            num_neg_samples (int): number of negative samples per positive sample.
        """
        self.interactions = interactions_df
        self.user_dict = user_dict
        self.item_dict = item_dict

        self.num_neg_samples = num_neg_samples

        # Precompute unique user -> set of item_ids for negative sampling
        self.user_pos_item_map = {}
        for row in self.interactions.itertuples(index=False):
            u_id = getattr(row, "user_id")
            i_id = getattr(row, "item_id")
            if u_id not in self.user_pos_item_map:
                self.user_pos_item_map[u_id] = set()
            self.user_pos_item_map[u_id].add(i_id)

        self.users = list(self.interactions["user_id"])
        self.items = list(self.interactions["item_id"])

        # All possible item IDs for negative sampling
        self.all_item_ids = list(self.item_dict.keys())
        self.num_items = len(self.all_item_ids)

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        """
        Return:
            user_emb:            float tensor [384]
            item_emb:            float tensor [384]
            label:               0 or 1
        For training, we return one positive sample and (num_neg_samples) negative samples.
        We'll pack them into a batch of (num_neg_samples+1) user-item pairs.
        """
        u_id = self.users[idx]
        pos_i_id = self.items[idx]

        user_emb = self.user_dict[u_id]
        pos_item_emb = self.item_dict[pos_i_id]

        # Positive sample
        data = []
        data.append((
            np.array(user_emb, dtype=np.float32),
            pos_item_emb.astype(np.float32),
            1  # label = 1 for positive
        ))

        # Negative samples
        neg_item_ids = negative_sampling(
            positive_item_ids=self.user_pos_item_map[u_id],
            all_item_ids=set(self.items),
            num_neg_samples=self.num_neg_samples
        )

        for neg_i in neg_item_ids:
            neg_item_emb = self.item_dict[neg_i]
            data.append((
                np.array(user_emb, dtype=np.float32),
                neg_item_emb.astype(np.float32),
                0  # label = 0 for negative
            ))

        return data


# -------------------
# Collate Function
# -------------------
def collate_fn(batch):
    """
    Collate function that merges a list of samples to form a mini-batch.
    Each element in batch is itself a list of (user, item, label) of length (num_neg_samples+1).
    We'll flatten them and return as Tensors.
    """
    # Flatten the list of lists
    flat_data = []
    for sample in batch:
        flat_data.extend(sample)

    user_embs = []
    item_embs = []
    labels = []

    for (u, i, lbl) in flat_data:
        user_embs.append(u)
        item_embs.append(i)
        labels.append(lbl)

    user_embs = torch.stack([torch.from_numpy(u) for u in user_embs])
    item_embs = torch.stack([torch.from_numpy(i) for i in item_embs])
    labels = torch.tensor(labels, dtype=torch.long)

    return user_embs, item_embs, labels


# -------------------
# Main Model
# -------------------
class RecSysModel(nn.Module):
    """
    Recommendation model that:
        1) Takes user_emb, and item_emb
        2) Computes the matching score with item_emb
        3) Outputs a probability for classification
    """

    def __init__(self, embed_dim=384, hidden_dim=128, dropout=0.2):
        super().__init__()

        # We'll combine user_rep and item_emb => size=2*embed_dim
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # Output for binary classification: 0 or 1
        )

        self._initialize_weights()

    def forward(self, user_emb, item_emb):

        # 1) Concatenate user_emb with item_emb
        x = torch.cat([user_emb, item_emb], dim=1)  # [B, 2*E]

        # 2) Pass through final fc
        logits = self.fc(x)  # [B, 2]
        return logits

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


# -------------------
# Evaluation Metrics
# -------------------
def accuracy(preds, labels):
    """
    Compute classification accuracy.
    preds: [batch_size, 2] raw logits
    labels: [batch_size] 0/1
    """
    predicted = torch.argmax(preds, dim=1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)


def precision_recall_ndcg(preds, labels):
    """
    This function is a placeholder for computing metrics such as Precision, Recall, NDCG, etc.
    Typically, you would want to evaluate these metrics by ranking items for each user.
    For demonstration, we treat the classification approach in batch:
        - Compute precision, recall, NDCG in a simplified manner.

    preds: [batch_size, 2] => raw logits
    labels: [batch_size] => 0 or 1
    """
    # Convert to predicted classes
    predicted = torch.argmax(preds, dim=1)
    tp = ((predicted == 1) & (labels == 1)).sum().item()
    fp = ((predicted == 1) & (labels == 0)).sum().item()
    fn = ((predicted == 0) & (labels == 1)).sum().item()

    # Precision
    precision = tp / (tp + fp + 1e-8)
    # Recall
    recall = tp / (tp + fn + 1e-8)

    # Dummy NDCG calculation for demonstration.
    # Real NDCG would require a ranking and multiple items per user.
    ndcg = precision  # Not a real formula, just a placeholder

    return precision, recall, ndcg


# -------------------
# Ranking Metrics
# -------------------

def dcg_at_k(relevance_list, k):
    """
    Compute Discounted Cumulative Gain (DCG) for a list of binary relevances,
    truncated at rank k.
    relevance_list: list of 0/1 (1 if item is relevant, else 0)
    k: rank cutoff
    """
    r = relevance_list[:k]
    dcg = 0.0
    for i, rel in enumerate(r):
        # log2(i+2) because i is 0-based
        dcg += rel / math.log2(i + 2)
    return dcg


def ndcg_at_k(relevance_list, k):
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) for binary relevances,
    truncated at rank k.
    """
    # DCG for the actual order
    dcg = dcg_at_k(relevance_list, k)
    # DCG for the ideal ordering (sort by most relevant first)
    ideal = sorted(relevance_list, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranking(
        model,
        user_dict,
        item_dict,
        test_df,
        device,
        top_k_list=[10, 20]
):
    """
    Evaluate ranking metrics (Precision@K, Recall@K, NDCG@K) for K in top_k_list.
    We assume:
        - model is your trained RecSysModel
        - user_s_dict: user_id -> short-term embedding (np.array shape [384])
        - user_l_dict: user_id -> long-term embedding (np.array shape [384])
        - item_dict: item_id -> item embedding (np.array shape [384])
        - test_df: DataFrame with columns ["user_id", "item_id"] for ground truth
        - device: torch.device("cuda" or "cpu")
        - top_k_list: list of K values to evaluate, e.g. [10, 20]

    Returns a dict of results, e.g.:
        {
            10: {"precision": X, "recall": Y, "ndcg": Z},
            20: {"precision": X, "recall": Y, "ndcg": Z},
        }
    """

    model.eval()

    # 1) Build user -> set of ground truth items from test_df
    user_to_test_items = {}
    for row in test_df.itertuples(index=False):
        u_id = getattr(row, "user_id")
        i_id = getattr(row, "item_id")
        if u_id not in user_to_test_items:
            user_to_test_items[u_id] = set()
        user_to_test_items[u_id].add(i_id)

    all_item_ids = list(item_dict.keys())  # all item IDs
    item_embs = [item_dict[i].astype("float32") for i in all_item_ids]  # list of np arrays
    item_embs_torch = torch.tensor(item_embs).to(device)  # shape [num_items, 384]

    # We'll store cumulative sums for each metric at each K
    metrics_accumulator = {k: {"precision": 0.0, "recall": 0.0, "ndcg": 0.0} for k in top_k_list}
    num_users_evaluated = 0

    # 2) For each user in test_df, compute predicted scores for all items, then rank
    for user_id, test_items in user_to_test_items.items():

        # Skip if user embeddings are missing
        if user_id not in user_dict:
            continue

        # Retrieve user embeddings
        # user_np = user_dict[user_id].astype("float32")
        user_np = np.array(user_dict[user_id], dtype=np.float32)

        user_torch = torch.tensor(user_np).unsqueeze(0).to(device)  # [1, 384]

        # We'll replicate user_rep across all items to compute logits in a single pass
        user_rep_expanded = user_torch.repeat(item_embs_torch.size(0), 1)  # [num_items, 384]

        # Concatenate user_rep and item_emb => [num_items, 2*384]
        concat_vec = torch.cat([user_rep_expanded, item_embs_torch], dim=1)  # [num_items, 768]

        # Pass through the final classifier (model.fc) to get logits => [num_items, 2]
        with torch.no_grad():
            logits = model.module.fc(concat_vec) if isinstance(model, torch.nn.DataParallel) else model.fc(concat_vec)

        # Convert logits => predicted probabilities for class=1
        probs = torch.softmax(logits, dim=1)[:, 1]  # shape [num_items]

        # Sort items by descending probability
        sorted_indices = torch.argsort(probs, descending=True)

        # Convert to a Python list of item_ids in ranked order
        ranked_item_ids = [all_item_ids[idx] for idx in sorted_indices.tolist()]

        # 3) For each top-K in [10, 20], compute metrics
        relevant_items = test_items  # ground truth for that user
        num_relevant = len(relevant_items)

        for k in top_k_list:
            top_k_items = ranked_item_ids[:k]

            # Count how many of these top_k_items are in relevant_items
            hits = sum((1 for i in top_k_items if i in relevant_items))

            precision_k = hits / k
            recall_k = hits / num_relevant if num_relevant > 0 else 0.0

            # Build relevance list for NDCG: 1 if item is relevant, else 0
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

    for batch_idx, (user_emb, item_emb, labels) in enumerate(train_loader):
        print(f'Training Batch: {batch_idx}')
        user_emb = user_emb.to(device)
        item_emb = item_emb.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(user_emb, item_emb)

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
        for batch_idx, (user_emb, item_emb, labels) in enumerate(val_loader):
            print(f'Validation Batch: {batch_idx}')
            user_emb = user_emb.to(device)
            item_emb = item_emb.to(device)
            labels = labels.to(device)

            logits = model(user_emb, item_emb)
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
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), self.checkpoint_path)
        self.val_loss_min = val_loss


# -------------------
# Main
# -------------------
def main(seed):
    cfg = Config()

    # -------------------
    # Initialize wandb
    # -------------------
    wandb_key = "YOUR_WANDB_KEY"
    wandb.login(key=wandb_key)
    wandb.init(
        project=cfg.WANDB_PROJECT,
        name=f"{cfg.WANDB_PRE_NAME}_Seed_{seed}",
        entity=cfg.WANDB_ENTITY,
        config={
            "batch_size": cfg.BATCH_SIZE,
            "num_neg_samples": cfg.NUM_NEG_SAMPLES,
            "lr": cfg.LR,
            "weight_decay": cfg.WEIGHT_DECAY,
            "epochs": cfg.EPOCHS,
            "early_stop_patience": cfg.EARLY_STOP_PATIENCE,
            "embedding_dim": cfg.EMBEDDING_DIM,
            "hidden_dim": cfg.HIDDEN_DIM,
            "dropout": cfg.DROPOUT
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------
    # Data Loading
    # -------------------
    print("Loading data...")
    item_df = load_pickle(cfg.ITEM_EMBEDDINGS_PATH)
    user_df = load_pickle(cfg.USER_EMBEDDINGS_PATH)

    train_df = load_interactions_csv(cfg.TRAIN_PATH)
    val_df = load_interactions_csv(cfg.VAL_PATH)
    test_df = load_interactions_csv(cfg.TEST_PATH)

    # Build dictionaries: item_id -> embedding, user_id -> embedding
    # In practice, ensure item_id and user_id are integer-encoded or handle them carefully.
    item_dict = {int(row["item_id"]): row["description"] for _, row in item_df.iterrows()}
    user_dict = {int(row["user_id"]): row["profile"] for _, row in user_df.iterrows()}

    # -------------------
    # Datasets & Loaders
    # -------------------
    train_dataset = RecSysDataset(
        interactions_df=train_df,
        user_dict=user_dict,
        item_dict=item_dict,
        num_neg_samples=cfg.NUM_NEG_SAMPLES
    )
    val_dataset = RecSysDataset(
        interactions_df=val_df,
        user_dict=user_dict,
        item_dict=item_dict,
        num_neg_samples=cfg.NUM_NEG_SAMPLES  # can be changed or same for val
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

    # -------------------
    # Model, Optimizer, Criterion
    # -------------------
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

    # -------------------
    # Early Stopping
    # -------------------
    early_stopper = EarlyStopping(
        patience=cfg.EARLY_STOP_PATIENCE,
        verbose=True,
        checkpoint_path=f"best_model_{cfg.WANDB_PRE_NAME}_Seed_{seed}.pt"
    )

    # -------------------
    # Training Loop
    # -------------------

    for epoch in range(cfg.EPOCHS):
        print(f"Processing Epoch (Only-Short/Long) {epoch + 1} ...")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_precision, val_recall, val_ndcg = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch + 1}/{cfg.EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f} | Val NDCG: {val_ndcg:.4f}")

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_ndcg": val_ndcg
        })

        # Early stopping check
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    # -------------------
    # Load best model
    # -------------------
    print("Loading best model weights...")
    model.load_state_dict(torch.load(f"best_model_{cfg.WANDB_PRE_NAME}_Seed_{seed}.pt"))

    # -------------------
    # Testing (batch-based evaluation)
    # -------------------
    test_dataset = RecSysDataset(
        interactions_df=test_df,
        user_dict=user_dict,
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

    wandb.log({
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_ndcg": test_ndcg
    })

    # -------------------
    # Ranking-Based Testing
    # -------------------
    print("Starting Ranking-Based Testing...")
    # (1) We assume you already have user_s_dict, user_l_dict, item_dict, and test_df
    #     loaded in memory. These are the same dictionaries/dataframes we used in
    #     training and test dataset creation.

    # (2) Evaluate top-10 and top-20 rankings
    top_k_list = [10, 20]
    ranking_results = evaluate_ranking(
        model=model,
        user_dict=user_dict,
        item_dict=item_dict,
        test_df=test_df,  # must contain columns ['user_id', 'item_id']
        device=device,
        top_k_list=top_k_list
    )

    # (3) Print the results
    for k in top_k_list:
        print(f"[Ranking] Top-{k} => "
              f"Precision: {ranking_results[k]['precision']:.4f}, "
              f"Recall: {ranking_results[k]['recall']:.4f}, "
              f"NDCG: {ranking_results[k]['ndcg']:.4f}")

    # Optionally, log these to wandb as well:
    for k in top_k_list:
        wandb.log({
            f"precision@{k}": ranking_results[k]['precision'],
            f"recall@{k}": ranking_results[k]['recall'],
            f"ndcg@{k}": ranking_results[k]['ndcg']
        })

    wandb.finish()

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
    print(f'Proposed-{DATASET}-Only-ShortTerm-NoAtt')
    args = parse_arguments()
    set_seed(args.seed)
    start_time = time.time()
    main(args.seed)
    print(f'Execution Time (Baseline): {time.time() - start_time}.')
    print(f'Proposed-{DATASET}-Only-ShortTerm-NoAtt')
