from data import load_corpus, build_vocab, get_dataloader, subsample_text
from model import CBOW

import argparse
import yaml
import random
import os 

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# python train.py --config configs/brown.yaml
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/subsample-on_window-2_epoch-50.yaml")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # gpu(cuda)로 돌릴 때도 연산 결과 일정하도록 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(config["seed"])

text = load_corpus(config["dataset"])

if config["subsample"] == "on":
    text = subsample_text(text, t=1e-5)

vocab, word2idx, idx2word = build_vocab(text)
dataloader = get_dataloader(config, word2idx)

subsample = config["subsample"]
window = config["window_size"]
embedding_dim = config["embedding_dim"]
vocab_size = len(vocab)
model = CBOW(vocab_size, embedding_dim)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config["lr"])

num_epoch = config["epochs"]
checkpoint_dir = "runs/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

for epoch in range(1, num_epoch + 1):
    total_loss = 0
    model.train()

    for context_idxs, target in dataloader:
        context_idxs = context_idxs.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(context_idxs)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} / {num_epoch}, Loss: {total_loss:.4f}")
    
    if epoch % 10 == 0 or epoch == num_epoch:
        checkpoint_name = f"subsample-{subsample}_window-{window}_epoch-{epoch}.pth"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": total_loss,
            "config": config,
        }, checkpoint_path)
