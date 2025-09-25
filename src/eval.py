# eval_analysis.py
import torch
import argparse
import yaml
import numpy as np
import time
from collections import Counter
from data import load_corpus, build_vocab
from src.model import CBOW

# -------------------------------
# 1. Config 불러오기
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/brown.yaml")
parser.add_argument("--checkpoint", type=str, required=True)
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 2. 데이터 및 vocab
# -------------------------------
text = load_corpus(config["dataset"])
vocab, word2idx, idx2word = build_vocab(text)
vocab_size = len(vocab)
embedding_dim = config["embedding_dim"]

# -------------------------------
# 3. 모델 불러오기
# -------------------------------
model = CBOW(vocab_size, embedding_dim).to(device)
checkpoint = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Checkpoint '{args.checkpoint}' 로드 완료")

# -------------------------------
# 4. 헬퍼 함수
# -------------------------------
def get_embedding(word):
    if word not in word2idx:
        return None
    idx = torch.tensor([word2idx[word]]).to(device)
    return model.embeddings(idx).detach().cpu().numpy()

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def most_similar(word, topk=5):
    emb = get_embedding(word)
    if emb is None:
        return []
    sims = {}
    for w in vocab:
        if w == word:
            continue
        sims[w] = cosine_similarity(emb, get_embedding(w))
    return sorted(sims.items(), key=lambda x: x[1], reverse=True)[:topk]

def analogy(word_a, word_b, word_c, topk=1):
    emb_a, emb_b, emb_c = get_embedding(word_a), get_embedding(word_b), get_embedding(word_c)
    if emb_a is None or emb_b is None or emb_c is None:
        return []
    target_vec = emb_b - emb_a + emb_c
    sims = {}
    for w in vocab:
        if w in [word_a, word_b, word_c]:
            continue
        sims[w] = cosine_similarity(target_vec, get_embedding(w))
    return sorted(sims.items(), key=lambda x: x[1], reverse=True)[:topk]

# -------------------------------
# 5. 평가 함수
# -------------------------------
def evaluate_analogy(analogy_set):
    total = 0
    correct = 0
    failures = []
    for a, b, c, d_true in analogy_set:
        total += 1
        preds = analogy(a, b, c, topk=5)
        pred_words = [w for w, _ in preds]
        if d_true in pred_words:
            correct += 1
        else:
            failures.append((a, b, c, d_true, pred_words))
    accuracy = correct / total
    print(f"Analogy Accuracy: {accuracy:.4f}")
    return failures

# -------------------------------
# 6. GPU 사용량/시간 측정
# -------------------------------
def measure_resources():
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    # 단순 전체 vocab 임베딩 조회 (샘플 연산)
    _ = [get_embedding(w) for w in vocab[:5000]]
    elapsed = time.time() - start_time
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)  # MB
    print(f"Elapsed time: {elapsed:.2f}s, Peak GPU memory: {peak_mem:.2f} MB")

# -------------------------------
# 7. 테스트 실행 예시
# -------------------------------
if __name__ == "__main__":
    # 단어 임베딩 확인
    test_words = ["money", "bank", "river"]
    for w in test_words:
        vec = get_embedding(w)
        if vec is not None:
            print(f"\n'{w}' 임베딩 벡터:", vec)

    # 가장 유사한 단어 확인
    for w in ["money", "bank"]:
        sims = most_similar(w, topk=5)
        print(f"\n'{w}'와 가장 유사한 단어 top-5:")
        for sw, sim in sims:
            print(f"{sw}: {sim:.4f}")

    # Analogy 평가
    # 예시 데이터: [(a,b,c,d_true), ...]
    analogy_testset = [
        ("man", "king", "woman", "queen"),
        ("paris", "france", "berlin", "germany"),
        ("walking", "walked", "swimming", "swam")
    ]
    failures = evaluate_analogy(analogy_testset)

    print("\n실패 사례 (최대 20개):")
    for i, (a, b, c, d_true, preds) in enumerate(failures[:20]):
        print(f"{i+1}. ({a}→{b}) == ({c}→?) | True: {d_true}, Predicted: {preds}")

    # GPU 시간/메모리 측정
    measure_resources()
