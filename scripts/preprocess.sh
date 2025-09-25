#!/bin/bash
# 데이터 다운로드, 토큰화, vocab 생성

# 예시: Brown Corpus 다운로드 (NLTK)
python - <<END
import nltk
nltk.download('brown')
END

# 데이터 로드 및 vocab 저장
python - <<END
from data import load_corpus, build_vocab
import yaml

# config 파일 불러오기
with open("configs/brown.yaml", "r") as f:
    config = yaml.safe_load(f)

text = load_corpus(config["dataset"])
vocab, word2idx, idx2word = build_vocab(text)

# vocab 저장
import pickle
with open("data/vocab.pkl", "wb") as f:
    pickle.dump({"vocab": vocab, "word2idx": word2idx, "idx2word": idx2word}, f)

print("Preprocessing 완료: vocab.pkl 생성됨")
END
