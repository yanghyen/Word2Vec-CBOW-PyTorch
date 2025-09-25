#!/bin/bash
# 평가 스크립트 실행
# 사용법: ./eval.sh configs/subsample-on_window-2_epoch-5.yaml checkpoints/best_model.pth data/analogy_testset.csv

CONFIG=$1
CHECKPOINT=$2
ANALOGY_CSV=$3

if [ -z "$CONFIG" ] || [ -z "$CHECKPOINT" ] || [ -z "$ANALOGY_CSV" ]; then
    echo "Usage: ./eval.sh <config.yaml> <checkpoint.pth> <analogy_testset.csv>"
    exit 1
fi

python eval_analysis_auto.py --config $CONFIG --checkpoint $CHECKPOINT --analogy_csv $ANALOGY_CSV
