#!/bin/bash
#
# Self-Play Match Script for LC0 Chess Engine
#
# This script runs a self-play match between two neural network models using the LC0 (Leela Chess Zero)
# chess engine. It is called by train.py to evaluate network performance during SPSA optimization.
#
# Usage:
#   ./selfplay_match.sh <games> <player1_weights> <player2_weights> <book>
#
# Arguments:
#   games: Number of games to play (must be even).
#   player1_weights: Path to the first network's weights file (e.g., networks/744706.pb.gz).
#   player2_weights: Path to the second network's weights file.
#   book: Path to the PGN book file containing opening positions.
#
# Dependencies:
#   - LC0 engine binary (https://github.com/LeelaChessZero/lc0)
#   - Syzygy tablebases (optional, for endgame evaluation)
#
# Output:
#   Prints match results, including Elo difference, draw rate, and LOS (likelihood of superiority).
#   The final line is parsed by train.py (grep "final" for "Elo:", "P1:", "LOS:").
#
# Environment:
#   - Adjust LC0_PATH and TABLEBASES_PATH as needed for your setup.
#   - Also adjust: cpuct, fpu-value-at-root, fpu-value, minibatch-size, policy-softmax-temp, visits.
#

# Configuration
LC0_PATH="/home/user/chess/engines/lc0/lc0-dev_master_05172b6"
TABLEBASES_PATH="/home/user/chess/tablebases"

# Validate arguments
if [ "$#" -ne 4 ]; then
    echo "Error: Incorrect number of arguments."
    echo "Usage: $0 <games> <player1_weights> <player2_weights> <book>"
    exit 1
fi

games="$1"
player1="$2"
player2="$3"
book="$4"

# Validate input files
if [ ! -f "$player1" ]; then
    echo "Error: Player 1 weights file not found: $player1"
    exit 1
fi
if [ ! -f "$player2" ]; then
    echo "Error: Player 2 weights file not found: $player2"
    exit 1
fi
if [ ! -f "$book" ]; then
    echo "Error: PGN book file not found: $book"
    exit 1
fi
if [ ! -x "$LC0_PATH" ]; then
    echo "Error: LC0 executable not found or not executable: $LC0_PATH"
    exit 1
fi

# Ensure games is a positive even number
if ! [[ "$games" =~ ^[0-9]+$ ]] || [ "$games" -le 0 ] || [ $((games % 2)) -ne 0 ]; then
    echo "Error: Number of games must be a positive even integer, got: $games"
    exit 1
fi

# Run LC0 self-play match
"$LC0_PATH" selfplay \
  --backend-opts=cuda-fp16 \
  --backend=multiplexing \
  --cpuct=1.745 \
  --fpu-strategy-at-root=reduction \
  --fpu-strategy=reduction \
  --fpu-value-at-root=0.330 \
  --fpu-value=0.330 \
  --games="$games" \
  --max-collision-events=917 \
  --max-collision-visits=80000 \
  --minibatch-size=32 \
  --mirror-openings=true \
  --moves-left-constant-factor=0.0 \
  --moves-left-max-effect=0.0345 \
  --moves-left-quadratic-factor=-0.65 \
  --moves-left-scaled-factor=1.65 \
  --moves-left-slope=0.0027 \
  --moves-left-threshold=0.8 \
  --no-share-trees \
  --noise-alpha=0 \
  --noise-epsilon=0 \
  --openings-mode=shuffled \
  --openings-pgn="$book" \
  --out-of-order-eval=true \
  --parallelism=16 \
  --player1.weights="$player1" \
  --player2.weights="$player2" \
  --policy-softmax-temp=1.359 \
  --resign-percentage=2.0 \
  --root-has-own-cpuct-params=false \
  --smart-pruning-factor=1.33 \
  --sticky-endgames=true \
  --syzygy-paths="$TABLEBASES_PATH" \
  --task-workers=0 \
  --temperature=0 \
  --threads=1 \
  --visits=800 2>&1 | grep final