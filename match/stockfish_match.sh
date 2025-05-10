#!/bin/bash
games="${1}"
player1="${2}"
player2="${3}"
book="${4}"

rounds=$((games / 2))

rm -f /dev/shm/games-cutechess.pgn

/home/user/chess/cutechess/cutechess-cli_LC0 \
  -pgnout /dev/shm/games-cutechess.pgn \
  -tournament gauntlet \
  -concurrency 2 \
  -rounds $rounds \
  -games 2 \
  -openings file=$book \
     format=pgn order=sequential policy=round \
  -resign movecount=3 score=450 \
  -draw movenumber=1 movecount=5 score=5 \
  -repeat \
  -tb /home/user/chess/tablebases \
  -engine \
     cmd=/home/user/chess/engines/stockfish/stockfish-dev \
     name=stockfish \
     option.Threads=1 \
     option.Hash=2 \
     option."Move Overhead"=0 \
     option.SyzygyProbeDepth=10 \
     nodes=800 \
  -engine \
     cmd=/home/user/chess/engines/lc0/lc0-dev_Ergodice-master_692b821 \
     name=lc0_test1 \
     option.WeightsFile=$player1 \
     nodes=32 \
     option.Backend=cuda-fp16 \
     option.Threads=1 \
     option.TaskWorkers=0 \
     option.MoveOverheadMs=0 \
     option.MinibatchSize=32 \
     option.CPuct=1.745 \
     option.FpuValue=0.330 \
     option.PolicyTemperature=1.359 \
  -engine \
     cmd=/home/user/chess/engines/lc0/lc0-dev_Ergodice-master_692b821 \
     name=lc0_test2 \
     option.WeightsFile=$player2 \
     nodes=32 \
     option.Backend=cuda-fp16 \
     option.Threads=1 \
     option.TaskWorkers=0 \
     option.MoveOverheadMs=0 \
     option.MinibatchSize=32 \
     option.CPuct=1.745 \
     option.FpuValue=0.330 \
     option.PolicyTemperature=1.359 \
  -each \
     proto=uci \
     timemargin=1000 \
     tc=inf \
     option.SyzygyPath=/home/user/chess/tablebases 2>&1

/home/user/chess/ordo/ordo -q -n 1 -z 200.24 -s 100 -U 0,1,2,6,4,10 -N 1 -a 0 -A lc0_test1 /dev/shm/games-cutechess.pgn | grep -A 3 '# ENGINE'
