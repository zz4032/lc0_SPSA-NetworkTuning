# Neural Network Training with SPSA

This repository contains a Python script (`tune.py`) for training a neural network using Simultaneous Perturbation Stochastic Approximation (SPSA), designed for optimizing chess engine networks. It supports self-play matches, PGN position book management, and detailed logging. Network layer structures are defined in `net_structure.py`, and self-play matches are executed via `selfplay_match.sh` using the LC0 chess engine.

## Features
- SPSA-based optimization for neural network weights
- Self-play matches to evaluate network performance using LC0
- PGN book parsing and shuffling for game positions
- Comprehensive logging of training progress
- Configurable network structures and training parameters

## Prerequisites
- Python 3.6+
- NumPy (`pip install numpy`)
- LC0 Chess Engine (Leela Chess Zero, see [LC0 GitHub](https://github.com/LeelaChessZero/lc0))
  - Requires a CUDA-enabled GPU for the `cuda-fp16` backend
- Syzygy Tablebases (optional, for endgame evaluation; see [Syzygy](https://syzygy-tables.info/))
- Custom Modules: `net` and `proto.net_pb2`
- PGN Book Files (e.g., `book_4moves_2023_cp-5to+5_95710pos.pgn`)
- Bash Environment (for `selfplay_match.sh`)

## Hardware Requirements
- **GPU**: CUDA-enabled NVIDIA GPU with enough VRAM (e.g., GTX 1060 or better)
- **Disk**: 10GB+ for networks, logs, and tablebases

### Custom Modules
The `net` and `proto.net_pb2` modules are required for network serialization and are part of Leela Chess Zero (LC0), licensed under GPL-3.0. To generate `proto/chunk_pb2.py` and `proto/net_pb2.py` for your system environment:
1. Clone the LC0 repository: `git clone https://github.com/LeelaChessZero/lczero-training.git`.
2. Run `./init.sh`.
3. Copy `proto/chunk_pb2.py` and `proto/net_pb2.py` to the project root.
Alternatively, these files were generated with protobuf version 3.20.3 and are included in this repository under the GPL-3.0 license. If you encounter errors due to protobuf version mismatch, regenerate them from the LC0 source.


## Installation
1. Clone the repository:
    ```
    git clone https://github.com/zz4032/lc0_SPSA-NetworkTuning.git
    cd lc0_SPSA-NetworkTuning
    ```
2. Install Python dependencies:
    ```
    pip install numpy
    ```
3. Install LC0:
   - Follow instructions at [LC0 GitHub](https://github.com/LeelaChessZero/lc0) to build or download the LC0 binary.
   - Update `LC0_PATH` in `match/selfplay_match.sh` to point to your LC0 executable.
4. (Optional) Set up Syzygy tablebases:
   - Download tablebases and update `TABLEBASES_PATH` in `match/selfplay_match.sh`.
5. Download `books.zip` (~6 MB) from [Google Drive](https://drive.google.com/file/d/1Fd2ugdm2BHZjzGXm1V_IpEqJSbDQjedM).
5. Place required files:
   - PGN books in `books/`
   - Network files in `networks/`
   - Match scripts in `match/`

## Usage
Run the training script with required arguments:
```bash
    python3 tune.py --config <CONFIG_PATH>
Example:
    ```
    python3 tune.py --config configs/config_T74.json
    ```
The script will:
- Load the network structure from `net_structure.py` (default: `T74`)
- Initialize training from a base network (e.g., `744706.pb.gz`)
- Perform SPSA optimization over specified iterations
- Execute self-play matches via `selfplay_match.sh`
- Log progress to `logs/train_run01.log`
- Save evaluation results to `match_results_run01.txt`

## Configuration
### tune.py
Key parameters in `tune.py`:
- `ITERATIONS`: Number of training iterations (default: 50)
- `MATCH_GAMES`: Games per training match (default: 184)
- `MATCH_ITERATIONS_GAMES`: Games for evaluation matches (default: 1000)
- `ROUNDS`: SPSA rounds per iteration (default: 16)
- `ADJ`: Factor to pull weights toward initial network, reducing SPSA tuning noise (default: 0.005)
- `R_END`: Final perturbation size (default: 0.0085)
- `A`: SPSA scaling factor (default: 0.0)
- `ALPHA`: SPSA learning rate decay (default: 0.0)
- `GAMMA`: SPSA perturbation decay (default: 0.0)
- `TRAINING_BOOK`: PGN book file for training matches
- `EVAL_BOOK`: PGN book file for evaluation matches
- `CURRENT_BEST`: Base network ID
- `NAME`: Training run identifier (default: `run01`)

### net_structure.py
Network structures:
- `T74`: Residual network with 10 blocks
- `T74_policy`: Policy-focused subset of T74
- `T79`: Residual network with 15 blocks and SE layers
To use a different structure, modify `total_data` in `net_structure.py` or use the `--structure` argument.

### selfplay_match.sh
Configurable paths:
- `LC0_PATH`: Path to the LC0 executable
- `TABLEBASES_PATH`: Path to Syzygy tablebases
LC0 parameters (e.g., `cpuct`, `visits`) are tuned for the training setup. Adjust with caution.

### Example Configuration
Create a `configs/config_T74.json` file with the following content:
```
{
    "name": "run01",
    "base_network": "744706",
    "structure": "T74",
    "match_script": "selfplay_match",
    "iterations": 50,
    "rounds": 16,
    "match_games": 184,
    "r_end": 0.0085,
    "adj": 0.005,
    "elo_average": 18.0,
    "history_interval": 1,
    "match_book": "book_4moves_2023_cp-5to+5_95710pos.pgn",
    "eval_book": "book_4moves_2023_cp-5to+5_95710pos_match_2500pos.pgn",
    "eval_iterations_games": 5000,
    "eval_points": [10, 20, 30, 40, 50],
    "A": 0.15,
    "alpha": 0.602,
    "gamma": 0.101
}
```
Place this file in the `configs/` directory and reference it with `--config configs/config_T74.json`.

## Data Files
- **PGN Books**: Place PGN files (e.g., `book_4moves_2023_cp-5to+5_95710pos.pgn`) in `books/`. These files are licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/). You may share and adapt them for non-commercial purposes, provided you credit zz4032 and license derivative works under the same terms. A sample file, `sample_opening_book.pgn`, is included. See [LICENSE_DATA.md](LICENSE_DATA.md) for details.
- **Network Files**: Place network files (e.g., `744706.pb.gz`) in `networks/`. These are outputs of LC0 and can be obtained here: http://training.lczero.org/networks/2 (search for `744706`).

### Running the Script
1. Ensure the directory structure is set up.
```
your-repo/
├── books/
│   ├── book_4moves_2023_cp-5to+5_95710pos.pgn
│   ├── book_4moves_2023_cp-5to+5_95710pos_match_2500pos.pgn
├── configs/
│   ├── config_T74.json
├── match/
│   ├── selfplay_match.sh
├── networks/
│   ├── 744706.pb.gz
├── proto/
│   ├── chunk_pb2.py
│   ├── net_pb2.py
├── net.py
├── net_structure.py
├── tune.py
```
2. Confirm that the layers of the network to be tuned are defined correctly in `net_structrure.py`.
Analyze network:
```
python3 tune.py --network-structure 744706
```
This generates a table of layers (ID, name, parameter count, standard deviation, max/min values) and saves it to 744706_structure.txt. Use this to define layers in net_structure.py (e.g., T74 structure with (layer_id, c_end) tuples).
`c_end` should be set to produce an average of absolute match results of 20 Elo for the given layer.
`*.json` config contains the setting to be used (e.g. `"structure": "T74"`).
3. Run the training script:
```
python3 tune.py --config configs/config_T74.json
```

## Outputs
- Logs: Training progress in `logs/train_run01.log`, including Elo, draw rates, and LOS for matches.
- Match Results: Evaluation match results in `match_results_run01.txt`, with columns:
  - Iteration
  - Match identifier
  - Elo difference
  - Draw rate
  - LOS (Likelihood of Superiority, 0 to 1)
  - Average absolute deviation
  - L2 norm
- Networks: Checkpoints saved in `networks/` at iterations specified in `eval_points` (e.g., `744706_run01_it0010.pb.gz` for iteration 10).

## Key Functions
- `run_match(script, net1, net2, games, book)`:
  - Runs a match between two networks using the specified script.
  - Returns `(elo, draw_rate, los)` for `selfplay_match` and `policy_match`, where:
    - `elo`: Elo difference (float)
    - `draw_rate`: Fraction of games drawn (float)
    - `los`: Likelihood of superiority (float, 0 to 1)
  - Returns `(elo, draw_rate, None)` for `stockfish_match`, as LOS is not provided.

## Acknowledgments
This project builds on the [Leela Chess Zero (LC0)](https://github.com/LeelaChessZero) chess engine and its neural network architecture.

## License
This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## GPL-3.0 and CC BY-NC-SA 4.0 Compliance
- **Code**: Licensed under GPL-3.0. Derivative works must be GPL-3.0 licensed and include source code (`net.py`, `chunk_pb2.py`, `net_pb2.py`). See [LICENSE](LICENSE).
- **PGN Books and Network Files**: Licensed under CC BY-NC-SA 4.0. Derivative works must be non-commercial, credited to zz4032, and licensed under CC BY-NC-SA 4.0. See [LICENSE_DATA.md](LICENSE_DATA.md).
When distributing the project, ensure compliance with both licenses.

## Contact
For questions about the PGN files, network files, or licensing, contact zz4032 via GitHub issues or [your.email@example.com].