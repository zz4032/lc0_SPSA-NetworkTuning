#!/usr/bin/env python3

# Copyright (c) 2025 zz4032
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
# See LICENSE file for details.

"""
Neural Network Training Script using SPSA Optimization

This script trains a neural network using Simultaneous Perturbation Stochastic Approximation (SPSA)
for optimizing network weights, primarily for a chess engine. It supports self-play matches, position
book management, and logging of training progress.

Dependencies:
- Python 3.6+
- NumPy
- Custom `net` module and `proto.net_pb2` for network serialization
- Bash scripts for match execution (e.g., selfplay_match.sh)
- PGN book files for chess positions

Usage:
    python3 tune.py [--config CONFIG_PATH]
"""

import random
import numpy as np
import subprocess
import os
import time
import copy
import math
import shutil
import argparse
import json
import sys
from datetime import datetime, timedelta
from tabulate import tabulate
import logging
from typing import List, Tuple, Optional, Dict

from net import Net
import proto.net_pb2 as pb

# --- Configuration Constants ---

# Directory paths
PATH_CURRENT = os.getcwd()
PATH_BOOKS = os.path.join(PATH_CURRENT, "books")
PATH_MATCH = os.path.join(PATH_CURRENT, "match")
PATH_NETWORKS = os.path.join(PATH_CURRENT, "networks")
PATH_LOGS = os.path.join(PATH_CURRENT, "logs")

# --- Configuration Loading ---

def parse_config_arg():
    parser = argparse.ArgumentParser(description="Neural Network Training with SPSA")
    parser.add_argument("--config", default="config.json", help="Path to configuration file")
    parser.add_argument("--network-structure", default=None, help="Network ID to print structure (e.g., 744706)")
    return parser.parse_args()

def load_config(config_path: str) -> Dict:
    """Load configuration from specified path, failing if the file is missing."""
    try:
        with open(config_path) as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}")
        exit(1)

def validate_config(config: Dict) -> None:
    """Validate that config contains all required keys and valid values."""
    required_keys = [
        "name", "base_network", "structure", "match_script", "iterations",
        "rounds", "match_games", "r_end", "adj", "elo_average", "history_interval",
        "match_book", "eval_book", "eval_iterations_games", "eval_points",
        "A", "alpha", "gamma"
    ]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        logging.error(f"Missing required keys in config.json: {missing_keys}")
        exit(1)

# --- Logging ---

def setup_logging(run_name: str, log_level: str = "INFO") -> None:
    """Configure logging to file and console."""
    os.makedirs(PATH_LOGS, exist_ok=True)
    log_file = os.path.join(PATH_LOGS, f"train_{run_name}.log")

    # Clear any existing handlers to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

# --- Network Structure Printing ---

def get_network_layers_with_names(net_proto: pb.Net) -> Tuple[List[pb.Weights.Layer], List[List[str]]]:
    """
    Extract all layers and their hierarchical names from a network protobuf.

    Returns:
        Tuple of (list of Layer objects, list of layer name paths)
    """
    layers = []
    layer_names = []
    current_layer = []
    temp = []

    def recursive_extract(obj):
        nonlocal layers, layer_names, current_layer, temp
        if obj.DESCRIPTOR.name == "Layer":
            if len(current_layer) == 1:
                layer_names.append(temp[:] + current_layer[:])
                temp = temp[:] + current_layer[:]
                if temp:
                    del temp[-1]  # Remove the last element to backtrack
            else:
                # For nested layers, use current_layer and update temp
                layer_names.append(current_layer[:])
                temp = current_layer[:-1] if current_layer else []
            layers.append(obj)
            if current_layer:
                del current_layer[-1]  # Backtrack
        for field_desc, value in obj.ListFields():
            field_name = field_desc.name
            if field_desc.type == field_desc.TYPE_MESSAGE:
                if field_desc.label == field_desc.LABEL_REPEATED:
                    for item in value:
                        if field_name not in ['encoder', 'min_version', 'format', 'network_format', 'training_params']:
                            current_layer.append(field_name)
                        recursive_extract(item)
                        if current_layer:
                            del current_layer[-1]
                else:
                    if field_name not in ['encoder', 'min_version', 'format', 'network_format', 'training_params']:
                        current_layer.append(field_name)
                    recursive_extract(value)
                    if current_layer:
                        del current_layer[-1]

    recursive_extract(net_proto)
    return layers, layer_names

def print_network_structure(network_id: str, path_networks: str = PATH_NETWORKS) -> None:
    """
    Print and save the structure of the specified network.

    Args:
        network_id: ID of the network (e.g., '744706')
        path_networks: Directory containing network files
    """
    net_path = os.path.join(path_networks, f"{network_id}.pb.gz")
    if not os.path.exists(net_path):
        logging.error(f"Network file not found: {net_path}")
        sys.exit(1)

    net = Net()
    net.parse_proto(net_path)
    layers, layer_names = get_network_layers_with_names(net.pb)
    np_weights = [net.denorm_layer_v2(layer) for layer in layers]

    output_table = [["layer id", "layer name", "parameters", "st-dev", "max", "min"]]
    for idx, (weights, names) in enumerate(zip(np_weights, layer_names)):
        layer_name = ".".join(names) if names else "root"
        output_table.append([
            idx,
            layer_name,
            weights.size,
            f"{np.std(weights):.8f}",
            f"{np.max(weights):.8f}",
            f"{np.min(weights):.8f}"
        ])

    # Print to console
    print(tabulate(output_table, headers="firstrow", tablefmt="simple", floatfmt=".8f"))

    # Save to file
    output_file = f"{network_id}_structure.txt"
    with open(output_file, "w") as f:
        f.write(tabulate(output_table, headers="firstrow", tablefmt="simple", floatfmt=".8f"))
    logging.info(f"Network structure saved to {output_file}")

# --- Book ---

available_positions = None  # Pool of PGN positions
used_position_indices = set()  # Tracks used position indices

def reset_position_pool(positions: List[List[str]]) -> None:
    """Reset the global position pool and clear used indices."""
    global available_positions, used_position_indices
    available_positions = positions.copy()
    used_position_indices.clear()
    logging.info("Position pool reset")

def read_pgn_book(book_path: str) -> Optional[List[List[str]]]:
    """
    Read a PGN book file and parse into a list of positions.

    Args:
        book_path: Path to the PGN file.

    Returns:
        List of positions, where each position is a list of PGN lines, or None if file is not found.
    """
    try:
        with open(book_path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        logging.error(f"Book file not found: {book_path}")
        return None

    positions = []
    current_pos = []
    header_count = 0
    for line in lines:
        current_pos.append(line)
        if line == "\n":
            header_count += 1
            if header_count == 2:
                positions.append(current_pos)
                current_pos = []
                header_count = 0
    if current_pos:
        positions.append(current_pos)
    return positions

def write_shuffled_book(
    positions: List[List[str]],
    games: int,
    book_path: str,
    run_name: str,
    iteration: int,
    layer_id: int,
    round_idx: int
) -> str:
    """
    Create a temporary PGN book with a shuffled subset of positions.

    Args:
        positions: List of PGN positions.
        games: Number of games to select (divided by 2 for actual positions).
        book_path: Base path for the book file.
        run_name: Training run identifier.
        iteration: Current iteration number.
        layer_id: Layer index.
        round_idx: Round index.

    Returns:
        Path to the temporary PGN file.
    """
    global available_positions, used_position_indices
    if available_positions is None:
        reset_position_pool(positions)

    games_needed = games // 2
    temp_path = f"{book_path.rsplit('.pgn', 1)[0]}_temp_{run_name}.pgn"
    random.seed(42 + iteration * 100000 + layer_id * 100 + round_idx)
    remaining = len(available_positions) - len(used_position_indices)

    available_indices = [i for i in range(len(available_positions)) if i not in used_position_indices]
    if remaining < games_needed:
        # Use remaining positions and reset pool if needed
        selected_indices = available_indices[:remaining]
        with open(temp_path, "w") as f:
            for idx in selected_indices:
                f.writelines(available_positions[idx])
        used_position_indices.update(selected_indices)

        reset_position_pool(positions)
        additional_needed = games_needed - remaining
        available_indices = list(range(len(available_positions)))
        random.shuffle(available_indices)
        additional_indices = available_indices[:additional_needed]
        with open(temp_path, "a") as f:
            for idx in additional_indices:
                f.writelines(available_positions[idx])
        used_position_indices.update(additional_indices)
        logging.debug(f"Used {remaining} remaining, reset, added {additional_needed} for total {games_needed}")
    else:
        random.shuffle(available_indices)
        selected_indices = available_indices[:games_needed]
        with open(temp_path, "w") as f:
            for idx in selected_indices:
                f.writelines(available_positions[idx])
        used_position_indices.update(selected_indices)
        logging.debug(f"Selected {games_needed} from {remaining} available")

    return temp_path

# --- Match ---

def run_match(
    script: str,
    net1: str,
    net2: str,
    games: int,
    book: str
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Run a match between two networks and parse the results.

    Args:
        script: Name of the match script (e.g., selfplay_match).
        net1: ID of the first network.
        net2: ID of the second network.
        games: Number of games to play (rounded to even).
        book: Path to the PGN book file.

    Returns:
        Tuple of (Elo difference, draw rate, LOS probability), or (None, None, None) on failure.
    """
    games = (games // 2) * 2
    command = (
        f"{os.path.join(PATH_MATCH, f'{script}.sh')} {games} "
        f"{os.path.join(PATH_NETWORKS, f'{net1}.pb.gz')} "
        f"{os.path.join(PATH_NETWORKS, f'{net2}.pb.gz')} "
        f"{os.path.join(PATH_BOOKS, book)}"
    )

    try:
        output = subprocess.check_output(command, shell=True, text=True).strip()
        if not output:
            logging.error(f"Empty output: {command}")
            return None, None, None

        if script in ["selfplay_match", "policy_match"]:
            parts = output.splitlines()[-1].split()
            if "Elo:" not in parts or "P1:" not in parts or "LOS:" not in parts:
                logging.error(f"Invalid output: {parts}")
                return None, None, None
            elo = float(parts[parts.index("Elo:") + 1])
            p1_stats = parts[parts.index("P1:") + 1:parts.index("Win:")]
            total = sum(int(stat.lstrip("+-=")) for stat in p1_stats)
            draw_rate = int(p1_stats[2].lstrip("=")) / total if total > 0 else 0.0
            los = float(parts[parts.index("LOS:") + 1].rstrip("%")) / 100
            return elo, draw_rate, los

        elif script == "stockfish_match":
            lines = output.splitlines()
            elo, draw_rate, los = None, None, None
            table_start = -1
            for i, line in enumerate(lines):
                if "Finished match" in line:
                    table_start = i + 1
                    break
            if table_start == -1:
                logging.error(f"No 'Finished match' marker: {output}")
                return None, None, None

            for line in lines[table_start:]:
                parts = line.split()
                if len(parts) < 7:
                    continue
                engine = parts[1]
                if engine == "lc0_test2":
                    elo = float(parts[3])
                elif engine == "stockfish":
                    draw_rate = float(parts[7]) / 100

            if elo is None or draw_rate is None:
                logging.error(f"Missing fields: {output}")
                return None, None, None
            return elo, draw_rate, None

        logging.error(f"Unsupported script: {script}")
        return None, None, None
    except (subprocess.CalledProcessError, ValueError, IndexError) as e:
        logging.error(f"Match failed: {e}")
        return None, None, None

# --- Scaling factor for r ---

def compute_exact_mean_abs(rounds: int) -> float:
    total = 0.0
    for k in range(rounds + 1):
        prob = math.comb(rounds, k) * (0.5 ** rounds)
        total += abs(2 * k - rounds) * prob
    return total / rounds

# --- Other functions ---

def generate_shifts(size: int, iteration: int, layer_id: int, round_idx: int) -> np.ndarray:
    """Generate random shifts (+1 or -1) for SPSA perturbations."""
    np.random.seed(42 + iteration * 100000 + layer_id * 100 + round_idx)
    return np.random.choice([-1, 1], size=size)

def get_network_layers(net_proto: pb.Net) -> List[pb.Weights.Layer]:
    """Recursively extract all layers from a network protobuf."""
    layers = []
    def recursive_extract(obj):
        if obj.DESCRIPTOR.name == "Layer":
            layers.append(obj)
        for field_desc, value in obj.ListFields():
            if field_desc.type == field_desc.TYPE_MESSAGE:
                if field_desc.label == field_desc.LABEL_REPEATED:
                    for item in value:
                        recursive_extract(item)
                else:
                    recursive_extract(value)
    recursive_extract(net_proto)
    return layers

def validate_dependencies(match_script: str, match_book: str, eval_book: str, base_network: str) -> None:
    required_paths = [
        (os.path.join(PATH_MATCH, f"{match_script}.sh"), "Match script"),
        (os.path.join(PATH_BOOKS, match_book), "Match book"),
        (os.path.join(PATH_BOOKS, eval_book), "Eval book"),
        (os.path.join(PATH_NETWORKS, f"{base_network}.pb.gz"), "Base network"),
    ]
    for path, desc in required_paths:
        if not os.path.exists(path):
            logging.error(f"{desc} not found: {path}")
            exit(1)

# --- Main function ---

def main_training_loop(config_path: str) -> None:
    """Main training loop using SPSA optimization."""
    SPSA_SCALE = round(1 / compute_exact_mean_abs(ROUNDS), 3)

    # Collect all configuration parameters for logging
    config_params = {
        "name": NAME,
        "iterations": ITERATIONS,
        "structure": STRUCTURE,
        "base_network": BASE_NETWORK,
        "match_script": MATCH_SCRIPT,
        "match_games": MATCH_GAMES,
        "rounds": ROUNDS,
        "SPSA_SCALE": SPSA_SCALE,
        "r_end": R_END,
        "adj": ADJ,
        "elo_average": ELO_AVERAGE,
        "match_book": MATCH_BOOK,
        "eval_book": EVAL_BOOK,
        "eval_iterations_games": EVAL_ITERATIONS_GAMES,
        "eval_points": EVAL_POINTS,
        "history_interval": HISTORY_INTERVAL,
        "A": A,
        "alpha": ALPHA,
        "gamma": GAMMA,
        "net_structure": total_data
    }
    logging.info(f"Starting {NAME}:\n{json.dumps(config_params, indent=2)}")

    NET_PREFIX = f"{BASE_NETWORK}_{NAME}"
    shutil.copy(f"{PATH_NETWORKS}/{BASE_NETWORK}.pb.gz", f"{PATH_NETWORKS}/{NET_PREFIX}_test0.pb.gz")
    os.system(f"md5sum {PATH_NETWORKS}/{NET_PREFIX}_test0.pb.gz")

    book_data = read_pgn_book(os.path.join(PATH_BOOKS, MATCH_BOOK))
    if not book_data:
        logging.error("Failed to load book data")
        return

    reset_position_pool(book_data)
    start_time = time.time()
    elo_history = {layer_id: [] for layer_id, _ in total_data}
    result_history = {layer_id: [] for layer_id, _ in total_data}

    initial_net = Net()
    initial_net.parse_proto(f"{PATH_NETWORKS}/{BASE_NETWORK}.pb.gz")
    initial_weights = {
        i: initial_net.denorm_layer_v2(layer).copy()
        for i, layer in enumerate(get_network_layers(initial_net.pb))
    }

    logging.info(f"Variant {NAME}, Iterations: {ITERATIONS}, Eval points: {EVAL_POINTS}")

    orig_net = Net()
    orig_net.parse_proto(f"{PATH_NETWORKS}/{NET_PREFIX}_test0.pb.gz")
    layers = get_network_layers(orig_net.pb)
    weights = [orig_net.denorm_layer_v2(layer).copy() for layer in layers]

    for iteration in range(1, ITERATIONS + 1):
        if os.path.exists(os.path.join(PATH_CURRENT, f"stop_{NAME}.txt")):
            logging.info(f"Stop file detected, terminating run for {NAME}")
            break
        logging.info(f"{NAME}, Iteration {iteration}/{ITERATIONS} ({(iteration - 1) / ITERATIONS:.1%})")

        # Applying perturbations and running matches
        for layer_id, c_end in total_data:
            # c and r constant, kept original SPSA functions anyway
            c_init = c_end * (ITERATIONS ** GAMMA)
            a_end = R_END * (c_end ** 2)
            a_init = a_end * ((ITERATIONS + A) ** ALPHA)
            c = c_init / (iteration ** GAMMA)
            a = a_init / ((A + iteration) ** ALPHA)
            r = a / (c ** 2)

            # Running [ROUNDS] number of matches and averaging normalized results
            std_array = np.full_like(weights[layer_id], np.std(weights[layer_id]))
            shifts_list, elo_results, converted_results, draw_rates = [], [], [], []
            for round_idx in range(ROUNDS):
                shifts = generate_shifts(weights[layer_id].size, iteration, layer_id, round_idx)

                for sign, suffix in [(1, "test1"), (-1, "test2")]:
                    net = copy.deepcopy(orig_net)
                    w = [w.copy() for w in weights]
                    w[layer_id] += sign * c * std_array * shifts
                    for layer, weights_layer in zip(get_network_layers(net.pb), w):
                        net.fill_layer_v2(layer, weights_layer)
                    net.save_proto(f"{PATH_NETWORKS}/{NET_PREFIX}_{suffix}.pb.gz", log=False, compresslevel=0)

                book_temp = write_shuffled_book(
                    book_data, MATCH_GAMES, os.path.join(PATH_BOOKS, MATCH_BOOK),
                    NAME, iteration, layer_id, round_idx
                )
                match_result, draw_rate, los = run_match(
                    MATCH_SCRIPT, f"{NET_PREFIX}_test1", f"{NET_PREFIX}_test2", MATCH_GAMES, book_temp
                )
                if match_result is None:
                    logging.warning(f"Skipping round {round_idx} for layer {layer_id}")
                    continue

                elo_history[layer_id].append(match_result)

                # Normalizing
                result_converted = match_result / ELO_AVERAGE

                result_history[layer_id].append(result_converted)
                shifts_list.append(shifts * result_converted)
                elo_results.append(match_result)
                converted_results.append(result_converted)
                draw_rates.append(draw_rate)

            logging.info(
                f"{NAME}, Iteration {iteration}, Layer {layer_id} ({weights[layer_id].size}), "
                f"Rounds: {ROUNDS}: Elo={np.mean(np.abs(elo_results)):.1f}, "
                f"Conv={np.mean(np.abs(converted_results)):.2f}, Drawrate={np.mean(draw_rates):.1%}"
            )
            # Averaging
            shifts_avg = np.mean(shifts_list, axis=0)

            # Applying update to weights (in memory)
            update = c * std_array * r * shifts_avg * SPSA_SCALE
            # Shifting weight updates towards initial network (reduces noise)
            if ADJ > 0:
                prospective_weights = weights[layer_id] + update
                update -= ADJ * c * std_array * r * np.sign(prospective_weights - initial_weights[layer_id])
            weights[layer_id] += update

        # Saving network after completed iteration
        for layer, w in zip(layers, weights):
            orig_net.fill_layer_v2(layer, w)
        orig_net.save_proto(f"{PATH_NETWORKS}/{NET_PREFIX}_test0.pb.gz", log=False, compresslevel=0)

        # Match results statistics
        if iteration % HISTORY_INTERVAL == 0:
            logging.info(f"{NAME}, Iteration {iteration}: History Statistics")
            all_mean_abs_elo = []
            all_mean_abs_result_conv = []
            for layer_id, elos in elo_history.items():
                if elos:
                    mean_abs_elo = np.mean(np.abs(elos))
                    mean_abs_result_conv = np.mean(np.abs(result_history[layer_id]))
                    num_items = len(elos)
                    logging.info(
                        f"Layer {layer_id}: MeanAbsElo={mean_abs_elo:.1f}, "
                        f"MeanAbsResultConv={mean_abs_result_conv:.2f} ({num_items})"
                    )
                    all_mean_abs_elo.append(mean_abs_elo)
                    all_mean_abs_result_conv.append(mean_abs_result_conv)
            if all_mean_abs_elo:
                overall_mean_elo = np.mean(all_mean_abs_elo)
                overall_mean_result_conv = np.mean(all_mean_abs_result_conv)
                logging.info(
                    f"Overall: MeanAbsElo={overall_mean_elo:.1f}, "
                    f"MeanAbsResultConv={overall_mean_result_conv:.2f}"
                )

        # Network weights statistics
        deviations = [weights[i] - initial_weights[i] for i in initial_weights]
        total_deviation = sum(np.sum(np.abs(d)) for d in deviations)
        total_params = sum(w.size for w in weights)
        avg_deviation = total_deviation / total_params
        l2_norm = np.sqrt(sum(np.sum(d ** 2) for d in deviations)) / total_params
        logging.info(f"{NAME}, Iteration {iteration}: AbsDev={avg_deviation:.8f}, L2={l2_norm:.11f}")

        # Saving checkpoint networks and running evaluation matches
        if iteration in EVAL_POINTS:
            eval_net = f"{NET_PREFIX}_it{iteration:04}"
            orig_net.save_proto(f"{PATH_NETWORKS}/{eval_net}.pb.gz", log=False)

            # Running match
            logging.info(f"TEST MATCH: {eval_net} vs {BASE_NETWORK}")
            start_match = time.time()
            match_result, draw_rate, los = run_match(MATCH_SCRIPT, eval_net, BASE_NETWORK,
                                                    EVAL_ITERATIONS_GAMES,
                                                    EVAL_BOOK)
            match_time = (time.time() - start_match) / 60
            with open(f"match_results_{NAME}.txt", "a") as f:
                if match_result is not None:
                    logging.info(f"EVALUATION ({NAME} vs Base): {match_result:.2f} Elo, "
                                f"Draw rate={draw_rate:.2%}, LOS={los:.2%}, Took {match_time:.1f}min")
                    f.write(f"Iteration: {iteration:04}, {NAME}_vs_Base: {match_result:.2f} Elo, Draw rate: {draw_rate:.3f}, "
                           f"LOS={los:.3f}, AbsDev: {avg_deviation:.8f}, L2={l2_norm:.11f}\n")
                else:
                    logging.warning(f"Evaluation failed, Took {match_time:.1f}min")
                    f.write(f"{iteration:04} {NAME}_vs_Base None None None {avg_deviation:.8f} {l2_norm:.11f}\n")

        elapsed = time.time() - start_time
        total_est = elapsed * ITERATIONS / iteration
        remaining = total_est - elapsed
        elapsed_h, elapsed_m = divmod(int(elapsed // 60), 60)
        total_est_h, total_est_m = divmod(int(total_est // 60), 60)
        remaining_h, remaining_m = divmod(int(remaining // 60), 60)
        avg_per_iter = elapsed / iteration / 60
        end_time = (datetime.now() + timedelta(seconds=remaining)).strftime("%a, %Y-%m-%d, %H:%M")
        logging.info(
            f"Variant {NAME}, Iteration {iteration}, Elapsed: {elapsed_h}h{elapsed_m:02d}m, "
            f"Total est: {total_est_h}h{total_est_m:02d}m, Remaining: {remaining_h}h{remaining_m:02d}m, "
            f"Avg/iter: {avg_per_iter:.2f}m, End: {end_time}"
        )

    logging.info(f"Training complete for {NAME}")

    # Backup the config file
    try:
        shutil.copy(config_path, os.path.join(PATH_LOGS, f"config_{NAME}.json"))
        logging.info(f"Backed up config file to {os.path.join(PATH_LOGS, f'config_{NAME}.json')}")
    except Exception as e:
        logging.error(f"Failed to back up config file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Network Training with SPSA")
    parser.add_argument("--config", default="config.json", help="Path to configuration file")
    parser.add_argument("--network-structure", default=None, help="Network ID to print structure (e.g., 744706)")
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    if args.network_structure:
        NAME = "structure"
        print_network_structure(args.network_structure)
        sys.exit(0)

    # Existing configuration loading
    config = load_config(args.config)

    NAME = config.get("name", "run01")
    BASE_NETWORK = config.get("base_network", "744706")
    STRUCTURE = config.get("structure", "T74_policy")
    MATCH_SCRIPT = config.get("match_script", "policy_match")
    ITERATIONS = config.get("iterations", 50)
    ROUNDS = config.get("rounds", 16)
    MATCH_GAMES = config.get("match_games", 184)
    R_END = config.get("r_end", 0.0085)
    ADJ = config.get("adj", 0.005)
    ELO_AVERAGE = config.get("elo_average", 18.0)
    HISTORY_INTERVAL = config.get("history_interval", 1)
    MATCH_BOOK = config.get("match_book", "book_4moves_2023_cp-5to+5_95710pos.pgn")
    EVAL_BOOK = config.get("eval_book", "book_4moves_2023_cp-5to+5_95710pos_match_2500pos.pgn")
    EVAL_ITERATIONS_GAMES = config.get("eval_iterations_games", 5000)
    EVAL_POINTS = config.get("eval_points", [10, 20, 30, 40, 50])
    A = config.get("A", 0.)
    ALPHA = config.get("alpha", 0.)
    GAMMA = config.get("gamma", 0.)
    LOG_LEVEL = config.get("log_level", "INFO")

    setup_logging(NAME, LOG_LEVEL)

    validate_config(config)
    validate_dependencies(MATCH_SCRIPT, MATCH_BOOK, EVAL_BOOK, BASE_NETWORK)

    # Load and validate network structure
    with open("./net_structure.py") as f:
        exec(f.read())
    if STRUCTURE not in globals():
        logging.error(f"Structure {STRUCTURE} not defined in net_structure.py")
        exit(1)
    total_data = globals()[STRUCTURE]
    logging.info(f"Starting training from {BASE_NETWORK}")
    main_training_loop(config_path=args.config)
