import numpy as np
import matplotlib.pyplot as plt
from pyldpc import make_ldpc, encode, decode, get_message
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

def apply_freezing(Lc, threshold=0.95):
    """
    Freezing heuristic: Fix the state of high-confidence variable nodes.
    Parameters:
    - Lc: Log-likelihood ratios (LLRs) of the received message.
    - threshold: Confidence threshold for freezing variable nodes.
    Returns:
    - Updated LLRs with high-confidence nodes frozen.
    """
    frozen_indices = np.abs(Lc) > np.log(threshold / (1 - threshold))
    Lc[frozen_indices] = np.sign(Lc[frozen_indices]) * np.inf  # Freeze the variable nodes
    return Lc

def apply_random_perturbations(Lc, epsilon=0.05):
    """
    Random perturbations heuristic: Introduce stochastic variations in prior probabilities.
    Parameters:
    - Lc: Log-likelihood ratios (LLRs) of the received message.
    - epsilon: Maximum magnitude of random perturbations.
    Returns:
    - Perturbed LLRs.
    """
    perturbations = np.random.uniform(-epsilon, epsilon, size=Lc.shape)
    return Lc + perturbations

def test_ldpc_with_heuristics():
    n = 15  # Length of the codeword
    d_v = 4  # Degree of variable nodes
    d_c = 5  # Degree of check nodes

    # Generate sparse and dense Low-Density Parity-Check (LDPC) matrices
    H_sparse, G_sparse = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
    H_dense, G_dense = make_ldpc(n, d_v, d_c, systematic=True, sparse=False)

    signal_to_noise_ratios = range(1, 21)  # Signal-to-Noise Ratio values from 1 to 20 decibels
    metrics = {
        "Standard": {"Sparse": {"latency": [], "ber": [], "success_rate": []},
                     "Dense": {"latency": [], "ber": [], "success_rate": []}},
        "Heuristic": {"Sparse": {"latency": [], "ber": [], "success_rate": []},
                      "Dense": {"latency": [], "ber": [], "success_rate": []}}
    }

    num_trials = 100  # Number of trials for each Signal-to-Noise Ratio

    # Iterate over Signal-to-Noise Ratios
    for signal_to_noise_ratio in signal_to_noise_ratios:
        logger.info(f"Testing Signal-to-Noise Ratio: {signal_to_noise_ratio} decibels")
        for heuristic_type in ["Standard", "Heuristic"]:
            for code_type, (H, G) in {"Sparse": (H_sparse, G_sparse), "Dense": (H_dense, G_dense)}.items():
                total_latency = 0
                total_errors = 0
                total_success = 0

                for trial in range(num_trials):
                    # Generate random message
                    k = G.shape[1]
                    original_message = np.random.randint(0, 2, size=k)

                    # Encode message
                    encoded_message = encode(G, original_message, signal_to_noise_ratio)

                    # Add noise
                    noise = np.random.normal(0, 10**(-signal_to_noise_ratio/20), encoded_message.shape)
                    noisy_message = encoded_message + noise

                    # Apply heuristics if applicable
                    Lc = 2 * noisy_message / (10**(-signal_to_noise_ratio / 10))
                    if heuristic_type == "Heuristic":
                        Lc = apply_freezing(Lc)
                        Lc = apply_random_perturbations(Lc)

                    # Decode message
                    start_time = time.time()
                    try:
                        decoded_codeword = decode(H, noisy_message, signal_to_noise_ratio, maxiter=1000)
                        decoding_time = time.time() - start_time

                        # Extract message
                        decoded_message = get_message(G, decoded_codeword)

                        # Check success
                        success = np.array_equal(original_message, decoded_message)
                        if not success:
                            total_errors += 1

                        total_success += int(success)
                    except Exception as e:
                        logger.warning(f"Trial {trial+1} for {code_type} {heuristic_type} at Signal-to-Noise Ratio {signal_to_noise_ratio} failed to converge.")
                        decoding_time = 0  # Failed decoding adds zero latency
                        total_errors += 1  # Count as a block error

                    # Accumulate decoding latency
                    total_latency += decoding_time

                # Compute average metrics
                metrics[heuristic_type][code_type]["latency"].append(total_latency / num_trials)
                metrics[heuristic_type][code_type]["ber"].append(total_errors / num_trials)
                metrics[heuristic_type][code_type]["success_rate"].append(total_success / num_trials)

                # Log results
                logger.info(
                    f"{heuristic_type} {code_type} Metrics at Signal-to-Noise Ratio {signal_to_noise_ratio}: "
                    f"Latency={metrics[heuristic_type][code_type]['latency'][-1]:.6f}, "
                    f"Block Error Rate={metrics[heuristic_type][code_type]['ber'][-1]:.4f}, "
                    f"Decoding Success Rate={metrics[heuristic_type][code_type]['success_rate'][-1]:.4f}"
                )

    # Generate separate plots for Standard and Heuristic approaches
    for heuristic_type in ["Standard", "Heuristic"]:
        plot_metrics(signal_to_noise_ratios, metrics[heuristic_type], "latency", f"{heuristic_type} Decoding Latency (s)", f"{heuristic_type} Decoding Latency vs Signal-to-Noise Ratio")
        plot_metrics(signal_to_noise_ratios, metrics[heuristic_type], "ber", f"{heuristic_type} Block Error Rate", f"{heuristic_type} Block Error Rate vs Signal-to-Noise Ratio", yscale="log")
        plot_metrics(signal_to_noise_ratios, metrics[heuristic_type], "success_rate", f"{heuristic_type} Decoding Success Rate", f"{heuristic_type} Decoding Success Rate vs Signal-to-Noise Ratio")

def plot_metrics(signal_to_noise_ratios, metrics, metric_key, ylabel, title, yscale=None):
    plt.figure(figsize=(10, 6))
    plt.plot(signal_to_noise_ratios, metrics["Sparse"][metric_key], marker='o', linestyle='--', color='red', label="Sparse")
    plt.plot(signal_to_noise_ratios, metrics["Dense"][metric_key], marker='s', color='blue', linestyle='--', label="Dense")
    plt.xlabel("Signal-to-Noise Ratio (decibels)")
    plt.ylabel(ylabel)
    if yscale:
        plt.yscale(yscale)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_ldpc_with_heuristics()
