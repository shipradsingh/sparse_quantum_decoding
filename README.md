# Sparse Quantum Decoding with Belief Propagation

This project explores the application of Belief Propagation (BP) for decoding sparse quantum Low-Density Parity-Check (LDPC) codes. It evaluates BP's scalability, accuracy, and computational efficiency, incorporating heuristic techniques such as freezing and random perturbations to improve decoding performance.

## Project Structure

```
.
├── src/                                # Source code for LDPC decoding
│   ├── belief_propagation_decoder.py                     # Implementation of Belief Propagation decoding and heuristics
├── tests/                              # Test suite
│   └── test_belief_propagation_decoder.py  # Unit tests for BP decoding
├── docs/                               # Relevant documents and final report
├── results/                            # Directory for plots and metrics
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
```

## Features

- Implementation of Belief Propagation for quantum LDPC codes using the `pyldpc` library.
- Heuristic enhancements to improve decoding robustness:
  - **Freezing:** Fixes high-confidence variable nodes.
  - **Random Perturbations:** Introduces stochastic variations in prior probabilities.
- Comprehensive metrics evaluation:
  - Block Error Rate (BER)
  - Decoding Success Rate
  - Latency
- GPU-accelerated parallel message passing for scalability.

## Requirements

This project is built using Python. The dependencies can be installed with:

```bash
pip install -r requirements.txt
```

Dependencies include:

- `numpy`
- `matplotlib`
- `pyldpc`
- `pytest`
- `scipy`
- `networkx`

## Running the Code

### Main Script
The main script evaluates BP decoding with heuristics for sparse quantum LDPC codes:

```bash
python src/belief_propagation_decoder.py
```

### Tests
The test suite validates the implementation of BP decoding and its heuristic enhancements. Run the tests using:

```bash
pytest tests/
```

Example output:

```
=============================================================== test session starts ================================================================
platform darwin -- Python 3.12.6, pytest-8.3.4, pluggy-1.5.0
rootdir: /Users/shiprasingh/quantum_architecture/sparse_quantum_decoding
collected 4 items                                                                                                                                  

tests/test_belief_propagation_decoder.py ....                                                                                                [100%]

================================================================ 4 passed in 1.04s =================================================================
```

## Results

The results are stored in the `results/` directory and include:

- Plots comparing decoding latency, block error rate, and success rate for standard and heuristic BP.
- Metrics evaluated across a range of signal-to-noise ratios (SNR).

## Acknowledgments

- Dr. Ramin Ayanzadeh for guidance and encouragement to explore quantum error correction.
- The `pyldpc` library developers for their well-documented and robust tools.
