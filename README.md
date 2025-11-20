# Direct Path Signal Cancellation

This repository provides a Python library for direct path signal cancellation, a common problem in signal processing applications like radar and sonar. It includes implementations of a block-adaptive Wiener filter and a sample-adaptive Normalized LMS (NLMS) filter.

## Project Structure

- `src/direct_path_cancellation/`: The core Python library.
  - `filters.py`: Contains the `WienerFilter` and `LMSFilter` implementations.
  - `simulation.py`: Provides tools to generate realistic simulation data.
- `tests/`: Unit and integration tests for the library.
- `examples/`: Example scripts demonstrating the library's usage.
  - `comparison.py`: A script to compare the performance of the two filters.
- `plots/`: (Git-ignored) A directory where output plots are saved.

## Installation and Usage

This project is managed using `pdm`.

### 1. Installation

To install the project and its dependencies, run the following command from the root of the repository:

```bash
pdm install
```

This will create a virtual environment and install all necessary packages (`numpy`, `scipy`, `pytest`, `matplotlib`).

### 2. Running the Example

To see the filters in action and compare their performance, run the example script:

```bash
python examples/comparison.py
```

This script will:
1.  Generate a realistic dataset with a strong direct path signal and a weaker target signal.
2.  Run both the Wiener and LMS filters to cancel the interference.
3.  Print a report to the console comparing their cancellation performance in dB.
4.  Generate and save a plot to `plots/cancellation_comparison.png` that visually compares the original and cleaned signals.
