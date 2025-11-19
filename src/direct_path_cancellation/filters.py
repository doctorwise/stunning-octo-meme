"""
This module contains the filter implementations for direct path cancellation.
"""

import numpy as np
from collections import deque

class WienerFilter:
    """
    A block-adaptive Wiener filter for signal cancellation.
    This filter processes data in blocks, calculating and applying the
    optimal filter for each block.
    """
    def __init__(self, filter_length: int):
        """
        Initializes the WienerFilter.

        Args:
            filter_length: The number of taps in the filter.
        """
        self.filter_length = filter_length
        self.weights = np.zeros(filter_length, dtype=np.complex128)

    def __call__(self, channel1: np.ndarray, channel2: np.ndarray) -> np.ndarray:
        """
        Processes a block of input signals and returns the cleaned signal.

        Args:
            channel1: The reference signal (containing the interference).
            channel2: The primary signal (containing interference + target).

        Returns:
            The cleaned signal, with the interference removed.
        """
        x = np.asarray(channel1)
        d = np.asarray(channel2)
        L = self.filter_length
        N = len(x)

        if N < L:
            raise ValueError("Input block size must be at least the filter length.")

        # Construct the data matrix A for the least-squares problem
        A = np.zeros((N - L + 1, L), dtype=np.complex128)
        for i in range(L):
            A[:, i] = x[L - 1 - i : N - i]

        d_vec = d[L - 1:]

        # Estimate the auto-correlation matrix and cross-correlation vector
        Rxx = A.conj().T @ A
        Rdx = A.conj().T @ d_vec

        # Add regularization for numerical stability
        reg = 1e-6 * np.trace(Rxx)
        Rxx_reg = Rxx + reg * np.eye(L)

        # Solve the Wiener-Hopf equations
        try:
            self.weights = np.linalg.solve(Rxx_reg, Rdx)
        except np.linalg.LinAlgError:
            # If solving fails (e.g., for a block of zeros), keep old weights
            pass

        # Apply the filter to estimate the interference
        estimated_interference = A @ self.weights

        # Subtract the interference to get the cleaned signal
        cleaned_signal = d_vec - estimated_interference

        # Pad the output to match the input length
        output = np.zeros_like(d)
        output[:L-1] = d[:L-1]  # The first L-1 samples are unprocessed
        output[L-1:] = cleaned_signal

        return output

class LMSFilter:
    """
    An LMS filter for sample-by-sample adaptive signal cancellation.
    """
    def __init__(self, filter_length: int, learning_rate: float):
        """
        Initializes the LMSFilter.

        Args:
            filter_length: The number of taps in the filter.
            learning_rate: The step size for the filter updates.
        """
        self.filter_length = filter_length
        self.learning_rate = learning_rate
        self.weights = np.zeros(filter_length, dtype=np.complex128)
        self._buffer = deque(maxlen=filter_length)

    def __call__(self, channel1: np.ndarray, channel2: np.ndarray) -> np.ndarray:
        """
        Processes the input signals sample-by-sample and returns the cleaned signal.

        Args:
            channel1: The reference signal (containing the interference).
            channel2: The primary signal (containing interference + target).

        Returns:
            The cleaned signal, with the interference removed.
        """
        x = np.asarray(channel1)
        d = np.asarray(channel2)
        N = len(x)

        if N != len(d):
            raise ValueError("Input channels must have the same length.")

        output = np.zeros(N, dtype=np.complex128)

        # Process sample by sample
        for n in range(N):
            self._buffer.append(x[n])

            # Ensure the buffer is full before starting
            if len(self._buffer) == self.filter_length:
                x_vec = np.array(list(reversed(self._buffer)))

                # Calculate filter output
                y = np.dot(self.weights.conj(), x_vec)

                # Calculate error
                e = d[n] - y

                # Update weights (Normalized LMS)
                power = np.dot(x_vec.conj(), x_vec)
                if power > 1e-6:
                    step = self.learning_rate / (power + 1e-6)
                    self.weights += step * e.conj() * x_vec

                output[n] = e
            else:
                # Not enough data to process yet, output the original signal
                output[n] = d[n]

        return output
