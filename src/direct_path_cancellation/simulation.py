"""
This module provides functions for simulating signals for direct path cancellation.
"""

import numpy as np

def generate_signal(n_samples: int, seed: int = None) -> np.ndarray:
    """
    Generates a base random complex-valued signal.

    Args:
        n_samples: The number of samples to generate.
        seed: An optional random seed for reproducibility.

    Returns:
        The generated complex-valued signal.
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(n_samples) + 1j * np.random.randn(n_samples)

def apply_doppler_shift(signal: np.ndarray, fs: float, doppler_freq: float) -> np.ndarray:
    """
    Applies a Doppler shift to a signal.

    Args:
        signal: The input signal.
        fs: The sampling frequency in Hz.
        doppler_freq: The Doppler frequency in Hz.

    Returns:
        The signal with the Doppler shift applied.
    """
    n_samples = len(signal)
    t = np.arange(n_samples) / fs
    doppler_shift = np.exp(1j * 2 * np.pi * doppler_freq * t)
    return signal * doppler_shift

def apply_delay_and_attenuation(
    signal: np.ndarray, delay_samples: int, attenuation_db: float
) -> np.ndarray:
    """
    Applies a time delay and attenuation to a signal.

    Args:
        signal: The input signal.
        delay_samples: The delay in samples.
        attenuation_db: The attenuation in dB.

    Returns:
        The delayed and attenuated signal.
    """
    # Apply attenuation
    attenuation_linear = 10 ** (-attenuation_db / 20)
    attenuated_signal = signal * attenuation_linear

    # Apply delay
    if delay_samples == 0:
        return attenuated_signal

    delayed_signal = np.zeros_like(attenuated_signal)
    if delay_samples < len(attenuated_signal):
        delayed_signal[delay_samples:] = attenuated_signal[:-delay_samples]

    return delayed_signal

def radar_range_equation_attenuation(
    pt: float, gt: float, gr: float, rcs: float, f: float, r: float
) -> float:
    """
    Calculates the attenuation in dB based on the radar range equation.

    Note: This is a simplified one-way path loss for simulation purposes.

    Args:
        pt: Transmit power in watts.
        gt: Transmit antenna gain.
        gr: Receive antenna gain.
        rcs: Radar cross-section in m^2.
        f: Frequency in Hz.
        r: Range in meters.

    Returns:
        The path loss in dB.
    """
    c = 299792458.0  # Speed of light in m/s
    wavelength = c / f
    lambda_sq = wavelength ** 2

    # Friis transmission equation for power received
    pr = pt * gt * gr * lambda_sq / ((4 * np.pi * r) ** 2)

    # For simulation, we'll consider the ratio of received to transmitted power
    path_loss_linear = pr / pt

    # Convert to dB
    path_loss_db = 10 * np.log10(path_loss_linear)

    return -path_loss_db # Return as a positive attenuation value

def simulate_scenario(
    n_samples: int,
    fs: float,
    snr_db: float,
    target_delay_s: float,
    target_doppler_hz: float,
    transmit_power_w: float,
    antenna_gain_db: float,
    radar_cross_section_m2: float,
    frequency_hz: float,
    target_range_m: float,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates a realistic two-channel scenario with independent noise.

    Returns:
        channel1 (np.ndarray): The reference channel (direct path + noise).
        channel2 (np.ndarray): The measurement channel (direct path + target + noise).
        target_signal_clean (np.ndarray): The ground truth target signal for verification.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate clean source signals
    direct_path_clean = generate_signal(n_samples)
    target_source_clean = generate_signal(n_samples)

    # Calculate target attenuation using the radar range equation
    attenuation_db = radar_range_equation_attenuation(
        pt=transmit_power_w,
        gt=10**(antenna_gain_db / 10),
        gr=10**(antenna_gain_db / 10),
        rcs=radar_cross_section_m2,
        f=frequency_hz,
        r=target_range_m,
    )

    # Create the clean target signal with all transformations
    target_attenuated = apply_delay_and_attenuation(
        target_source_clean, 0, attenuation_db
    )
    target_delayed = apply_delay_and_attenuation(
        target_attenuated, int(target_delay_s * fs), 0
    )
    target_signal_clean = apply_doppler_shift(
        target_delayed, fs, target_doppler_hz
    )

    # Define a noise generation function based on a reference signal's power
    def generate_noise_for_signal(ref_signal, snr_db):
        signal_power = np.mean(np.abs(ref_signal) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = (
            np.sqrt(noise_power / 2)
            * (np.random.randn(len(ref_signal)) + 1j * np.random.randn(len(ref_signal)))
        )
        return noise

    # Channel 1: direct path + noise
    noise1 = generate_noise_for_signal(direct_path_clean, snr_db)
    channel1 = direct_path_clean + noise1

    # Channel 2: direct path + target + independent noise
    channel2_clean = direct_path_clean + target_signal_clean
    noise2 = generate_noise_for_signal(direct_path_clean, snr_db) # Noise floor is the same
    channel2 = channel2_clean + noise2

    return channel1, channel2, target_signal_clean
