"""
A script to compare the performance of the Wiener and LMS filters.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from direct_path_cancellation.simulation import simulate_scenario
from direct_path_cancellation.filters import WienerFilter, LMSFilter

def plot_results(original, wiener_cleaned, lms_cleaned, target_clean, fs, wiener_db, lms_db):
    """Generates and saves a spectral plot of the signals."""
    nperseg = 2048

    # Calculate PSDs
    freqs, psd_orig = welch(original, fs=fs, nperseg=nperseg)
    _, psd_wiener = welch(wiener_cleaned, fs=fs, nperseg=nperseg)
    _, psd_lms = welch(lms_cleaned, fs=fs, nperseg=nperseg)
    _, psd_target = welch(target_clean, fs=fs, nperseg=nperseg)

    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(freqs / 1e6, 10 * np.log10(psd_orig), label="Original Signal", alpha=0.7)
    plt.plot(freqs / 1e6, 10 * np.log10(psd_wiener), label="After Wiener Filter", alpha=0.7)
    plt.plot(freqs / 1e6, 10 * np.log10(psd_lms), label="After LMS Filter", alpha=0.7)
    plt.plot(freqs / 1e6, 10 * np.log10(psd_target), label="Target Signal (Ground Truth)",
             color='black', linestyle='--', linewidth=2)

    plt.title("Spectral Comparison of Cancellation Filters")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power Spectral Density (dB/Hz)")
    plt.grid(True)
    plt.legend()
    plt.ylim(bottom=-140) # Show the noise floor

    # Add text box with results
    results_text = (
        f"Cancellation:\n"
        f"Wiener: {wiener_db:.2f} dB\n"
        f"LMS:    {lms_db:.2f} dB"
    )
    plt.text(0.05, 0.95, results_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Auto-create plots directory
    if not os.path.exists("plots"):
        os.makedirs("plots")

    plt.savefig("plots/spectral_comparison.png")
    print("\nSaved plot to plots/spectral_comparison.png")

def calculate_cancellation_db(original_signal: np.ndarray, cleaned_signal: np.ndarray) -> float:
    """Calculates the cancellation ratio in dB."""
    power_before = np.mean(np.abs(original_signal)**2)
    power_after = np.mean(np.abs(cleaned_signal)**2)
    if power_after == 0:
        return np.inf
    return 10 * np.log10(power_before / power_after)

def main():
    """Main function to run the comparison."""
    # --- Simulation Parameters ---
    filter_length = 256
    lms_learning_rate = 0.5
    lms_epochs = 10
    n_samples = 32768 # More samples for better PSD
    fs = 10e6  # 10 MHz sampling rate
    signal_bandwidth_hz = 5e6  # 5 MHz signal bandwidth
    snr_db = 40
    target_delay_s = 100e-6
    target_doppler_hz = 200
    transmit_power_w = 1000
    antenna_gain_db = 20
    radar_cross_section_m2 = 1
    frequency_hz = 10e9
    target_range_m = 7000
    seed = 123

    # --- Generate Data ---
    print("Generating simulation data...")
    channel1, channel2, target_signal_clean = simulate_scenario(
        n_samples=n_samples,
        fs=fs,
        snr_db=snr_db,
        signal_bandwidth_hz=signal_bandwidth_hz,
        target_delay_s=target_delay_s,
        target_doppler_hz=target_doppler_hz,
        transmit_power_w=transmit_power_w,
        antenna_gain_db=antenna_gain_db,
        radar_cross_section_m2=radar_cross_section_m2,
        frequency_hz=frequency_hz,
        target_range_m=target_range_m,
        seed=seed,
    )

    # --- Run Wiener Filter ---
    print("Running Wiener filter...")
    wiener_filter = WienerFilter(filter_length=filter_length)
    wiener_cleaned = wiener_filter(channel1, channel2)
    wiener_cancellation = calculate_cancellation_db(channel2, wiener_cleaned)

    # --- Run LMS Filter (Multi-Epoch) ---
    print(f"Running LMS filter for {lms_epochs} epochs...")
    lms_filter = LMSFilter(filter_length=filter_length, learning_rate=lms_learning_rate)
    for _ in range(lms_epochs):
        lms_cleaned = lms_filter(channel1, channel2)
    lms_cancellation = calculate_cancellation_db(channel2, lms_cleaned)

    # --- Print Report & Generate Plots ---
    print("\n--- Filter Performance Comparison ---")
    print(f"Wiener Filter Cancellation: {wiener_cancellation:.2f} dB")
    print(f"LMS Filter Cancellation:    {lms_cancellation:.2f} dB")
    print("------------------------------------")
    plot_results(channel2, wiener_cleaned, lms_cleaned, target_signal_clean, fs, wiener_cancellation, lms_cancellation)

if __name__ == "__main__":
    main()
