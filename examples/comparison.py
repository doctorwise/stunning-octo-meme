"""
A script to compare the performance of the Wiener and LMS filters.
"""

import numpy as np
import matplotlib.pyplot as plt
from direct_path_cancellation.simulation import simulate_scenario
from direct_path_cancellation.filters import WienerFilter, LMSFilter

def plot_results(original, wiener_cleaned, lms_cleaned, fs):
    """Generates and saves plots of the signals."""
    time = np.arange(len(original)) / fs * 1e6 # Time in microseconds

    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True, sharey=True)

    # Original Signal
    axs[0].plot(time, np.real(original), label="Real Part")
    axs[0].plot(time, np.imag(original), label="Imaginary Part")
    axs[0].set_title("Original Signal (Channel 2)")
    axs[0].grid(True)
    axs[0].legend()

    # Wiener Filter Output
    axs[1].plot(time, np.real(wiener_cleaned), label="Real Part")
    axs[1].plot(time, np.imag(wiener_cleaned), label="Imaginary Part")
    axs[1].set_title("After Wiener Filter")
    axs[1].grid(True)
    axs[1].legend()

    # LMS Filter Output
    axs[2].plot(time, np.real(lms_cleaned), label="Real Part")
    axs[2].plot(time, np.imag(lms_cleaned), label="Imaginary Part")
    axs[2].set_title("After LMS Filter")
    axs[2].grid(True)
    axs[2].legend()

    axs[2].set_xlabel("Time (us)")
    fig.tight_layout()

    plt.savefig("plots/cancellation_comparison.png")
    print("\nSaved plot to plots/cancellation_comparison.png")

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
    filter_length = 128
    lms_learning_rate = 0.5
    lms_epochs = 10

    # --- Generate Data ---
    print("Generating simulation data...")
    channel1, channel2, _ = simulate_scenario(
        n_samples=8192,
        fs=1e6,
        snr_db=40,
        target_delay_s=100e-6,
        target_doppler_hz=200,
        transmit_power_w=1000,
        antenna_gain_db=20,
        radar_cross_section_m2=1,
        frequency_hz=10e9,
        target_range_m=7000, # Further target for weaker signal
        seed=123,
    )

    # --- Run Wiener Filter ---
    print("Running Wiener filter...")
    wiener_filter = WienerFilter(filter_length=filter_length)
    wiener_cleaned = wiener_filter(channel1, channel2)
    wiener_cancellation = calculate_cancellation_db(channel2, wiener_cleaned)

    # --- Run LMS Filter (Multi-Epoch) ---
    print(f"Running LMS filter for {lms_epochs} epochs...")
    lms_filter = LMSFilter(filter_length=filter_length, learning_rate=lms_learning_rate)
    for epoch in range(lms_epochs):
        # Always use the original channel data
        lms_cleaned = lms_filter(channel1, channel2)
        print(f"  Epoch {epoch+1}/{lms_epochs} complete.")

    lms_cancellation = calculate_cancellation_db(channel2, lms_cleaned)

    # --- Print Report ---
    print("\n--- Filter Performance Comparison ---")
    print(f"Wiener Filter Cancellation: {wiener_cancellation:.2f} dB")
    print(f"LMS Filter Cancellation:    {lms_cancellation:.2f} dB")
    print("------------------------------------")

    # --- Generate Plots ---
    plot_results(channel2, wiener_cleaned, lms_cleaned, fs)

if __name__ == "__main__":
    main()
