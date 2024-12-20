import matplotlib.pyplot as plt
import numpy as np

def plot_roofline(performance_metrics, arithmetic_intensities, 
                  system_peak_tflops, memory_bandwidth_tbps, grid_sizes):
    """
    Plots a roofline chart for a given system's theoretical limits and performance metrics.

    Parameters:
        performance_metrics (list): A list of performance metrics in TFLOPs/s.
        arithmetic_intensities (list): A list of arithmetic intensities (FLOPs/byte).
        system_peak_tflops (float): The theoretical peak performance of the system in TFLOPs/s.
        memory_bandwidth_tbps (float): The system's memory bandwidth in TB/s.
        grid_sizes (list): List of grid sizes corresponding to each data point.
    """
    # Memory bandwidth in TB/s is already in the correct units for TFLOPs/byte
    memory_bandwidth_tflops_per_byte = memory_bandwidth_tbps

    # Define x-axis (arithmetic intensity)
    x = np.logspace(-2, 2, 500)  # Logarithmic scale for x-axis

    # Define the roofline model
    roofline = np.minimum(memory_bandwidth_tflops_per_byte * x, system_peak_tflops)

    # Plot the roofline
    plt.figure(figsize=(10, 6))
    plt.plot(x, roofline, label="Roofline", color="blue", linewidth=2)

    # Plot the performance metrics
    plt.scatter(arithmetic_intensities, performance_metrics, color="red", label="Measured Performance", zorder=5)

    # Set chart limits and scales
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Arithmetic Intensity (FLOPs/byte)")
    plt.ylabel("Performance (TFLOPs/s)")
    plt.title("Roofline Chart")

    # Add legends and grid
    plt.axhline(system_peak_tflops, color="gray", linestyle="--", label="Compute Peak")
    plt.axvline(system_peak_tflops / memory_bandwidth_tflops_per_byte, color="purple", linestyle="--", 
                label="Memory Bandwidth Bound")
    plt.legend()
    plt.grid(which="both", linestyle="--", linewidth=0.5)

    # Show the plot
    plt.tight_layout()
    plt.show()

# Example usage:
grid_sizes = [128, 256, 512, 1024, 2048, 4096]

# Fused kernel, pinned memory, no streams
# performances = [0.128017, 0.478837, 0.574791, 0.575178, 0.599341]
# arithmetic_intensities = [0.248729, 0.249026, 0.249174, 0.249248, 0.249285]

# Num streams = 2
# performances = [0.010066, 0.106564, 0.276888, 0.457067, 0.567327, 0.531406]
# arithmetic_intensities = [0.248729, 0.249026, 0.249174, 0.249248, 0.249285, 0.242366]

# Num streams = 4
# performances = [0.005788, 0.059295, 0.188104, 0.378843, 0.519462, 0.521576]
# arithmetic_intensities = [0.248729, 0.249026, 0.249174, 0.249248, 0.249285, 0.242366]

# Num streams = 8
# performances = [0.005569, 0.034584, 0.116214, 0.330147, 0.459892, 0.507910]
# arithmetic_intensities = [0.248729, 0.249026, 0.249174, 0.249248, 0.249285, 0.242366]

# Num streams = 16
performances = [0.003435, 0.017445, 0.063776, 0.199340, 0.383715, 0.471399]
arithmetic_intensities = [0.248729, 0.249026, 0.249174, 0.249248, 0.249285, 0.242366]

system_peak_tflops = 2.5
memory_bandwidth_tbps = 1.6

plot_roofline(performances, arithmetic_intensities, system_peak_tflops, memory_bandwidth_tbps, grid_sizes)
# save plot
plt.savefig("roofline_optimized+num_streams=16.png")