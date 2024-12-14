import matplotlib.pyplot as plt
import numpy as np

def plot_roofline(performance_metrics, arithmetic_intensities, 
                  system_peak_tflops, memory_bandwidth_gbps):
    """
    Plots a roofline chart for a given system's theoretical limits and performance metrics.

    Parameters:
        performance_metrics (list): A list of performance metrics in TFLOPs/s.
        arithmetic_intensities (list): A list of arithmetic intensities (FLOPs/byte).
        system_peak_tflops (float): The theoretical peak performance of the system in TFLOPs/s.
        memory_bandwidth_gbps (float): The system's memory bandwidth in GB/s.
    """
    # Convert memory bandwidth to TFLOPs/byte
    memory_bandwidth_tflops_per_byte = memory_bandwidth_gbps / 1

    # Define x-axis (arithmetic intensity)
    x = np.logspace(-2, 2, 500)  # Logarithmic scale for x-axis

    # Define the roofline model
    # Performance is bounded by memory bandwidth (slope) and compute peak (horizontal line)
    roofline = np.minimum(memory_bandwidth_tflops_per_byte * x, system_peak_tflops)

    # Plot the roofline
    plt.figure(figsize=(10, 6))
    plt.plot(x, roofline, label="Roofline", color="blue", linewidth=2)

    # Plot the performance metrics
    plt.scatter(arithmetic_intensities, performance_metrics, color="red", label="Measured Performance", zorder=5)

    # Annotate each point
    for ai, perf in zip(arithmetic_intensities, performance_metrics):
        plt.annotate(f"({ai:.2f}, {perf:.2f})", (ai, perf), textcoords="offset points", xytext=(5, 5), fontsize=9)

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
arithmetic_intensities = [0.198, .198]
performances = [.3099, .0797]
system_peak_tflops = 2.5                      # System peak performance in TFLOPs/s
memory_bandwidth_tbps = 1.6                # System memory bandwidth in GB/s

plot_roofline(performances, arithmetic_intensities, system_peak_tflops, memory_bandwidth_tbps)
