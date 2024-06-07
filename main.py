import numpy as np
import time
import matplotlib.pyplot as plt

def f(x):
    return 4 / (1 + x**2)

def simpson_13(a, b, N):
    if N % 2 == 1:
        N += 1
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    S = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2])
    return (h / 3) * S

def rms_error(estimated_pi, true_pi=3.14159265358979323846):
    return np.sqrt((estimated_pi - true_pi)**2)

def main():
    a, b = 0, 1
    N_values = [10, 100, 1000, 10000]
    true_pi = 3.14159265358979323846

    results = []
    for N in N_values:
        start_time = time.time()
        estimated_pi = simpson_13(a, b, N)
        end_time = time.time()
        execution_time = end_time - start_time
        error = rms_error(estimated_pi, true_pi)
        results.append((N, estimated_pi, error, execution_time))
    
    for result in results:
        N, estimated_pi, error, execution_time = result
        print(f"N: {N}, Estimated Pi: {estimated_pi}, RMS Error: {error}, Execution Time: {execution_time} seconds")

    N_values = [result[0] for result in results]
    estimated_pis = [result[1] for result in results]
    errors = [result[2] for result in results]
    execution_times = [result[3] for result in results]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(N_values, estimated_pis, marker='o')
    plt.axhline(y=true_pi, color='r', linestyle='--', label='True Pi')
    plt.title('Estimated Pi')
    plt.xlabel('N')
    plt.ylabel('Estimated Pi')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(N_values, errors, marker='o', color='orange')
    plt.title('RMS Error')
    plt.xlabel('N')
    plt.ylabel('RMS Error')

    plt.subplot(1, 3, 3)
    plt.plot(N_values, execution_times, marker='o', color='green')
    plt.title('Execution Time')
    plt.xlabel('N')
    plt.ylabel('Time (seconds)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
