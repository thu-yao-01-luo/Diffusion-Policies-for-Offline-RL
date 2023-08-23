from helpers import compute_entropy
import numpy as np

def test_entropy():
    # Parameters
    mean = 0  # Mean of the Gaussian distribution
    std_dev = 2  # Standard deviation of the Gaussian distribution
    num_samples = 100  # Number of samples to generate

    # Generate Gaussian samples
    samples = np.random.normal(mean, std_dev, (num_samples, 6))

    # print("Generated Samples:", samples)
    print(compute_entropy(samples))

if __name__ == '__main__':
    test_entropy()
