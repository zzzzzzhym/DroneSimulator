import numpy as np

class Normalization:
    def __init__(self):
        self.mean = 0.0
        self.m2 = 0.0   # variance * count (the "total variance" before average)
        self.count = 0  # number of samples seen so far
        self.threshold = 1e-6  # threshold for variance to treat the data as constant

    def add_batch(self, data: np.ndarray) -> None:
        """Calculate the mean and standard deviation after feed in a single batch of data.
        Use batch version of Welford's method to calculate the mean and variance of the data.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        data: 1D array"""
        batch_count = len(data)
        batch_mean = np.mean(data)
        batch_m2 = np.var(data)*batch_count

        delta_mean = batch_mean - self.mean
        new_mean = self.mean + delta_mean * batch_count / (self.count + batch_count)
        new_m2 = self.m2 + batch_m2 + delta_mean**2 * self.count * batch_count / (self.count + batch_count)

        self.count = self.count + batch_count
        self.mean = new_mean
        self.m2 = new_m2

    def get_normalization_params(self) -> tuple[float, float]:
        """x_scaled = (x - mean)*scale"""
        if self.count > 0:
            variance = self.m2 / self.count
            if variance < self.threshold:
                scale = 1.0
            else:
                scale = 1.0 / np.sqrt(variance)
        else:
            scale = 1.0
        return float(self.mean), float(scale)   # prevent returning np.float which confuses yaml
    
