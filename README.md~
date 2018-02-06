Reliability
===========

Python package to compute the reliability of measurements.


Example usage
-------------

~~~
import numpy
from reliability import split_half

# Fake data details.
m_range = [300, 1000]
sd_range = [250, 750]
min_value = 50.0
# Construct fake data.
n_trials = 100
n_subjects = 30
x = numpy.zeros((n_trials, n_subjects), dtype=float)
for i in range(n_subjects):
# Choose a random mean for this participant.
m = numpy.random.rand() * (m_range[1]-m_range[0]) + m_range[0]
# Choose a random standard deviation for this participant.
sd = numpy.random.rand() * (sd_range[1]-sd_range[0]) + sd_range[0]
# Create random values.
x[:,i] = m + numpy.random.randn(n_trials)*sd
# Replace all values below the lowest value.
x[x<min_value] = min_value

# Compute the split-half reliabilities.
r, sem = split_half(x, n_splits=100, mode='spearman-brown')
print("Test-retest reliability is %.2f (SEM=%.2f)" % (r, sem))
~~~
