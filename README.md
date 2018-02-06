Reliability
===========

Python package to compute the reliability of measurements.


Example usage
-------------

~~~ .python
import numpy
from reliability import split_half, test_retest

# Fake data details.
m_range = [300, 1000]
sd_range = [250, 750]
min_value = 50.0
n_trials = 100
n_subjects = 30
n_measurements = 5

# Construct fake data for split-half reliability.
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
print("Split-half reliability is %.2f (SEM=%.2f)" % (r, sem))

# Construct fake data for repeated measurements.
x = numpy.zeros((n_measurements, n_subjects), dtype=float)
for i in range(n_subjects):
# Choose a random mean for this participant.
m = numpy.random.rand() * (m_range[1]-m_range[0]) + m_range[0]
# Choose a random standard deviation for this participant.
sd = numpy.random.rand() * (sd_range[1]-sd_range[0]) + sd_range[0]
# Create random values for each measurement.
for j in range(n_measurements):
    x[j,i] = numpy.mean(m + numpy.random.randn(n_trials)*sd)

# Compute the test-retest reliability.
r = test_retest(x, mode='harris')
print("Test-retest reliability is %.2f" % (r))
~~~
