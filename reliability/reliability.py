import numpy
from scipy.stats import pearsonr


def split_half(x, n_splits=100, mode='spearman-brown'):
    
    """Computes the split-half reliability, which speaks to the internal
    consistency of the measurement.
    
    Example usage: Say you have a sample of 100 participants, and you assessed
    them with a questionnaire with 20 items that all measure the same
    construct. This data is in a variable 'x' with shape (20, 100). To compute
    the split-half reliability, call:

    r, sem = split_half(x, n_splits=100, mode='spearman-brown')
    
    The variable 'r' tells you the split-half reliability, the variable 'sem'
    reflects the standard error of the mean, computed as the square root of
    the standard deviation of r divided by the square root of the number of
    splits, i.e. sem = sd / sqrt(n_splits)
    
    Arguments
    
    x           -   A NumPy array with shape (M,N), where M is the number of
                    observations and N is the number of participants or tests.
                    M will be split in half to compute the reliability, not N!
    
    Keyword Arguments
    
    n_splits    -   An integer that indicates the number of times you would
                    like to split the data in X. Default value is 100.
    
    mode        -   A string that indicates the type of split-half reliability.
                    You can choose from: 'correlate' or 'spearman-brown'.
                    Default value is 'spearman-brown'.
    
    Returns
    (r, sem)    -   r is the average split-half reliability over n_splits.
                    sem standard error of the mean split-half reliability.
    """
    
    # Check the input.
    if n_splits < 1:
        raise Exception("Expected n_splits to be 1 or more, not '%s'." % \
            (n_splits))
    allowed_modes = ['correlation', 'spearman-brown']
    if mode not in allowed_modes:
        raise Exception("Mode '%s' not supported! Please use a mode from %s" \
            % (mode, allowed_modes))
    
    # Get the number of observations per subject, and the number of subjects.
    n_observations, n_subjects = x.shape
    
    # Compute the size of each group.
    n_half_1 = n_observations//2
    n_half_2 = n_observations - n_half_1
    # Generate a split-half-able vector. Assign the first half 1 and the
    # second half 2.
    halves = numpy.ones((n_observations, n_subjects), dtype=int)
    halves[n_half_1:, :] = 2
    
    # Run through all runs.
    r_ = numpy.zeros(n_splits, dtype=float)
    for i in range(n_splits):

        # Shuffle the split-half vector along the first axis.
        numpy.random.shuffle(halves)

        # Split the data into two groups.
        x_1 = numpy.reshape(x[halves==1], (n_half_1, n_subjects))
        x_2 = numpy.reshape(x[halves==2], (n_half_2, n_subjects))
        
        # Compute the averages for each group.
        m_1 = numpy.mean(x_1, axis=0)
        m_2 = numpy.mean(x_2, axis=0)
        
        # Compute the correlation between the two averages.
        pearson_r, p = pearsonr(m_1, m_2)

        # Store the correlation coefficient.
        if mode == 'correlation':
            r_[i] = pearson_r
        elif mode == 'spearman-brown':
            r_[i] = 2.0 * pearson_r / (1.0 + pearson_r)
    
    # Compute the average R value.
    r = numpy.mean(r_, axis=0)
    # Compute the standard error of the mean of R.
    sem = numpy.std(r_, axis=0) / numpy.sqrt(n_splits)
    
    return r, sem


def test_retest(x, mode='harris'):
    
    """Computes the test-retest reliability.
    
    Arguments
    
    x           -   A NumPy array with shape (M,N) where M is the number of
                    measurements per participant, and N is the number of
                    participants.
    
    Keyword Arguments
    
    mode        -   String indicating the method for computing the rest-retest
                    reliability. Choose from 'fisher' (only whith two
                    measurements per participant!) or 'harris' (works for any
                    number of measurements per participants).
    
    Returns
    
    r           -   Float that is the test-retest reliability.
    """
    
    # Get the number of measurements per participant, and the number of
    # participants.
    n_measurements, n_subjects = x.shape
    
    # Fisher (for two measurements).
    if mode == 'fisher' and n_measurements == 2:
        # Compute the pooled mean.
        m = numpy.sum(x[0,:] + x[1,:]) / (2 * n_subjects)
        # Compute the pooled variance.
        var = (numpy.sum((x[0,:] - m)**2) + numpy.sum((x[1,:] - m)**2)) \
            / (2 * n_subjects)
        # Compute the intraclass correlation according to Fisher.
        r = numpy.sum((x[0,:]-m)*(x[1,:]-m)) / (n_subjects * var)
    
    # Harris for 2 measurements of over.
    elif mode == 'harris':
        # Compute the pooled mean.
        m = numpy.sum(numpy.sum(x, axis=0)) / (n_measurements*n_subjects)
        # Compute the pooled standard deviation.
        var = numpy.sum(numpy.sum((x - m)**2, axis=0)) \
            / (n_measurements * n_subjects)
        # Compute the intraclass correlation according to Harris.
        a = n_measurements / float(n_measurements-1)
        b = numpy.sum((numpy.mean(x, axis=0) - m)**2) / n_subjects
        c = 1.0 / float(n_measurements-1)
        r = a * (b / var) - c
    
    return r
    



if __name__ == "__main__":

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
