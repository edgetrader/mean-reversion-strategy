## REFERENCE
## https://medium.com/bluekiri/simple-stationarity-tests-on-time-series-ad227e2e6d48
## https://www.quantstart.com/articles/Basics-of-Statistical-Mean-Reversion-Testing/

import numpy as np

from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.tsatools import lagmat, add_trend
from statsmodels.tsa.adfvalues import mackinnonp
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint

import statsmodels.api as sm
from arch.unitroot import VarianceRatio

def adf(ts):
    """
    Augmented Dickey-Fuller unit root test
    """
    # make sure we are working with an array, convert if necessary
    ts = np.asarray(ts)
    
    # Get the dimension of the array
    nobs = ts.shape[0]
    
    # We use 1 as maximum lag in our calculations
    maxlag = 1
    
    # Calculate the discrete difference
    tsdiff = np.diff(ts)
    
    # Create a 2d array of lags, trim invalid observations on both sides
    tsdall = lagmat(tsdiff[:, None], maxlag, trim='both', original='in')
    # Get dimension of the array
    nobs = tsdall.shape[0] 
    
    # replace 0 xdiff with level of x
    tsdall[:, 0] = ts[-nobs - 1:-1]  
    tsdshort = tsdiff[-nobs:]
    
    # Calculate the linear regression using an ordinary least squares model    
    results = OLS(tsdshort, add_trend(tsdall[:, :maxlag + 1], 'c')).fit()
    adfstat = results.tvalues[0]
    
    # Get approx p-value from a precomputed table (from stattools)
    pvalue = mackinnonp(adfstat, 'c', N=1)
    return pvalue

def cadf(x, y):
    """
    Returns the result of the Cointegrated Augmented Dickey-Fuller Test
    """
    # Calculate the linear regression between the two time series
    ols_result = OLS(x, y).fit()
     
    # Augmented Dickey-Fuller unit root test
    return adf(ols_result.resid)

def hurst(ts):
    """
    Returns the Hurst Exponent of the time series vector ts
    """
    # make sure we are working with an array, convert if necessary
    ts = np.asarray(ts)

    # Helper variables used during calculations
    lagvec = []
    tau = []
    # Create the range of lag values
    lags = range(2, 100)

    #  Step through the different lags
    for lag in lags:
        #  produce value difference with lag
        pdiff = np.subtract(ts[lag:],ts[:-lag])
        #  Write the different lags into a vector
        lagvec.append(lag)
        #  Calculate the variance of the difference vector
        tau.append(np.sqrt(np.std(pdiff)))

    #  linear fit to double-log graph
    m = np.polyfit(np.log10(np.asarray(lagvec)),
                   np.log10(np.asarray(tau).clip(min=0.0000000001)),
                   1)
    # return the calculated hurst exponent
    return m[0]*2.0

def variance_ratio(ts, lag = 2):
    """
    Returns the variance ratio test result
    """
    # make sure we are working with an array, convert if necessary
    ts = np.asarray(ts)
    
    # Apply the formula to calculate the test
    n = len(ts)
    mu  = sum(ts[1:n]-ts[:n-1])/n;
    m=(n-lag+1)*(1-lag/n);
    b=sum(np.square(ts[1:n]-ts[:n-1]-mu))/(n-1)
    t=sum(np.square(ts[lag:n]-ts[:n-lag]-lag*mu))/m
    return t/(lag*b)

def half_life(ts):  
    """ 
    Calculates the half life of a mean reversion
    """
    # make sure we are working with an array, convert if necessary
    ts = np.asarray(ts)
    
    # delta = p(t) - p(t-1)
    delta_ts = np.diff(ts)
    
    # calculate the vector of lagged values. lag = 1
    # lag_ts = np.vstack([ts[1:], np.ones(len(ts[1:]))]).T
    lag_ts = np.vstack([ts[:-1], np.ones(len(ts[:-1]))]).T
   
    # calculate the slope of the deltas vs the lagged values 
    beta = np.linalg.lstsq(lag_ts, delta_ts)
    
    # compute and return half life
    # return (np.log(2) / beta[0])[0]
    return -(np.log(2) / beta[0])[0]


def half_life_v2(series):  
    ts = series.to_frame()
    ts['lag'] = series.shift(1)
    ts['ret'] = series - ts.lag
    ts = sm.add_constant(ts)
    ts.dropna(inplace=True)

    model = sm.OLS(ts.ret, ts[['const','lag']])
    res = model.fit()

    halflife = - np.log(2) / res.params[1]
    return halflife

def johansen(ts, lags):
    """
    Calculate the Johansen Test for the given time series
    """
    # Make sure we are working with arrays, convert if necessary
    ts = np.asarray(ts)
 
    # nobs is the number of observations
    # m is the number of series
    nobs, m = ts.shape
 
    # Obtain the cointegrating vectors and corresponding eigenvalues
    eigenvectors, eigenvalues = maximum_likelihood_estimation(ts, lags)
 
    # Build table of critical values for the hypothesis test
    critical_values_string = """2.71   3.84    6.63
             13.43   15.50   19.94
             27.07   29.80   35.46
             44.49   47.86   54.68
             65.82   69.82   77.82
             91.11   95.75   104.96
             120.37  125.61  135.97
             153.63  159.53  171.09
             190.88  197.37  210.06
             232.11  239.25  253.24
             277.38  285.14  300.29
             326.53  334.98  351.25"""
    select_critical_values = np.array(
            critical_values_string.split(),
            float).reshape(-1, 3)
    critical_values = select_critical_values[:, 1]
 
    # Calculate numbers of cointegrating relations for which
    # the null hypothesis is rejected
    rejected_r_values = []
    for r in range(m):
        if h_test(eigenvalues, r, nobs, lags, critical_values):
            rejected_r_values.append(r)
 
    return eigenvectors, rejected_r_values


def h_test(eigenvalues, r, nobs, lags, critical_values):
    """
    Helper to execute the hypothesis test
    """
    # Calculate statistic
    t = nobs - lags - 1
    m = len(eigenvalues)
    statistic = -t * np.sum(np.log(np.ones(m) - eigenvalues)[r:])
 
    # Get the critical value
    critical_value = critical_values[m - r - 1]
 
    # Finally, check the hypothesis
    if statistic > critical_value:
        return True
    else:
        return False

def maximum_likelihood_estimation(ts, lags):
    """
    Obtain the cointegrating vectors and corresponding eigenvalues
    """
    # Make sure we are working with array, convert if necessary
    ts = np.asarray(ts)
 
    # Calculate the differences of ts
    ts_diff = np.diff(ts, axis=0)
 
    # Calculate lags of ts_diff.
    ts_diff_lags = lagmat(ts_diff, lags, trim='both')
 
    # First lag of ts
    ts_lag = lagmat(ts, 1, trim='both')
 
    # Trim ts_diff and ts_lag
    ts_diff = ts_diff[lags:]
    ts_lag = ts_lag[lags:]
 
    # Include intercept in the regressions
    ones = np.ones((ts_diff_lags.shape[0], 1))
    ts_diff_lags = np.append(ts_diff_lags, ones, axis=1)
 
    # Calculate the residuals of the regressions of diffs and lags
    # into ts_diff_lags
    inverse = np.linalg.pinv(ts_diff_lags)
    u = ts_diff - np.dot(ts_diff_lags, np.dot(inverse, ts_diff))
    v = ts_lag - np.dot(ts_diff_lags, np.dot(inverse, ts_lag))
 
    # Covariance matrices of the calculated residuals
    t = ts_diff_lags.shape[0]
    Svv = np.dot(v.T, v) / t
    Suu = np.dot(u.T, u) / t
    Suv = np.dot(u.T, v) / t
    Svu = Suv.T
 
    # ToDo: check for singular matrices and exit
    Svv_inv = np.linalg.inv(Svv)
    Suu_inv = np.linalg.inv(Suu)
 
    # Calculate eigenvalues and eigenvectors of the product of covariances
    cov_prod = np.dot(Svv_inv, np.dot(Svu, np.dot(Suu_inv, Suv)))
    eigenvalues, eigenvectors = np.linalg.eig(cov_prod)
 
    # Use Cholesky decomposition on eigenvectors
    evec_Svv_evec = np.dot(eigenvectors.T, np.dot(Svv, eigenvectors))
    cholesky_factor = np.linalg.cholesky(evec_Svv_evec)
    eigenvectors = np.dot(eigenvectors, np.linalg.inv(cholesky_factor.T))
 
    # Order the eigenvalues and eigenvectors
    indices_ordered = np.argsort(eigenvalues)
    indices_ordered = np.flipud(indices_ordered)
 
    # Return the calculated values
    return eigenvalues[indices_ordered], eigenvectors[:, indices_ordered]    

def perform_adf_test(series, notes=False):
    """
    Augmented Dickey-Fuller Test
    Test whether a time series is stationary.  
    Stationary means the mean and standard deviation do not change over time.
    Returns tuples of ADF Statistics, p-value and True if series is stationary.
    """
    result = adfuller(series)
    if notes: 
        print('ADF Statistics: {:.3f}'.format(result[0]))    
        print('p-value: {:.3f}'.format(result[1]))
        if result[1] < 0.05:
            print('Reject null hypothesis. Series is stationary because p-value is less than 0.05.')
        else:
            print('Do not reject null hypothesis. Series is not stationary because p-value is greater than 0.05.')
    else:
        return result[0], result[1], result[1] < 0.05

def perform_hurst_exp_test(series, notes=False):
    """
    Computing Hurst Exponent
    Random Walk has a hurst exponent, H = 0.5 
    Mean Reversion has a hurst exponent, H < 0.5 
    Trending has a hurst exponent, H > 0.5 
    Returns the Hurst Exponent and whether the Time Series is mean reverting
    """
    result = hurst(series)
    
    if notes:
        print('Hurst Exponent: {:.3f}'.format(result))
        if result < 0.5:
            print('Mean Reverting')
        elif result > 0.5:
            print('Trending')
        else:
            print('Random Walk')
    else:
        return result, result < 0.5     


def perform_variance_ratio_test(series, lag=2, notes=False):
    """
    The null hypothesis of a VR is that the process is a random walk, possibly plus drift. 
    Rejection of the null with a positive test statistic indicates the presence of positive 
    serial correlation in the time series.
    Return p-value.  Less than 0.05, reject null hypothesis.  It is not a random walk.
    """
    result = VarianceRatio(series, lag).pvalue
    
    if notes:
        print('Variance Ratio: {:.3f}'.format(result))
        if result < 0.05:
            print('Not a random walk')
        else:
            print('Random Walk')
    else:
        return result, result < 0.05           


def perform_coint_test(ts1, ts2, notes=False):
    """
    Cointegration Test
    Time Series pair is cointegrated if p-value is less than 0.05
    Returns p-value and whether the pair is cointegrated
    """
    results = coint(ts1, ts2)
    if notes:
        if results[1] < 0.05:
            print('Cointegrated')
        else:
            print('Not cointegrated')
    else:
        return results[1], results[1] < 0.05, results[0]
            