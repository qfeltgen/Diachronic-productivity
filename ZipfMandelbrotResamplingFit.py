# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 01:03:08 2025

@author: Quentin Feltgen
"""

# This script provides the function resampling_fit that performs a Zipf-Mandelbrot
# fit of an empirical sample. An example illustrates how to use this function. 

import numpy as np

def ZM_sampler(zipf_coeff,b,N):
    
    #  This function provides a N-sized sample from a Zipf-Mandelbrot
    #  distribution over the types 1, 2, 3, ... , i , ... etc., where  i is
    #  the rank of the type. The probability for the i-ranked  type to be 
    #  sampled is given by pi_i = A / (i + b) ** zipf_coeff. The constant A
    #  is a normalization factor and the parameters b and zipf_coeff are 
    #  to be provided as arguments.
    
    
    x_exp = 1 / (zipf_coeff - 1)
    U = np.random.random(N)
    x = (1 + b) / (1 - U) ** x_exp - b
    x = np.floor(x)
    r_sample = x.astype(np.int64)
    sample = np.array(r_sample)
    
    return sample

def get_values(a,b,N,M):
    
    # For a given sample size N and given Zipf-Mandelbrot parameters a and b,
    # this function returns the mean number of types and its standard deviation,
    # as well as the mean of the top frequency and its standard deviation,
    # over M samples of size N of the Zipf-Mandelbrot distribution.
    
    V_dist = np.zeros(M)
    f1_dist = np.zeros(M)
    
    for km in range(M):
        
        sample = ZM_sampler(a,b,N)
        types, counts = np.unique(sample,return_counts=True)
        V_dist[km] = len(types)
        f1_dist[km] = max(counts)
        
    V_star = np.median(V_dist)
    f1_star = np.median(f1_dist)
    V_err = np.std(V_dist)
    f1_err = np.std(f1_dist)
    
    return V_star, f1_star, V_err, f1_err
   
def resampling_fit(f1,V,N,M=500,max_iter=5000,a=1.5,b=10,epsilon=0.5):
    
    # From the top frequency f1 and the number of types V associated with an 
    # empirical frequency distribution over a sample of N tokens, this function
    # computes the parameters a and b of a Zipf-Mandelbrot distribution that 
    # generates samples consistent with these values.
    
    # Output:
    # a: fitted slope of the Zipf-Mandelbrot distribution
    # b: Mandelbrot parameter of the Zipf-Mandelbrot distribution
    
    # Mandatory parameters:
    # f1: top frequency value in the empirical sample
    # V: number of types in the empirical sample
    # N: size of the empirical samples
    
    # Optional parameters:
    # M: number of samples generated from the candidate Zipf-Mandelbrot distribution
    # over which the resampling distribution for f1 and V is computed
    # max_iter: maximum number of iterations of the a and b values. If the algorithm
    # fails to find appropriate parameter values within this maximum number of
    # iterations, the algorithm stops and an error message is returned.
    # a: initial parameter value for the a parameter
    # b: initial parameter value for the b parameter
    # epsilon: learning rate

    stop = True
    iteration  = 0
    V_star, f1_star, V_err, f1_err = get_values(a,b,N,M)

    while ((np.abs(V - V_star) > V_err) | (np.abs(f1 - f1_star) > f1_err)) & stop :

        a += - epsilon * (V - V_star) / V
        b += - epsilon * (f1 - f1_star) / f1
        b = max(b,0)
        
        V_star, f1_star, V_err, f1_err = get_values(a,b,N,M)

        iteration += 1

        if iteration > 5000:
            stop = False
            print('Maximum number of iterations has been reached')
            
    return a, b

# Example of a parameter fit from a Zipf-Mandelbrot-generated sample

# Underlying Zipf-Mandelbrot parameter values
a_target = 1.74
b_target = 4.8

sample_size = 10000

# Generation of the "empirical" evidence
sample_to_fit = ZM_sampler(a_target,b_target,sample_size)

# Computation of the top frequency and the number of types
types, counts = np.unique(sample_to_fit,return_counts=True)
top_frequency = max(counts)
nb_of_types = len(types)

# Parameter fit using resampling_fit
a_fit, b_fit = resampling_fit(top_frequency,nb_of_types,sample_size)

print('Results of the fit: a = %.2f (target: %.2f); b = %.2f. (target: %.2f)'%(a_fit,a_target,b_fit,b_target))



# Parameter fit using resampling_fit
a_fit, b_fit = resampling_fit(top_frequency,nb_of_types,sample_size,M=1000,epsilon=1)

print('Results of the fit: a = %.2f (target: %.2f); b = %.2f. (target: %.2f)'%(a_fit,a_target,b_fit,b_target))


