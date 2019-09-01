
# coding: utf-8

import numpy as np
from numpy import fft

def isPrime(n):
    if n > 1:
        for i in range(2, n // 2):
            if (n % i) == 0:
                return False
        return True
    else:
        return False
    
def primeFactors(factors, n):
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n = n // i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def modulo(x, y, z):
    result = 1
    x = x % z
    while y > 0:
        if (y & 1):
            result = (result * x) % z
            
        y = y >> 1
        x = (x * x) % z
    return result

def smallestPrimitive(n):
    factors = []
    
    if (isPrime(n) == False):
        return -1
    
    #using Euler function
    phi = n - 1
    primeFactors(factors, phi)
    
    for i in range(2, phi + 1):
        flag = False
        for j in factors:
            if (modulo(i, phi // j, n) == 1):
                flag = True
                break
        if flag == False:
            return i
    
    return -1


def dftPrime(arr, n):

    # find smallest primitive
    g = smallestPrimitive(n)
    
    # make an array of g^i
    g_arr = np.zeros(n, dtype=int) 
    for i in range(0, n):
        g_arr[i] = modulo(g, i, n)
    
    # make the second product (change comment)
    fft2 = []
    f = np.exp(-2j*np.pi*(1/n))
    for i in g_arr:
        fft2.append(f*i)

    # make the first product   
    #TO DO: change indices for - 1
    fft1 = np.zeros(n, dtype=int)
    fft1 = arr[g_arr]
    
    # initialize the result
    A = np.zeros(n, dtype=complex)
    A[0] = np.sum(arr)
    
    # do the fft
    inv_dft_arr = fft.ifft(fft.fft(fft1)*fft.fft(fft2))
    
    # populate the result
    for k in range(1, n):
        A[modulo(g, k, n)] = arr[0] + inv_dft_arr[k] 
    return A

# test
arr = np.array([3, 2, 1, -3, 0, 4, 6])
dftPrime(arr, 7)

