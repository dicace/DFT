import numpy as np
from numpy import fft

# check if number is Prime
def isPrime(n):
    if n > 1:
        for i in range(2, n // 2):
            if (n % i) == 0:
                return False
        return True
    else:
        return False

# find prime factors of a number
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

# calculate  x^y%z
def powerModulo(x, y, z):
    result = 1
    x = x % z
    while y > 0:
        if (y & 1):
            result = (result * x) % z
            
        y = y >> 1
        x = (x * x) % z
    return result

# find smallest primitive root of a number
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
            if (powerModulo(i, phi // j, n) == 1):
                flag = True
                break
        if flag == False:
            return i
    
    return -1


# compute the DFT
def dftPrime(arr, n):

    # find smallest primitive
    g = smallestPrimitive(n)
    
    # make an array of g^i
    g_i = np.zeros(n-1, dtype=int) 
    for i in range(0, n-1):
        g_i[i] = powerModulo(g, i+1, n)
    

    # make an array of g^(-i)
    g_minus_i = np.zeros(n-1, dtype=int) 
    for i in range(0, n-1):
        g_minus_i[i] = powerModulo(g, n-i-2, n)
    
    # make the first product for ifft   
    fft1 = arr[g_minus_i]
 
    # make the second product for ifft
    fft2 = []
    for i in g_i:
        fft2.append(np.exp(-2j*np.pi*(1/n)*i))

    # initialize the result
    A = np.zeros(n, dtype=complex)
    A[0] = np.sum(arr)
    
    # compute the ifft
    inv_dft_arr = fft.ifft(fft.fft(fft1)*fft.fft(fft2))

    # populate the result
    for k in range(1, n):
        A[powerModulo(g, k+1, n)] = arr[0] + inv_dft_arr[k-1] 
    return A



# test

arr = np.array([3, 2, 1, -3, 0, 4, 6])
A = dftPrime(arr, 7)
print(A)
print(fft.fft(arr))
print()

arr = np.array([3, 2, 1, -3, 0])
A = dftPrime(arr, 5)
print(A)
print(fft.fft(arr))
print()

arr = np.array([3, 2, 1, -3, 0, 4, 6, 13, -2, 0, 4])
A = dftPrime(arr, 11)
print(A)
print(fft.fft(arr))

