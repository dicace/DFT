import numpy as np
import time
from numpy import fft

import matplotlib.pyplot as plt

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

def nextPowerOfTwo(n): 
    i = 0; 
    if (n and not(n & (n - 1))): 
        return n 
      
    while(n != 0): 
        n >>= 1
        i += 1
      
    return 1 << i; 

def improvedDftPrime(arr, n):
      
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
    
    # find next power of two greater than 2 * n - 4
    nP = 2 * n - 3
    nP = nextPowerOfTwo(nP)
    
    # add zeros between zeroth and fisrt element in the first product
    tmp_fft1 = np.zeros(nP, dtype=int)
    tmp_fft1[0] = fft1[0]
    for i in range(nP - n + 2, nP):
        tmp_fft1[i] = fft1[i - nP  + n - 1]
     
    # make the second product for ifft
    fft2 = np.zeros(n-1, dtype=complex)
    j = 0
    for i in g_i:
        fft2[j] = (np.exp(-2j*np.pi*(1/n)*i))
        j += 1

    counter = n - 1
    tmp_fft2 = np.zeros(nP, dtype=complex)
    i = 0
    while (i < nP):
        for j in range(0, n-1):
            if i >= nP: 
                break
            tmp_fft2[i] = fft2[j] 
            i += 1
            
    
    # initialize the result
    A = np.zeros(n, dtype=complex)
    A[0] = np.sum(arr)
    
    # compute the ifft
    inv_dft_arr = fft.ifft(fft.fft(tmp_fft1)*fft.fft(tmp_fft2))

    # populate the result
    for k in range(1, n):
        A[powerModulo(g, k+1, n)] = arr[0] + inv_dft_arr[k-1] 
    return A

# function that compares the computation time between two arrays of prime numbers
# if compare is set to True it is used for comparing dftPrime and improvedDftPrime

def calculateTimeDiff(prime_array, compare=False):
    n = prime_array.shape[0]
    # arrays for storing computation time for Rader's DFT and for fft.fft
    prime_dft = []
    dft = []
    for i in range(n):
        arr = np.random.randint(100, size=prime_array[i])
        start = time.time()
        A = dftPrime(arr, prime_array[i])
        end = time.time()
        # time in s
        prime_dft.append(end - start)

        if (compare == False):
            start = time.time()
            A = fft.fft(arr)
            end = time.time()
            dft.append(end - start)
        else:
            start = time.time()
            A = improvedDftPrime(arr, prime_array[i])
            end = time.time()
            dft.append(end - start)
   
    return prime_dft, dft



##################### TESTING #########################


# compare numpy fft with dftPrime and improvedDftPrime to see if they return the right values

test_array = np.array([3, 2, -1, 0, 6, 4, 1])
print("dftPrime = ", dftPrime(test_array, 7))
print("improvedDftPrime = ", improvedDftPrime(test_array, 7))
print("numpy = ", fft.fft(test_array))

# make an array of non prime numbers for testing computation time difference
# between numpy fft and dftPrime and display it on graph

non_prime_array = np.array([6, 60, 350, 1024, 10000, 25000, 50000])
prime_dft, dft = calculateTimeDiff(non_prime_array)
print("Computation time for Rader's DFT: ", prime_dft)
print("Computation time for DFT: ", dft)

plt.plot(non_prime_array, prime_dft, label='Raderov', color='blue')
plt.plot(non_prime_array, dft, label='numpy', color='red')
plt.xlabel('Broj tacaka')
plt.ylabel('Vreme izvrsavanja u sekundama')
plt.legend()
plt.show()

# make an array of prime numbers for testing computation time difference
# between numpy fft and dftPrime and display it on graph

prime_array = np.array([7, 181, 563, 721, 10007, 10081, 19997, 24989, 34981, 49999, 99991])
prime_dft, dft = calculateTimeDiff(prime_array)
print("Computation time for Rader's DFT: ", prime_dft)
print("Computation time for DFT: ", dft)

plt.plot(prime_array, prime_dft, label='Raderov', color='blue')
plt.plot(prime_array, dft, label='numpy', color='red')
plt.xlabel('Broj tacaka')
plt.ylabel('Vreme izvrsavanja u sekundama')
plt.legend()
plt.show()

# now use array of prime numbers with the restriction that N-1 is not highly composite

prime_array = np.array([7, 563, 10007, 19997, 24989, 34981, 49999])
prime_dft, dft = calculateTimeDiff(prime_array)
print("Computation time for Rader's DFT: ", prime_dft)
print("Computation time for DFT: ", dft)

plt.plot(prime_array, prime_dft, label='Raderov', color='blue')
plt.plot(prime_array, dft, label='numpy', color='red')
plt.xlabel('Broj tacaka')
plt.ylabel('Vreme izvrsavanja u sekundama')
plt.legend()
plt.show()

# try with the restriction that N-1 must be highly composite number

prime_array = np.array([7, 181, 721, 10081, 20161, 45361])
prime_dft, dft = calculateTimeDiff(prime_array)
print("Computation time for Rader's DFT: ", prime_dft)
print("Computation time for DFT: ", dft)

plt.plot(prime_array, prime_dft, label='Raderov', color='blue')
plt.plot(prime_array, dft, label='numpy', color='red')
plt.xlabel('Broj tacaka')
plt.ylabel('Vreme izvrsavanja u sekundama')
plt.legend()
plt.show()

# compare dftPrime and improvedDftPrime when N - 1 is not highly composite

prime_array = np.array([7, 563, 10007, 19997, 24989, 34981, 49999])
prime_dft, dft = calculateTimeDiff(prime_array, True)
print("Computation time for dftPrime: ", prime_dft)
print("Computation time for improvedDftPrime: ", dft)

plt.plot(prime_array, prime_dft, label='Raderov', color='blue')
plt.plot(prime_array, dft, label='Unapredjeni Raderov', color='red')
plt.xlabel('Broj tacaka')
plt.ylabel('Vreme izvrsavanja u sekundama')
plt.legend()
plt.show()
