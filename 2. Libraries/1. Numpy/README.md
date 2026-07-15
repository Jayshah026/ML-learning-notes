# NumPy Array Creation

## Importing NumPy
```python

- Use case: NumPy is used for fast numerical computing in Python.

- It gives us the 'array' object (ndarray) which is much faster and more memory-efficient than a normal Python list.

import numpy as np (To import and use the Numpy we write this line)

```
---

## np.array()
```python

- Use case: Create an array (1D, 2D, or n-D) from a Python list.

- This is the most basic way to make a NumPy array.

arr1 = np.array([1, 2, 3, 4, 5])
print(arr1)
OUTPUT : array([1, 2, 3, 4, 5])

print(type(arr1))
OUTPUT : numpy.ndarray

```
---

## Creating a 2D array
```python

- Use case: Pass a list of lists to np.array() to create a matrix (rows and columns).

- Useful for representing tables, images, etc.

arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2)

OUTPUT :
array([[1, 2, 3],
       [4, 5, 6]])
```
---

## np.zeros()
```python

- Use case: Create an array filled entirely with 0s.

- Commonly used to initialize an array/matrix before filling it with real values later (e.g., placeholder for weights in ML).

arr3 = np.zeros((2, 2))
print(arr3)

OUTPUT :
array([[0., 0.],
       [0., 0.]])

```
---

## np.ones()
```python

- Use case: Create an array filled entirely with 1s.

- Useful for initializing bias values, masks, or default weights.

arr4 = np.ones((10, 10))
print(arr4)

OUTPUT : a 10x10 matrix where every element is 1.

```
---

## np.identity()
```python

- Use case: Create an identity matrix (1s on the diagonal, 0s elsewhere).

- Identity matrices are used a lot in linear algebra, e.g., multiplying by identity gives the same matrix back.

arr5 = np.identity(8)
print(arr5)

OUTPUT : 8x8 matrix with 1s on the diagonal, 0s everywhere else.

```
---

## np.arange()
```python

- Use case: Create an array of numbers within a range, similar to Python's range(), but returns a NumPy array.

- The end value is excluded.

arr6 = np.arange(11)
print(arr6)

OUTPUT : array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

```
---

## np.arange() with start and stop

```python

- Use case: np.arange(start, stop) starts from 'start' and ends before 'stop'.

arr7 = np.arange(1, 6)
print(arr7)

OUTPUT : array([1, 2, 3, 4, 5])

```
---

## np.arange() with step
```python

- Use case: np.arange(start, stop, step) jumps by 'step' each time.

arr8 = np.arange(0, 21, 2)
print(arr8)

OUTPUT : array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

```
---

## .copy()
```python

- Use case: Create a completely independent copy of an existing array.

- Important because normal assignment (arr9 = arr6) only creates a reference, not a real copy. .copy() avoids that problem.

arr9 = arr6.copy()
print(arr9)

OUTPUT : array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

```
---

## np.linspace()
```python

- Use case: Create 'n' evenly spaced numbers between a start and end value (end value is included).

- Useful when you need a fixed number of points spread out equally, e.g., for plotting graphs.

arr10 = np.linspace(1, 10, 5)
print(arr10)

OUTPUT : array([ 1.  ,  3.25,  5.5 ,  7.75, 10.  ])

```
---


# NumPy Array Attributes


## .shape
```python

- Use case: Tells you the dimensions of an array — how many elements are along each axis (rows, columns, depth, etc.).

- Returns a tuple, e.g. (rows, columns) for a 2D array.

arr11 = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
arr11.shape
OUTPUT : (4, 3)

arr12 = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
arr12.shape
OUTPUT : (2, 2, 2)
```
---

## .ndim
```python

- Use case: Tells you the number of dimensions (axes) an array has.

- A 1D array has ndim 1, a 2D array (matrix) has ndim 2, a 3D array has ndim 3, and so on.

arr11.ndim
OUTPUT : 2

arr12.ndim
OUTPUT : 3

arr13 = np.array([[[[1,2,3,4,5],[6,7,8,9,10]]], [[[42,23,42,55,64],[42,12,122,54,52]]]])
arr13.ndim
OUTPUT : 4

```
---

## .size
```python

- Use case: Tells you the total number of elements present in the array (multiplies all the dimensions together).

arr13.size
OUTPUT : 20

```
---

## .itemsize
```python

- Use case: Tells you the memory (in bytes) that a single element in the array takes up.

- Depends on the dtype (e.g. int64 takes 8 bytes, float64 takes 8 bytes).

arr13.itemsize
OUTPUT : 8

arr11.itemsize
OUTPUT : 8

```
---

## .dtype
```python

- Use case: Tells you the data type of the elements stored in the array (e.g. int64, float64).

- Useful to check before doing operations, since mixing dtypes can cause unexpected results or extra memory usage.

arr13.dtype
OUTPUT : dtype('int64')

arr11.dtype
OUTPUT : dtype('int64')

arr10.dtype
OUTPUT : dtype('float64')

arr4.dtype
OUTPUT : dtype('float64')

```
---

## .astype()
```python

- Use case: Creates a new array with the elements converted to a different data type.

- Does not change the original array — it returns a new one with the converted dtype.

arr4.astype('int')
OUTPUT :
array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

```
---


# NumPy Array Operations

## Element-wise Addition
```python

- Use case: Adding two arrays adds their elements position by position (both arrays must be the same shape).

- This is much faster than looping through lists manually in plain Python.

arr1 = np.array([1,2,3,4,5])
arr2 = np.array([6,7,8,9,10])

arr1 + arr2

OUTPUT : array([ 7,  9, 11, 13, 15])

```
---

## Element-wise Subtraction

```python

- Use case: Subtracts the elements of the second array from the first, position by position.

arr1 - arr2

OUTPUT : array([-5, -5, -5, -5, -5])

```
---

## Element-wise Multiplication

```python

- Use case: Multiplies the elements of both arrays position by position.

- Note: this is NOT matrix multiplication, just simple element-by-element multiplication.

arr1 * arr2

OUTPUT : array([ 6, 14, 24, 36, 50])

```
---

## Element-wise Division

```python

- Use case: Divides the elements of the first array by the elements of the second, position by position.

arr1 / arr2

OUTPUT : array([0.16666667, 0.28571429, 0.375, 0.44444444, 0.5])

```
---

## Scalar Broadcasting

```python

- Use case: When you perform an operation between an array and a single number (scalar), NumPy applies that operation to every element of the array. This is called broadcasting.

arr2 + 5

OUTPUT : array([11, 12, 13, 14, 15])

```
---

## Comparison between two arrays

```python

- Use case: Comparing two arrays checks each pair of elements position by position and returns a boolean array (True/False) for the result.

arr1 > arr2

OUTPUT : array([False, False, False, False, False])

```
---

## Comparison with a scalar

```python

- Use case: Comparing an array with a single number checks every element against that number and returns a boolean array.

arr2 > 7

OUTPUT : array([False, False, True, True, True])

```
---
## .max() and .min()

```python

- Use case: .max() returns the largest value in the array, .min() returns the smallest value.

- Useful for quickly finding extremes in data without writing a loop.

arr2.max()
OUTPUT : np.int64(10)

arr2.min()
OUTPUT : np.int64(6)

```
---

## .max(axis) and .min(axis) on 2D arrays

```python

- Use case: When used on a 2D array, the axis parameter controls the direction of the operation.

- axis=0 -> works column-wise (compares values going down each column).

- axis=1 -> works row-wise (compares values going across each row).

arr3 = np.array([[1,2,3], [4,5,6], [7,8,9]])
arr3

OUTPUT :
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])


arr3.max(axis=0)

OUTPUT : array([7, 8, 9])

arr3.min(axis=1)

OUTPUT : array([1, 4, 7])

```
---
## .sum()

```python

- Use case: Adds up all the elements in the array and returns a single total.

- Very handy for quick totals without writing a manual loop.

arr3.sum()
OUTPUT : np.int64(45)

arr3.min()
OUTPUT : np.int64(1)
```
---

