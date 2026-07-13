# Tensors

## 1. What is a Tensor

```python

Concept

- A tensor is a data type that stores numerical values.

- Tensors are classified based on their RANK (also called DIMENSION or
NUMBER OF AXES).

- rank = dimension = number of axes  -> all mean the same thing

- As rank increases, each new axis wraps the previous structure inside it
(a bunch of matrices stacked -> 3D, a bunch of 3D tensors stacked -> 4D,
and so on).

```
---

## 2. Scalar (0D Tensor)

```python

Concept

- A single value. Has NO axis -> rank 0.

Example: temperature = 36.6

import numpy as np
a = np.array(5)
a.ndim   -> 0
a.shape  -> ()

```
---

## 3. Vector (1D Tensor)

```python

Concept

- A list of values along ONE axis -> rank 1.

- Important: the number of values inside a vector does NOT change its rank.

- A vector with 4 values is still a 1D tensor (rank 1), just with shape (4,).

Example: marks of a student in 4 subjects -> [85, 90, 78, 88]

b = np.array([1, 2, 3, 4])
b.ndim   -> 1
b.shape  -> (4,)

```
---

## 4. Matrix (2D Tensor)

```python

Concept

- Values arranged in rows and columns (two axes) -> rank 2.

Example: marks of 3 students in 4 subjects -> shape (3, 4)
(each row = one student's vector of marks)

c = np.array([[1, 2, 3],
              [4, 5, 6]])
c.ndim   -> 2
c.shape  -> (2, 3)

```
---

## 5. 3D Tensor

```python

Concept

- A stack of matrices (like a cube) -> rank 3.

Example: a color image -> shape (height, width, channels)
e.g. (28, 28, 3) = 28x28 pixels, 3 channels (R, G, B)

d = np.array([[[1, 2], [3, 4]],
              [[5, 6], [7, 8]]])
d.ndim   -> 3
d.shape  -> (2, 2, 2)

```
---

## 6. 4D Tensor

```python

Concept

- A batch of 3D tensors -> rank 4.

Example: a batch of color images
shape = (batch_size, height, width, channels)
e.g. (2, 28, 28, 3) -> 2 color images, each 28x28
This is the standard input format fed into a CNN.

e = np.random.rand(2, 28, 28, 3)
e.ndim   -> 4
e.shape  -> (2, 28, 28, 3)

```
---

## 7. 5D Tensor

```python

Concept

- A batch of 4D tensors -> rank 5.

Example: a batch of videos (sequences of frames)
shape = (batch_size, frames, height, width, channels)
e.g. (4, 2, 28, 28, 3) -> 4 videos, each with 2 frames,
each frame a 28x28 color image

f = np.random.rand(4, 2, 28, 28, 3)
f.ndim   -> 5
f.shape  -> (4, 2, 28, 28, 3)

```
---

## 8. Summary Table

```python

Tensor    : ndim : Shape example        : Real-world example
Scalar    : 0    : ()                   : single number
Vector    : 1    : (4,)                 : list of marks
Matrix    : 2    : (3,4)                : table / grayscale image
3D Tensor : 3    : (28,28,3)            : one color image
4D Tensor : 4    : (2,28,28,3)          : batch of images
5D Tensor : 5    : (4,2,28,28,3)        : batch of videos

```
---

## 9. Important Additions 

```python

1. Shape vs Rank vs Size - three different things 

   - Rank (ndim)  : number of axes            e.g. 2
   - Shape        : size along each axis      e.g. (2, 3)
   - Size         : total number of elements  e.g. 2 * 3 = 6

   x = np.array([[1,2,3],[4,5,6]])
   x.ndim   -> 2
   x.shape  -> (2, 3)
   x.size   -> 6


2. dtype - tensors also have a data type

   Every tensor has a dtype (int32, int64, float32, float64, bool, etc.)
   This matters a lot in deep learning - most frameworks (PyTorch,
   TensorFlow) expect float32 for model inputs, not float64 or int.

   x.dtype  -> dtype('int64')


3. Reshaping - changing shape without changing data

   You can convert one tensor shape into another as long as the total
   number of elements (size) stays the same.

   a = np.array([1,2,3,4,5,6])
   a.reshape(2,3)   -> [[1,2,3],
                        [4,5,6]]

   This is used constantly in ML - e.g. flattening an image (28,28,3)
   into a 1D vector (2352,) before feeding it into a simple neural network.


4. Axis - direction of an operation

   In 2D+ tensors, operations (sum, mean, max) need to know WHICH axis
   to operate along.

   axis=0 -> operate column-wise (down the rows)
   axis=1 -> operate row-wise (across the columns)

   m = np.array([[1,2,3],[4,5,6]])
   m.sum(axis=0)  -> [5,7,9]      (column-wise sum)
   m.sum(axis=1)  -> [6,15]       (row-wise sum)


5. Broadcasting

   NumPy allows operations between tensors of DIFFERENT shapes by
   automatically "stretching" the smaller one, without actually copying
   data. This is why you can add a scalar to a whole matrix, or a
   vector to every row of a matrix, without writing a loop.

   m = np.array([[1,2,3],[4,5,6]])
   m + 10   -> adds 10 to every element (scalar broadcast)

```