# %% [markdown]
# # In this note book we're going to cover some of the most fundamental concepts of tensors using TensorFlow
# More specifically, we're going to cover:
# * introduction to tensors
# * Getting information from tensors
# * Manipulating tensors
# * Tensors & Numpy
# * Using @tf.function (a way to speed up regular Python functions)
# * Using GPUs with TensorFlow (or TPUs)
# * Exercises to try myself

# %% [markdown]
# ## Introduction to Tensors

# %%
import tensorflow as tf
print(tf.__version__)

# %%
# Create tensors with tf.constant()
scalar = tf.constant(7)
scalar

# %%
# Check number of dims of a tensor (ndim = number of dimensions)
scalar.ndim

# %%
# Create a vector
vector = tf.constant([10, 10])
vector

# %%
# Dim of vector
vector.ndim

# %%
# create a matrix
matrix = tf.constant([[10, 7], [7, 10]])
matrix

# %%
# Dim of matrix
matrix.ndim

# %%
# Another matrix
another_matrix = tf.constant([[10., 7.], 
    [3., 2.], 
    [8., 9.]], 
    dtype=tf.float16)
another_matrix

# %%
# Dim of another_matrix
another_matrix.ndim

# %%
# Create a tensor
tensor = tf.constant(
      [[[1, 2, 3],
        [4, 5, 6]],
       [[7, 8, 9],
        [10, 11, 12]],
       [[13, 14, 15],
        [16, 17, 18]]])
tensor

# %%
tensor.ndim

# %% [markdown]
# So far:
# * Scalar: a single number
# * Vector: a number with direction (e.g. wind speed and direction)
# * Matrix: a 2-d array of numbers
# * Tensor: a n-d array of numbers 

# %% [markdown]
# ### Creating tensors with tf.variable

# %%
# Create the same tensors with tf.Variable() as above
changeable_tensor = tf.Variable([10, 7])
unchangeable_tensor = tf.constant([10, 7])
changeable_tensor, unchangeable_tensor

# %%
# Try to change elements
changeable_tensor[0] = 7
changeable_tensor

# %%
# try .assign()
changeable_tensor[0].assign(7)
changeable_tensor

# %%
# try at unchangeable tensor
unchangeable_tensor[0].assign(7)
unchangeable_tensor

# %% [markdown]
# ***
# 🔑 **Note:** Rarely in practive i will need to decide whether to use `tf.constant` or `tf.Variable` to create tensors, as TensorFlow does this for me. However, if in doubt, i'm going to use `tf.constant` and change it later if needed.
# ***

# %% [markdown]
# ### Creating random tensors
# Random tensors are tensors of some arbitrary size which contain random numbers

# %%
# Create to random (but the same) tensors
random_1 = tf.random.Generator.from_seed(42) # set seed for reproducibility
random_1 = random_1.normal(shape=(3, 2))
random_2 = tf.random.Generator.from_seed(42)
random_2 = random_2.normal(shape=(3, 2))

# Are they equal?
random_1, random_2, random_1 == random_2

# %% [markdown]
# ### Shuffle the order of the tensor's elements

# %%
# Shuffle a tensor (valuable for when i want to shuffle my data so the inherent order doesn't effect learning)
not_shuffled = tf.constant([[10, 7],
                        [3, 4],
                        [2, 5]])

tf.random.shuffle(not_shuffled)

# %%
not_shuffled

# %%
tf.random.set_seed(42)
tf.random.shuffle(not_shuffled, seed=42)

# %% [markdown]
# ***
# ⚒️ **Exercise:** Read through TensorFlow documentation on random seed generation and practice writing 5 random tensors and shuffle them
# ***
# It looks like if we want out shuffled tensors to be in the same order, we've got to use the global level random seed as well as the operation level random seed:
# 
# > Rule 4: If both the global and the operation seed are set: Both seeds are used in conjunction to determine the random sequence.

# %%
tf.random.set_seed(42) # Global level random seed
tf.random.shuffle(not_shuffled, seed=42) # operation level random seed

# %% [markdown]
# ### Other ways to make tensors

# %%
# Create a tensor of all ones
tf.ones([10, 7])

# %%
# Create a tensor of all zeros
tf.zeros(shape=(3, 4))

# %% [markdown]
# ### Turn NumPy arrays into tensors
# The main difference between NumPy arrays and TensorFlow tensors is that tensors can be run on a GPU (much faster for numerical computing)

# %%
# You can also NumPy arrays into tensors
import numpy as np
numpy_A = np.arange(1, 25, dtype=np.int32) # create a NumPy array between 1 and 25
numpy_A

# X = tf.constant(matrix) # capital for matrix or tensor
# y = tf.constant(vector) # non-capital for vector

# %%
A = tf.constant(numpy_A, shape=(2, 3, 4))
B = tf.constant(numpy_A)
A, B

# %% [markdown]
# ### Getting information from tensors
# When dealing with tensors i probably want to be aware of the following attributes
# * Shape
# * Rank
# * Axis or dimension
# * Size 

# %%
# Create a rank 4 tensor
rank_4_tensor = tf.zeros(shape=(2, 3, 4, 5))
rank_4_tensor

# %%
rank_4_tensor[0]

# %%
rank_4_tensor.shape, rank_4_tensor.ndim, tf.size(rank_4_tensor)

# %%
# Get various attributes of our tensor
print("Datatype of every element: ", rank_4_tensor.dtype)
print("Number of dimensions: ", rank_4_tensor.ndim)
print("Shape of tensor: ", rank_4_tensor.shape)
print("Elements along the 0 axis: ", rank_4_tensor.shape[0])
print("Elements along the last axis: ", rank_4_tensor.shape[-1])
print("Total number of elements in tensor: ", tf.size(rank_4_tensor).numpy())

# %% [markdown]
# ### Indexing tensors
# Tensors can be indexed just like Python lists.

# %%
some_list = [1, 2, 3, 4]
some_list[:2]

# %%
# Get the first 2 elements of each dimension
rank_4_tensor[:2, :2, :2, :2]

# %%
some_list[1: -1]

# %%
# Get the first element from each dimension from each index except for the final one
rank_4_tensor[:1, :1, :1, :]

# %%
# Create a rank 2 tensor (2 dimensions)
rank_2_tensor = tf.constant([
    [10, 7],
    [3, 4]])
rank_2_tensor.shape, rank_2_tensor.ndim

# %%
# Get the last item of each of our rank 2 tensor
rank_2_tensor[:,-1]

# %%
# Add extra dimension to our rank 2 tensor
rank_3_tensor = rank_2_tensor[..., tf.newaxis] # :, :, -> ... every axis like before
rank_3_tensor

# %%
# Alternative to tf.newaxis
tf.expand_dims(rank_2_tensor, axis=-1) # "-1" means expand the final axis

# %%
tf.expand_dims(rank_2_tensor, axis=0) # expand the 0-axis

# %% [markdown]
# ### Manipulating tensors (tensor operations)
# **Basic operations**
# 
# `+`,`-`,`*`,`/`

# %%
# You can add values to a tensor using the addtition operator
tensor = tf.constant(
    [[10, 7], 
    [3, 4]])
tensor + 10

# %%
# Original tensor is unchanged
tensor

# %%
# Multiplikation
tensor * 10

# %%
# We can use the tensorflow built-in function too
tf.multiply(tensor, 10)

# %%
# Substraction
tensor - 10

# %% [markdown]
# ***
# ⭐ If you want your code to speed up, use the tensorflow version eg `tf.multiply()`, `tf.add()` 
# ***

# %% [markdown]
# **Matrix multiplication**
# 
# In machine learning, matrix multiplication is one of the most common tensor operations

# %%
# Matrix multiplication in tensorflow
print(tensor)
tf.matmul(tensor, tensor)

# %%
tensor * tensor # elementwise multiplication

# %%
tensor_1 = tf.constant([
    [1, 2, 5],
    [7, 2, 1],
    [3, 3, 3]])
tensor_2 = tf.constant([
    [3, 5],
    [6, 7],
    [1, 8]])
tensor_1, tensor_2

# %%
# Matrix multiplication with Python operator "@"
tensor_1 @ tensor_2

# %%
X = tf.constant([
    [1, 2],
    [3, 4],
    [5, 6]])
Y = tf.constant([
    [7, 8],
    [9, 10],
    [11, 12]])
X, Y

# %%
# X @/matmul Y won't work because of the dimensions: [3, 2] @ [3, 2]

# %% [markdown]
# ***
# 📖 **Resource:** Info and example of matrix multiplication https://www.mathsisfun.com/algebra/matrix-multiplying.html
# ***

# %%
Y, tf.transpose(Y), tf.reshape(Y, shape=(2, 3))

# %%
X @ tf.transpose(Y), X @ tf.reshape(Y, shape=(2,3)), tf.matmul(X, tf.reshape(Y, shape=(2,3)))

# %%
tf.matmul(tf.reshape(X, shape=(2, 3)), Y)

# %% [markdown]
# **The dot product**
# 
# Matrix multiplication is also referred to as the dot product.
# 
# You can perform matrix multiplication using:
# * `tf.matmul()`
# * `tf.tensordot()`

# %%
X, Y

# %%
# Perform the dot product on X and Y (requires X or Y to be transposed)
tf.tensordot(X, tf.transpose(Y), axes=1)

# %%
# Perform matrix multiplication between X and Y (transposed)
tf.matmul(tf.transpose(X), Y)

# %%
# Perform matrix multiplication between X and y (reshaped)
tf.matmul(tf.reshape(X, shape=(2, 3)), Y)

# %%
# Check values of X, reshape X and transposed X
print("Normal X: \n", X.numpy(), "\n")
print("Reshaped X: \n", tf.reshape(X, shape=(2,3)).numpy(), "\n")
print("Transposed X: \n", tf.transpose(X).numpy(), "\n")

# %% [markdown]
# Generally, when performing matrix multiplacation on two tensor and one of dthe axes doesn't line up, you will transpose rather than reshape one of the tensors to satisfy the matrix multiplation rules.

# %% [markdown]
# ### Changing the datatype of a tensor

# %%
# Create a new tensor with default datatype (float32)
B = tf.constant([1.7, 7.4])
B, B.dtype

# %%
C = tf.constant([7, 10])
C, C.dtype

# %%
# Change from float32 to float16 (reduced precision)
D = tf.cast(B, dtype=tf.float16)
D, D.dtype

# %%
# int32 --> float 32
E = tf.cast(C, dtype=tf.float32)
E, E.dtype

# %% [markdown]
# ### Aggregating tensors
# 
# Aggregatin tensors = condensing them from multiple values down to a smaller amount of values.

# %%
D = tf.constant([-7, -10])
D

# %%
# Get theb absolute values
tf.abs(D)

# %% [markdown]
# Let's go through the following forms of aggregation:
# * Get the minimum
# * Get the maximum
# * Get the mean of a tensor
# * Get the sum of a tensor

# %%
# Create a random tensor with values between 0 and 100 of size 50
E = tf.constant(np.random.randint(0, 100, 50))
E

# %%
tf.size(E), E.shape, E.ndim

# %%
# Find the minimum
tf.reduce_min(E)

# %%
# Find the maximum
tf.reduce_max(E)

# %%
# Find the mean
tf.reduce_mean(E)

# %%
# Find the sum
tf.reduce_sum(E)

# %% [markdown]
# ***
# ⚒️ **Exercise:** With what we've learned, find the variance and standard deviation of our `E` tensor using TensorFlow methods.
# ***

# %%
# To find the variance of out tensor, we need access to tensorflow_probability
import tensorflow_probability as tfp
tfp.stats.variance(E)

# %%
tf.math.reduce_variance(tf.cast(E, dtype=tf.float32))

# %%
# Standard deviation
tf.math.reduce_std(tf.cast(E, dtype=tf.float32))

# %% [markdown]
# ### Finde the positional maximum and minimum

# %%
E,tf.math.argmin(E), tf.math.argmax(E)

# %%
E[tf.math.argmin(E)], E[tf.math.argmax(E)]

# %%
tf.reduce_min(E), tf.reduce_max(E)

# %% [markdown]
# ### Squeezin a tensor (removing all single dimensions)

# %%
# Create a tensor to get started
tf.random.set_seed(42)
G = tf.constant(tf.random.uniform(shape=[50]), shape=(1, 1, 1, 1, 50))
G

# %%
G.shape

# %%
G_squeezed = tf.squeeze(G)
G_squeezed, G_squeezed.shape

# %% [markdown]
# ### One-hot encoding tensors

# %%
# Create a list of indices
some_list = [0, 1, 2, 3] # could be red, green, blue, purple

# One hot encode our list of indices
tf.one_hot(some_list, 4)

# %%
# Specify custom values for one hot encoding
tf.one_hot(some_list, depth=4, on_value="foo", off_value="bar")

# %% [markdown]
# ### Squarin, log, square root

# %%
# Create a tensor
H = tf.range(1, 10)
H 

# %%
# Find the square
tf.square(H)

# %%
# Find the square root
tf.sqrt(tf.cast(H, dtype=tf.float32))

# %%
# Find log
tf.math.log(tf.cast(H, dtype=tf.float32))

# %% [markdown]
# ### Tensors and NumPy
# 
# Tensorflow interacts beautifully with NumPy arrays.
# 
# >🔑 **Note:** One of the main differences between a TensorFlow and a NumPy array is that a TensorFlow tensor can be run on a GPU or TPU (fo faster numerical processing)

# %%
# Create a tensor directly from a NumPy array
J = tf.constant(np.array([3., 7.]))
J

# %%
# Covnert tensor back to NumPy array
np.array(J), type(np.array(J))

# %%
# Convert tensor J to NumPy array
J.numpy(), type(J.numpy())

# %%
# Default types of each are slightly different
numpy_J = tf.constant(np.array([3., 7., 10.]))
tensor_J = tf.constant([3., 7., 10.])
# Check datatypes
numpy_J.dtype, tensor_J.dtype

# %% [markdown]
# ### Finding access to GPUs

# %%
# Check tensorflow GPU usage
tf.config.list_physical_devices("GPU")

# %%
!nvidia-smi

# %% [markdown]
# > 🔑 **Note:** If you have access to a CUDA-enabled GPU, TensorFlow will automatically use it whenever possibel


