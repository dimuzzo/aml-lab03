# PyTorch Fundamentals Exercises

# 2. Create a random tensor with shape `(7, 7)`.
import torch
t1 = torch.rand(7, 7)
print("t1:", t1)

# -----------------------------------------------------------------------------
# 3. Perform a matrix multiplication on the tensor from 2 with another random tensor with shape `(1, 7)`
t2 = torch.rand(1, 7)
t3 = torch.matmul(t1, t2.T)
print("t3:", t3)

# -----------------------------------------------------------------------------
# 4. Set the random seed to `0` and do 2 & 3 over again.
torch.manual_seed(0)
t1 = torch.rand(7, 7)
t2 = torch.rand(1, 7)
t3 = torch.matmul(t1, t2.T)
print("t3 (seed 0):", t3, t3.shape)

# -----------------------------------------------------------------------------
# 5. Set GPU random seed to `1234`
if torch.cuda.is_available():
    torch.cuda.manual_seed(1234)

# -----------------------------------------------------------------------------
# 6. Create two random tensors of shape `(2, 3)` and send them both to GPU
torch.manual_seed(1234)
t1 = torch.rand(2, 3)
t2 = torch.rand(2, 3)

if torch.cuda.is_available():
    t1 = t1.to("cuda")
    t2 = t2.to("cuda")

print("Tensors on device:", t1, t2)

# -----------------------------------------------------------------------------
# 7. Perform matrix multiplication on tensors from 6
t3 = torch.matmul(t1, t2.T)
print("t3 matmul:", t3, t3.shape)

# -----------------------------------------------------------------------------
# 8. Find maximum and minimum values of the output of 7
max_val = t3.max()
min_val = t3.min()
print("Max and Min:", max_val, min_val)

# -----------------------------------------------------------------------------
# 9. Find maximum and minimum index values of the output of 7
arg_max = t3.argmax()
arg_min = t3.argmin()
print("ArgMax and ArgMin:", arg_max, arg_min)

# -----------------------------------------------------------------------------
# 10. Squeeze dimensions
torch.manual_seed(7)
t1 = torch.rand(1, 1, 1, 10)
t2 = t1.squeeze()
print("Squeezed:", t1, t1.shape, t2, t2.shape)