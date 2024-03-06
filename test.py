from tensors import Tensor

t1 = Tensor([[1,2,3,4]])
t2 = Tensor([2,2,2,2])

print(t1.shape)
print(t2.shape)
print(t1 + t2)