import torch
a = torch.ones((2, 6))
list_of_tensors = torch.chunk(a, dim=1, chunks=3)
print(list_of_tensors.index(2))

for idx, t in enumerate(list_of_tensors):
	print("第{}个张量：{}, shape is {}".format(idx+1, t, t.shape))
