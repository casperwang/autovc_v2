import torch 
#model constants
bottle_neck = 32
dim_style = 256
dim_pre = 512
freq = 32

#training constants
learning_rate = 0.001
batch_size = 1

#Loss constants
lmb = 1
mu = 1

#other constants
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Uses one GPU when available

