from model import Generator
from train import train_one_epoch
import data.dataLoader as data
import torch 
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import argparse
import pdb

bottle_neck = 32
dim_style = 256
learning_rate = 0.0001
dim_pre = 512
freq = 32
batch_size = 32
#other consts 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Uses GPU when availabl


#Init generator, optimizer 

#Get Args:
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--epochs', type = int, default = 1000)  #How many epochs to train
	parser.add_argument('--save_dir', type = str, default = './weights') #Save dir for weights
	parser.add_argument('--load_path', type = str, default = './weights/last.ckpt') #where to load weights from
	parser.add_argument('--mode', type = str, default = 'train') #train for training, eval for evaluation
	parser.add_argument('--write', type = bool, default = True) #Whether to write to TensorBoard
	parser.add_argument('--resume', type = bool, default = False) #Whether to resume training

	opts = parser.parse_args()
	print(opts)

	save_dir = opts.save_dir
	load_path = opts.load_path

	datas = data.voiceDataset()
	dataset = torch.utils.data.DataLoader(datas, batch_size = batch_size, shuffle = True)
	G = Generator(bottle_neck, dim_style, dim_pre, freq).to(device).double()
	optimizer = optim.Adam(G.parameters(), lr = learning_rate) #Not sure what the parameters do, just copying it

	if opts.resume:
		g_checkpoint = torch.load(load_path, map_location = torch.device(device)) #Load from
		G.load_state_dict(g_checkpoint['model'])
		optimizer.load_state_dict(g_checkpoint['optimizer'])

	if(opts.mode == "train"):
		G = G.train()
		current_iter = 0
		for i in range(opts.epochs):
			#Put config as argument
			current_iter = train_one_epoch(G, optimizer, dataset, device, save_dir, current_iter, opts.write)
	else:
		print("All other modes other than train are not available")

