import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Encoder(nn.Module):
    """Encoder module:
    """
    def __init__(self, dim_neck, dim_emb, freq):
        super(Encoder, self).__init__()
        self.dim_neck = dim_neck
        self.freq = freq
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(80+dim_emb if i==0 else 512,
                         512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(512))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, x, c_org):
        x = x.squeeze(1).transpose(2,1)
        c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat((x, c_org), dim=1)
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, :self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck:]
        
        codes = []
        for i in range(0, outputs.size(1), self.freq):
            codes.append(torch.cat((out_forward[:,i+self.freq-1,:],out_backward[:,i,:]), dim=-1))

        return codes
      
        
class Decoder(nn.Module):
    """Decoder module:
    """
    def __init__(self, dim_neck, dim_emb, dim_pre):
        super(Decoder, self).__init__()
        
        self.lstm1 = nn.LSTM(dim_neck*2+dim_emb, dim_pre, 1, batch_first=True)
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(dim_pre,
                         dim_pre,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(dim_pre))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True)
        
        self.linear_projection = LinearNorm(1024, 80)

    def forward(self, x):
        
        #self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        outputs, _ = self.lstm2(x)
        
        decoder_output = self.linear_projection(outputs)

        return decoder_output   
    
    
class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(80, 512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(512))
        )

        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(512,
                             512,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(512))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(512, 80,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(80))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        x = self.convolutions[-1](x)

        return x    
    

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, dim_neck, dim_emb, dim_pre, freq):
        super(Generator, self).__init__()
        
        self.encoder = Encoder(dim_neck, dim_emb, freq)
        self.decoder = Decoder(dim_neck, dim_emb, dim_pre)
        self.postnet = Postnet()

    def forward(self, x, c_org, c_trg):
                
        codes = self.encoder(x, c_org)
        if c_trg is None:
            return torch.cat(codes, dim=-1)
        
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1,int(x.size(1)/len(codes)),-1))
        code_exp = torch.cat(tmp, dim=1)
        
        encoder_outputs = torch.cat((code_exp, c_trg.unsqueeze(1).expand(-1,x.size(1),-1)), dim=-1)
        
        mel_outputs = self.decoder(encoder_outputs)
                
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2,1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2,1)
        
        mel_outputs = mel_outputs.unsqueeze(1)
        mel_outputs_postnet = mel_outputs_postnet.unsqueeze(1)
        
        return mel_outputs, mel_outputs_postnet, torch.cat(codes, dim=-1)

'''
import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

class LinearNorm(torch.nn.Module): #Linear Norm
	def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
		super(LinearNorm, self).__init__()
		self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

		torch.nn.init.xavier_uniform_(
			self.linear_layer.weight,
			gain=torch.nn.init.calculate_gain(w_init_gain))

	def forward(self, x):
		return self.linear_layer(x)

class ConvNorm(torch.nn.Module): #Convolutional Norm 
	def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
				 padding=None, dilation=1, bias=True, w_init_gain='linear'):
		super(ConvNorm, self).__init__()
		if padding is None:
			assert(kernel_size % 2 == 1)
			padding = int(dilation * (kernel_size - 1) / 2)

		self.conv = torch.nn.Conv1d(in_channels, out_channels,
									kernel_size=kernel_size, stride=stride,
									padding=padding, dilation=dilation,
									bias=bias)

		torch.nn.init.xavier_uniform_(
			self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

	def forward(self, signal):
		conv_signal = self.conv(signal)
		return conv_signal

class Encoder(nn.Module):
	"""Encoder module:
	"""
	def __init__(self, dim_neck, dim_emb, freq): #Set up 5x1 ConvNorm * 3 and BLSTM * 2
		#
		# dim_neck: bottleneck 
		# dim_emb:  size of style encoding		  (Line 61, hidden size of LSTM)
		# freq:	 frequency of sampling the audio (Paper 4.2, freq是downsampling/upsampling留下的取timestamp的frequency)
		super(Encoder, self).__init__()
		self.dim_neck = dim_neck 
		self.freq = freq #What is freq?
		
		convolutions = []
		for i in range(3):
			conv_layer = nn.Sequential(
				ConvNorm(80+dim_emb if i==0 else 512,
						 512,
						 kernel_size=5, stride=1,
						 padding=2,
						 dilation=1, w_init_gain='relu'),
				nn.BatchNorm1d(512)) 

			#Normalizes the current batch, making sure the features have the  same scaling 
			#If this is not done, then the effects different features will have on the output will differ
			#(if the scales of the features differ greatly),
			#and the gradients calculated when performing backpropogation will be different, making training difficult
			#More info: https://www.youtube.com/watch?v=BZh1ltr5Rkg
			
			#irene: 針對這次要訓練的batch做normalization，network中要加batchnorm的目的主要是要讓
			#	   該batch的每一筆feature有一樣的scaling，如果feature因為scale差很多，對output的影響
			#	   不一樣，做backpropagation時算得的gradient會不同，造成訓練困難
			#	   詳細可以參考李宏毅老師的課程影片 https://www.youtube.com/watch?v=BZh1ltr5Rkg
			# OK 已了解!
		
		
			convolutions.append(conv_layer)
		self.convolutions = nn.ModuleList(convolutions) #Turns the list of Models into one big model, chains them together according to the order in the list 
		
		self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True) #Adds BLSTM * 2 

	def forward(self, x, c_org): 
		#x:		 tensor containing mel-spectrogram
		#c_org:	 original speaker embedding

		#What is the preprocessing on the 3 lines below for?
		#irene: 前兩行基本上是把它變成吃進model的 dimension, pytorch是 BxCxHxW 
		#	   (B: batch size, C: channel, H: height, W: weight)
		#	   第三行是把這兩個input concatenate，變成真的input
		x = x.double()
		c_org = c_org.double()

		x = x.squeeze(1).transpose(2,1) #Only one channel so squeeze out Was: time->frqs, now: frq->times
		c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1)) #I think it's to make it fit the dimensions of x
		x = torch.cat((x, c_org), dim=1)
		
		for conv in self.convolutions: #Apply the convolutions to x
			x = F.relu(conv(x))
		x = x.transpose(1, 2) #I guess transpose back to the original format, there is a transpose above
		
		self.lstm.flatten_parameters()
		outputs, _ = self.lstm(x) #_ means to ignore the returned value 

		out_forward = outputs[:, :, :self.dim_neck]
		out_backward = outputs[:, :, self.dim_neck:]
		
		codes = []
		for i in range(0, outputs.size(1), self.freq):
			codes.append(torch.cat((out_forward[:,i+self.freq-1,:],out_backward[:,i,:]), dim=-1)) #???
		# I think it is to combine the forward output and backward output of LSTM

		return codes
		
class Decoder(nn.Module):
	"""Decoder module:
	"""
	def __init__(self, dim_neck, dim_emb, dim_pre): #dim_neck seems to be Down1 and Down2? dim_emb seems to be E_S, dim_neck looks to be E_C
		#There should be Up1, Up2 and Copy (from the paper)
		super(Decoder, self).__init__()
		
		self.lstm1 = nn.LSTM(dim_neck*2+dim_emb, dim_pre, 1, batch_first=True) #Why is there an extra LSTM?
		
		convolutions = [] 
		for i in range(3): # 5x1 ConvNorm * 3
			conv_layer = nn.Sequential(
				ConvNorm(dim_pre,
						 dim_pre,
						 kernel_size=5, stride=1,
						 padding=2,
						 dilation=1, w_init_gain='relu'),
				nn.BatchNorm1d(dim_pre))
			convolutions.append(conv_layer)
		self.convolutions = nn.ModuleList(convolutions)
		
		self.lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True) 
		self.linear_projection = LinearNorm(1024, 80)
	def forward(self, x): #Inputs are concatenated beforehand 
		x, _ = self.lstm1(x)
		x = x.transpose(1, 2)
		
		for conv in self.convolutions:
			x = F.relu(conv(x))
		x = x.transpose(1, 2)
		
		outputs, _ = self.lstm2(x)
		
		decoder_output = self.linear_projection(outputs)

		return decoder_output   
	
	
class Postnet(nn.Module):
	"""Postnet
		- Five 1-d convolution with 512 channels and kernel size 5
	"""

	def __init__(self):
		super(Postnet, self).__init__()
		self.convolutions = nn.ModuleList()

		self.convolutions.append(
			nn.Sequential(
				ConvNorm(80, 512,
						 kernel_size=5, stride=1,
						 padding=2,
						 dilation=1, w_init_gain='tanh'),
				nn.BatchNorm1d(512))
		)

		for i in range(1, 5 - 1):
			self.convolutions.append(
				nn.Sequential(
					ConvNorm(512,
							 512,
							 kernel_size=5, stride=1,
							 padding=2,
							 dilation=1, w_init_gain='tanh'),
					nn.BatchNorm1d(512))
			)

		self.convolutions.append(
			nn.Sequential(
				ConvNorm(512, 80,
						 kernel_size=5, stride=1,
						 padding=2,
						 dilation=1, w_init_gain='linear'),
				nn.BatchNorm1d(80))
			)

	def forward(self, x):
		for i in range(len(self.convolutions) - 1):
			x = torch.tanh(self.convolutions[i](x))

		x = self.convolutions[-1](x)

		return x	
	

class Generator(nn.Module):
	"""Generator network."""
	def __init__(self, dim_neck, dim_emb, dim_pre, freq): 
		super(Generator, self).__init__()
		
		self.encoder = Encoder(dim_neck, dim_emb, freq)
		self.decoder = Decoder(dim_neck, dim_emb, dim_pre)
		self.postnet = Postnet()

	def forward(self, x, c_org, c_trg):
		#x:	   spectrogram  
		#c_org:   original style encoding
		#c_trg:   target style encoding

		codes = self.encoder(x, c_org) #Gets content encoding

		if c_trg is None:
			return torch.cat(codes, dim=-1)
		
		tmp = []
		for code in codes:
			tmp.append(code.unsqueeze(1).expand(-1,int(x.size(1)/len(codes)),-1)) #x.size(1) should be the chronological dimension
		code_exp = torch.cat(tmp, dim=1)
		encoder_outputs = torch.cat((code_exp, c_trg.unsqueeze(1).expand(-1,x.size(1),-1)), dim=-1)

		
		mel_outputs = self.decoder(encoder_outputs)
				
		mel_outputs_postnet = self.postnet(mel_outputs.transpose(2,1)) 
		mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2,1) #Adds postnet
		
		return mel_outputs, mel_outputs_postnet, torch.cat(codes, dim=-1)

'''