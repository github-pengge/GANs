import torch
import torch.nn as nn
from torch.autograd import Variable
try:
	reduce
except Exception as e:
	from functools import reduce


class G_conv(nn.Module):
	def __init__(self, channel=3, size=4, zdim=100):
		super(G_conv, self).__init__()
		self.channel = channel
		self.size = size
		self.zdim = zdim
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.linear1 = nn.Linear(self.zdim, self.size * self.size * 512)
		self.bn1 = nn.BatchNorm2d(512)
		self.conv_trans1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(256)
		self.conv_trans2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(128)
		self.conv_trans3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
		self.bn4 = nn.BatchNorm2d(64)
		self.conv_trans4 = nn.ConvTranspose2d(64, self.channel, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)

		# initialize weights
		for m in self.modules():
			if isinstance(m, nn.ConvTranspose2d):
				m.weight.data.normal_(0, 0.02)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.normal_(1.0, 0.02)
				m.bias.data.fill_(0)
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.02)
				m.bias.data.fill_(0)

	def forward(self, z):
		g = self.linear1(z)
		g = g.view(-1, 512, self.size, self.size)
		g = self.relu(self.bn1(g))
		g = self.relu(self.bn2(self.conv_trans1(g)))
		g = self.relu(self.bn3(self.conv_trans2(g)))
		g = self.relu(self.bn4(self.conv_trans3(g)))
		g = self.sigmoid(self.conv_trans4(g))
		return g


class D_conv(nn.Module):
	def __init__(self, channel=3, img_size=64, last_layer_with_activation=True):
		super(D_conv, self).__init__()
		self.channel = channel
		self.img_size = img_size
		self.last_layer_with_activation = last_layer_with_activation
		self.conv1 = nn.Conv2d(self.channel, 64, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(128)
		self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(256)
		self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn4 = nn.BatchNorm2d(512)
		self.d_ = self.infer_size((self.channel, self.img_size, self.img_size))
		self.linear1 = nn.Linear(self.d_, 256, bias=True)
		self.linear2 = nn.Linear(256, 1, bias=True)
		self.lrelu = nn.LeakyReLU(negative_slope=0.2)
		self.sigmoid = nn.Sigmoid()

		# initialize weights
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				m.weight.data.normal_(0, 0.02)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.normal_(1.0, 0.02)
				m.bias.data.fill_(0)
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.02)
				m.bias.data.fill_(0)

	def infer_size(self, shape):
		x = Variable(torch.rand(1, *shape))
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		return int(x.data.view(1, -1).size(1))

	def forward(self, x):
		d = self.lrelu(self.bn1(self.conv1(x)))
		d = self.lrelu(self.bn2(self.conv2(d)))
		d = self.lrelu(self.bn3(self.conv3(d)))
		d = self.lrelu(self.bn4(self.conv4(d)))
		d = d.view(-1, self.d_)
		d = self.lrelu(self.linear1(d))
		d = self.linear2(d)
		if self.last_layer_with_activation:
			d = self.sigmoid(d)
		return d


class D_autoencoder(nn.Module):
	def __init__(self, channel=3, n_hidden=256, img_size=64):
		super(D_autoencoder, self).__init__()
		self.channel = channel
		self.n_hidden = n_hidden
		self.img_size = img_size

		# conv
		self.conv1 = nn.Conv2d(self.channel, 64, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(128)
		self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(256)
		self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn4 = nn.BatchNorm2d(512)
		self.dim = self.infer_size((channel, img_size, img_size))
		self.d_ = int(reduce(lambda x, y: x*y, self.dim))
		self.linear1 = nn.Linear(self.d_, self.n_hidden, bias=True)

		# deconv
		self.linear2 = nn.Linear(self.n_hidden, self.d_, bias=True)
		self.bn5 = nn.BatchNorm2d(512)
		self.conv_trans1 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1, bias=False)
		self.bn6 = nn.BatchNorm2d(256)
		self.conv_trans2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1, bias=False)
		self.bn7 = nn.BatchNorm2d(128)
		self.conv_trans3 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1, bias=False)
		self.bn8 = nn.BatchNorm2d(64)
		self.conv_trans4 = nn.ConvTranspose2d(64, self.channel, 3, stride=2, padding=1, output_padding=1, bias=False)

		# activation
		self.lrelu = nn.LeakyReLU(negative_slope=0.2)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

		# initialize weights
		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
				m.weight.data.normal_(0, 0.02)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.normal_(1.0, 0.02)
				m.bias.data.fill_(0)
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.02)
				m.bias.data.fill_(0)

	def infer_size(self, shape):
		x = Variable(torch.rand(1, *shape))
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		return x.size()[1:]
	
	def forward(self, x):
		d = self.lrelu(self.bn1(self.conv1(x)))
		d = self.lrelu(self.bn2(self.conv2(d)))
		d = self.lrelu(self.bn3(self.conv3(d)))
		d = self.lrelu(self.bn4(self.conv4(d)))
		d = d.view(-1, self.d_)
		d = self.lrelu(self.linear1(d))
		d = self.sigmoid(self.linear2(d))
		d = d.view(-1, *self.dim)
		d = self.relu(self.bn5(d))
		d = self.relu(self.bn6(self.conv_trans1(d)))
		d = self.relu(self.bn7(self.conv_trans2(d)))
		d = self.relu(self.bn8(self.conv_trans3(d)))
		d = self.sigmoid(self.conv_trans4(d))
		return d


class D_vae(nn.Module):
	def __init__(self, dim=100, channel=3, n_hidden=256, img_size=64):
		super(D_vae, self).__init__()
		self.dim = dim
		self.channel = channel
		self.n_hidden = n_hidden
		self.img_size = img_size

		# conv
		self.conv1 = nn.Conv2d(self.channel, 64, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(128)
		self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(256)
		self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn4 = nn.BatchNorm2d(512)
		self.dim = self.infer_size((channel, img_size, img_size))
		self.d_ = int(reduce(lambda x, y: x*y, self.dim))
		self.linear1 = nn.Linear(self.d_, self.n_hidden, bias=True)
		self.linear2 = nn.Linear(self.n_hidden, self.dim, bias=True)
		self.linear3 = nn.Linear(self.n_hidden, self.dim, bias=True)

		self.lrelu = nn.LeakyReLU(negative_slope=0.2)

		# initialize weights
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				m.weight.data.normal_(0, 0.02)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.normal_(1.0, 0.02)
				m.bias.data.fill_(0)
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.02)
				m.bias.data.fill_(0)

	def infer_size(self, shape):
		x = Variable(torch.rand(1, *shape))
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		return x.size()[1:]

	def forward(self, x):
		d = self.lrelu(self.bn1(self.conv1(x)))
		d = self.lrelu(self.bn2(self.conv2(d)))
		d = self.lrelu(self.bn3(self.conv3(d)))
		d = self.lrelu(self.bn4(self.conv4(d)))
		d = d.view(-1, self.d_)
		d = self.lrelu(self.linear1(d))
		mu = self.linear2(d)
		sigma = self.linear3(d)
		return mu, sigma
