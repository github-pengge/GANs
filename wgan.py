import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys, time
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append('utils')
from nets import *
from data import *


def sample_z(m, n):
	return np.random.uniform(-1., 1., size=[m, n])


class WGAN():
	def __init__(self, generator, discriminator, data, cuda=True):
		self.generator = generator
		self.discriminator = discriminator
		self.data = data
		self.cuda = cuda

		self.z_dim = self.data.z_dim
		self.size = self.data.size
		self.channel = self.data.channel

		if self.cuda:
			self.generator.cuda()
			self.discriminator.cuda()

	def train(self, sample_dir, ckpt_dir, training_epochs=500000, batch_size=32):
		fig_count = 0
		g_lr = 2e-4
		d_lr = 1e-4
		if self.cuda:
			input = Variable(torch.FloatTensor(batch_size, self.channel, self.size, self.size).cuda())
			z = Variable(torch.FloatTensor(batch_size, self.z_dim).cuda())
		else:
			input = Variable(torch.FloatTensor(batch_size, self.channel, self.size, self.size))
			z = Variable(torch.FloatTensor(batch_size, self.z_dim))

		optimizer_D = optim.Adam(self.discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999))
		optimizer_G = optim.Adam(self.generator.parameters(), lr=g_lr, betas=(0.5, 0.999))

		n_d = 5

		for epoch in range(training_epochs):
			begin_time = time.time()

			# update D
			for i in range(n_d):
				self.discriminator.zero_grad()
				# real samples
				X_b_real = self.data(batch_size)
				input.data.copy_(torch.from_numpy(X_b_real))
				D_real = self.discriminator(input)

				# fake samples
				z.data.copy_(torch.from_numpy(sample_z(batch_size, self.z_dim)))
				X_b_fake = self.generator(z)
				D_fake = self.discriminator(X_b_fake.detach())  # detach so that backward will not apply to generator
				D_loss = torch.mean(D_fake) - torch.mean(D_real)
				D_loss.backward()

				optimizer_D.step()
				
				for p in self.discriminator.parameters():
					p.data.clamp_(-0.01, 0.01)

			# update G
			self.generator.zero_grad()
			D_fake = self.discriminator(X_b_fake)
			G_loss = -torch.mean(D_fake)
			G_loss.backward()
			optimizer_G.step()

			elapse_time = time.time() - begin_time
			print('Iter[%s], d_loss: %.4f, g_loss: %.4f, time elapsed: %.4fsec' % \
					(epoch+1, D_loss.cpu().data.numpy(), G_loss.cpu().data.numpy(), elapse_time))

			if epoch % 500 == 0:
				z.data.copy_(torch.from_numpy(sample_z(batch_size, self.z_dim)))
				samples = self.generator(z).cpu().data.numpy()
				fig = self.data.data2fig(samples)
				plt.savefig('{}/{}.png'.format(sample_dir, str(fig_count).zfill(3)), bbox_inches='tight')
				fig_count += 1
				plt.close(fig)

			if epoch % 5000 == 0:
				torch.save(self.generator.state_dict(), os.path.join(ckpt_dir, 'G_epoch-%s.pth' % epoch))
				torch.save(self.discriminator.state_dict(), os.path.join(ckpt_dir, 'D_epoch-%s.pth' % epoch))


if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES'] = '1'

	# save generated images
	sample_dir = 'Samples/wgan'
	ckpt_dir = 'Models/wgan'
	if not os.path.exists(sample_dir):
		os.makedirs(sample_dir)
	if not os.path.exists(ckpt_dir):
		os.makedirs(ckpt_dir)

	generator = G_conv()
	discriminator = D_conv(last_layer_with_activation=False)

	data = celebA()

	wgan = WGAN(generator, discriminator, data)
	wgan.train(sample_dir, ckpt_dir, batch_size=64)
	