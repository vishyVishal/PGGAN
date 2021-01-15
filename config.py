import argparse

parser = argparse.ArgumentParser('Args for PGGAN')
parser.add_argument('--data_root', type=str, default='data/celeba_faces')
parser.add_argument('--resolution', type=int, default=256)      # resolution for image to generate.
parser.add_argument('--use_ema', type=bool, default=True)       # whether to use EMA to weights of generator.
parser.add_argument('--ema_mu', type=float, default=0.999)      # smoothing factor for smoothed generator.
parser.add_argument('--latent_dim', type=int, default=512)      # input dimension of noise.
parser.add_argument('--use_cuda', type=bool, default=True)      # whether to use cuda device.
parser.add_argument('--switch_mode_number', type=int, default=800000)  # number of passed images to switch mode.
parser.add_argument('--switch_number_increase', type=int, default=0)  # higher resolution need more real images to train.
parser.add_argument('--n_critic', type=int, default=1)           # n(D)/n(G)
parser.add_argument('--max_feature_map', type=int, default=256)   # max number of feature maps of Convolution layers.
parser.add_argument('--use_weightscale', type=bool, default=True)  # whether to use equalized-learning rate.
parser.add_argument('--use_pixelnorm', type=bool, default=True)    # whether to use pixelwise normalization for generator.
parser.add_argument('--use_gdrop', type=bool, default=False)        # whether to use generalized dropout layer for discriminator.
parser.add_argument('--use_leaky', type=bool, default=True)         # whether to use leaky relu instead of relu.
parser.add_argument('--negative_slope', type=float, default=0.2)   # negative slope for leaky relu
parser.add_argument('--tanh_at_end', type=bool, default=False)     # whether to use tanh at the end of the generator.
parser.add_argument('--sigmoid_at_end', type=bool, default=False)  # whether to use sigmoid at the end of the discriminator.
parser.add_argument('--minibatch_stat_concat', type=bool, default=True)  # whether to add minbatch-std-channel in the discriminator.
parser.add_argument('--normalize_latent', type=bool, default=True)  # whether to use pixelwise normalization of latent vector.


parser.add_argument('--lr', type=float, default=0.001)              # learning rate.
parser.add_argument('--lr_decay', type=float, default=0.87)         # learning rate decay at every resolution transition, no decay if set to 0.
parser.add_argument('--beta0', type=float, default=0.0)             # beta0 for adam.
parser.add_argument('--beta1', type=float, default=0.99)            # beta1 for adam.


config, _ = parser.parse_known_args()
