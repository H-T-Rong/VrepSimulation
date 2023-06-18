from copy import copy
import time
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# 绘制reward可视化图

fig, axes = plt.subplots(nrows=2, figsize=(6, 7), layout='constrained')


# generate multiple time series of reward  
# Make some data; a 1D random walk + small fraction of sine waves
num_series = 1000
num_points = 100

norm_mean = -1
norm_std = 10

SNR = 0.10  # Signal to Noise Ratio
x = np.linspace(0, 4 * np.pi, num_points)

# Generate unbiased Gaussian random walks
error = (np.random.randn(num_series, num_points) * norm_std) + norm_mean
kp = 0.1
ki = 0.1
kd = 0.1
kn = 0.1
reward_p = - np.absolute(kp * error)
reward_i = - np.absolute(ki * np.cumsum(error, axis=-1))
reward_d = np.zeros(error.shape)
reward_n = - kn * np.arange(0,100)
for pt_idx in range(num_points):
    try:
        reward_d[:,pt_idx] = np.absolute(error[:,pt_idx+1] - error[:,pt_idx])
    except:
        reward_d[:,pt_idx] = reward_d[:,pt_idx-1]
reward_d = -np.absolute(kd * reward_d)

reward = reward_p + reward_i + reward_d
for i in range(reward.shape[0]):
    reward[i,:] = reward[i,:] + reward_n
Y = reward
# Y = np.cumsum(np.random.randn(num_series, num_points), axis=-1)


# Generate sinusoidal signals
# num_signal = round(SNR * num_series)
# phi = (np.pi / 8) * np.random.randn(num_signal, 1)  # small random offset
# Y[-num_signal:] = (
#     np.sqrt(np.arange(num_points))[None, :]  # random walk RMS scaling factor
#     * (np.sin(x[None, :] - phi)
#        + 0.05 * np.random.randn(num_signal, num_points))  # small random noise
# )


# Plot series using `plot` and a small value of `alpha`. With this view it is
# very difficult to observe the sinusoidal behavior because of how many
# overlapping series there are. It also takes a bit of time to run because so
# many individual artists need to be generated.
tic = time.time()
axes[0].plot(x, Y.T, color="C0", alpha=0.1)
toc = time.time()
axes[0].set_title("reward")
print(f"{toc-tic:.3f} sec. elapsed")


# Now we will convert the multiple time series into a histogram. Not only will
# the hidden signal be more visible, but it is also a much quicker procedure.
tic = time.time()
# Linearly interpolate between the points in each time series
num_fine = 800
x_fine = np.linspace(x.min(), x.max(), num_fine)
y_fine = np.empty((num_series, num_fine), dtype=float)
for i in range(num_series):
    y_fine[i, :] = np.interp(x_fine, x, Y[i, :])
y_fine = y_fine.flatten()
x_fine = np.matlib.repmat(x_fine, num_series, 1).flatten()


# Plot (x, y) points in 2d histogram with log colorscale
# It is pretty evident that there is some kind of structure under the noise
# You can tune vmax to make signal more visible
cmap = copy(plt.cm.plasma)
cmap.set_bad(cmap(0))
h, xedges, yedges = np.histogram2d(x_fine, y_fine, bins=[400, 100])
pcm = axes[1].pcolormesh(xedges, yedges, h.T, cmap=cmap,
                         norm=LogNorm(vmax=1.5e2), rasterized=True)
fig.colorbar(pcm, ax=axes[1], label="# points", pad=0)
axes[1].set_title("reward illustrated with log color scale")

# # Same data but on linear color scale
# pcm = axes[2].pcolormesh(xedges, yedges, h.T, cmap=cmap,
#                          vmax=1.5e2, rasterized=True)
# fig.colorbar(pcm, ax=axes[2], label="# points", pad=0)
# axes[2].set_title("reward illustrated with linear color scale")

toc = time.time()
print(f"{toc-tic:.3f} sec. elapsed")
plt.savefig('./evaluation/reward.jpg', dpi=1000)