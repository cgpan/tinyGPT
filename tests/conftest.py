import matplotlib
import torch

# The test models are tiny, and tiny models run faster on one CPU thread than on many.
torch.set_num_threads(1)

# Draw charts off-screen.
matplotlib.use("Agg")
