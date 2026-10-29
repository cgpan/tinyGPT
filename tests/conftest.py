import os

import matplotlib
import torch

# The test models are tiny, and tiny models run faster on one CPU thread than on many.
torch.set_num_threads(1)

# Draw charts off-screen.
matplotlib.use("Agg")

# Chapter 7: tinygpt.run_tests(bug="...") sets this to run the suite with a deliberate bug.
if os.environ.get("TINYGPT_BUG"):
    from tinygpt.bugs import plant

    plant(os.environ["TINYGPT_BUG"])
