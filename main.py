import rp
import src.peekaboo as peekaboo
from src.peekaboo import run_peekaboo
import torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

#Mario vs Luigi Part 1
torch.cuda.empty_cache()
results=run_peekaboo('Mario', "https://i1.sndcdn.com/artworks-000160550668-iwxjgo-t500x500.jpg")