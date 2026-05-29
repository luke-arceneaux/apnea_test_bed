'''
Create a Pytorch Mobile-optimized model from a Pytorch model
Saves as a .ptl file

This .ptl file can then be used run inference on an Android/iOS app.
''' 
import torch 
import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile
from models import CNN

# Instantiate Pytorch model
model = CNN(1,2)
model.eval()
example = torch.rand(1,120)

# Trace with torch.jit
traced_script_module = torch.jit.trace(model, example)
# Apply optimizations for the edg
traced_script_module_optimized = optimize_for_mobile(traced_script_module)
# Save output model
traced_script_module_optimized._save_for_lite_interpreter("model.ptl")