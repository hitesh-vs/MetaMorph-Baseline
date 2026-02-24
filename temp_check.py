import torch
model, ob_rms = torch.load('output_basic4/Modular-v0.pt')
print("Model loaded successfully!")
print("Number of parameters:", sum(p.numel() for p in model.parameters()))
print("Sample weight value:", list(model.parameters())[0][0][0].item())