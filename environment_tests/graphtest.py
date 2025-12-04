# file: test_cudagraphs.py
import torch, torch.nn as nn

torch.manual_seed(0)
m = nn.Sequential(nn.Linear(4096,4096), nn.ReLU(), nn.Linear(4096,4096)).to("cuda").bfloat16()
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
x = torch.randn(8,4096, device="cuda").bfloat16()
y = torch.randn(8,4096, device="cuda").bfloat16()

# optional: try compiled path (Savanna may not use compile; this just exercises it)
try:
    m = torch.compile(m)  # should be okay inside NGC 24.09
except Exception:
    pass

for step in range(2):
    opt.zero_grad(set_to_none=True)
    out = m(x)
    loss = (out - y).pow(2).mean()
    loss.backward()
    opt.step()
print("OK")