import torch, deepspeed as ds
torch.manual_seed(0); torch.cuda.set_device(0)
model = torch.nn.Sequential(torch.nn.Linear(1024,4096), torch.nn.GELU(), torch.nn.Linear(4096,1024)).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
engine, _, _, _ = ds.initialize(model=model, optimizer=optimizer, config={
  "train_batch_size": 2,
  "train_micro_batch_size_per_gpu": 1,
  "zero_optimization": {"stage": 1},
  "bf16": {"enabled": True},
})
x = torch.randn(2,1024, device="cuda")
for step in range(2):  # two fwd/bwd/step cycles
    y = engine(x).sum()
    engine.backward(y)
    engine.step()
print("OK")