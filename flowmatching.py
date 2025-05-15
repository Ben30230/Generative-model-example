import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

import os
from torch.utils.tensorboard import SummaryWriter

class Flow(nn.Module):
    def __init__(self, dim: int = 2, h: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, dim))
    
    def forward(self, t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((t, x_t), -1))
    
    def step(self, x_t: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor) -> torch.Tensor:
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1) # shape: [batch, 1]
        
        return x_t + (t_end - t_start) * self(t=t_start + (t_end - t_start) / 2, x_t= x_t + self(x_t=x_t, t=t_start) * (t_end - t_start) / 2)
    

def train(save_path, device="cuda"):
    flow = Flow().to(device)
    writer = SummaryWriter(os.path.join(save_path, "logs"))

    optimizer = torch.optim.Adam(flow.parameters(), 1e-2)
    loss_fn = nn.MSELoss()
    loss_best = 1e10

    for epoch in range(10000+1):
        optimizer.zero_grad()
        x_1 = torch.Tensor(make_moons(256, noise=0.05)[0]).to(device) # shape: [batch, 2]
        x_0 = torch.randn_like(x_1).to(device)
        t = torch.rand(len(x_1), 1).to(device)
        
        x_t = (1 - t) * x_0 + t * x_1
        dx_t = x_1 - x_0
        
        loss = loss_fn(flow(t=t, x_t=x_t), dx_t)
        loss.backward()
        optimizer.step()
        writer.add_scalar("loss", loss.item())
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            if loss.item() < loss_best:
                loss_best = loss.item()
                torch.save(flow.state_dict(), os.path.join(save_path, "flow.pth"))
        

def samples(save_path, device="cuda"):
    flow = Flow().to(device)
    flow.load_state_dict(torch.load(os.path.join(save_path, "flow.pth"), weights_only=True))
    flow.eval()

    x = torch.randn(300, 2).to(device)
    n_steps = 8
    fig, axes = plt.subplots(1, n_steps + 1, figsize=(30, 4), sharex=True, sharey=True)
    time_steps = torch.linspace(0, 1.0, n_steps + 1).to(device)

    axes[0].scatter(x.cpu().detach()[:, 0], x.cpu().detach()[:, 1], s=10)
    axes[0].set_title(f't = {time_steps[0]:.2f}')
    axes[0].set_xlim(-3.0, 3.0)
    axes[0].set_ylim(-3.0, 3.0)

    for i in range(n_steps):
        x = flow.step(x_t=x, t_start=time_steps[i], t_end=time_steps[i + 1])
        axes[i + 1].scatter(x.cpu().detach()[:, 0], x.cpu().detach()[:, 1], s=10)
        axes[i + 1].set_title(f't = {time_steps[i + 1]:.2f}')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "samples.pdf"))
    plt.show()

def main():
    dir_base = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(dir_base, "results", "flowmatching")    
    os.makedirs(save_dir, exist_ok=True)

    device = 'cuda' # cuda or cpu
    train(save_dir, device)
    samples(save_dir, device)

if __name__ == '__main__':
    main()