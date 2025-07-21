import torch 
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from src.sde import VP_SDE
from sklearn.datasets import make_moons

# ------------------------------
# Data: Mixture of Two Gaussians
# ------------------------------
def sample_data(batch_size):
    """
    Sample from a mixture of two Gaussians:
      0.5 * N(-3, 1) + 0.5 * N(3, 1)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Randomly choose which Gaussian to sample from
    choices = torch.randint(0, 2, (batch_size,), device=device)
    means = torch.where(choices == 0, -3.0, 3.0)
    # Standard deviation 1
    x0 = torch.randn(batch_size, device=device) + means
    return x0.unsqueeze(1)  # shape: [batch_size, 1]

# -----------------------
# Score Network (MLP)
# -----------------------
class ScoreNet(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, embed_dim=32):
        super(ScoreNet, self).__init__()
        self.fc1 = nn.Linear(input_dim + embed_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.act = nn.ELU()

    def forward(self, x, t):
        # x: [batch, 1]; t: [batch, 1] or [batch]
        if t.dim() == 1:
            t = t.unsqueeze(1)
        t_embed = self.get_timestep_embedding(t, embed_dim=32)  # [batch, embed_dim]
        h = torch.cat([x, t_embed], dim=1)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        out = self.fc3(h)
        return out  # approximates the score
    
    @staticmethod
    def get_timestep_embedding(t, embed_dim):
        """
        Create sinusoidal embeddings for t.
        t: tensor of shape [batch] (or [batch, 1])
        """
        if t.dim() == 2:
            t = t.squeeze(1)
        half_dim = embed_dim // 2
        # log(10000) as scaling constant
        emb_scale = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb_scale)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embed_dim % 2 == 1:  # if odd, pad with zeros
            emb = torch.cat([emb, torch.zeros(t.shape[0], 1, device=t.device)], dim=1)
        return emb  # shape: [batch, embed_dim]



def train(save_dir):
    # ---------------------------
    # Training Setup
    # ---------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sde = VP_SDE(rescale=True).to(device)
    score_net = ScoreNet(input_dim=2, output_dim=2, embed_dim=32).to(device)
    optimizer = optim.Adam(score_net.parameters(), lr=1e-2)
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "logs"))

    batch_size = 1024
    num_steps = 10000  # training iterations
    loss_best = 1e10
    print("Starting training...")
    for step in range(num_steps):
        optimizer.zero_grad()
        # x0 = sample_data(batch_size)  # shape: [batch, 1]
        x0 = torch.Tensor(make_moons(n_samples=batch_size, noise=0.05)[0]).to(device)
        loss = sde.score_matching_loss(score_net, x0)
        loss.backward()
        optimizer.step()
        writer.add_scalar("loss", loss.item(), step)
        
        if step % 100 == 0:
            print(f"Step {step:05d}, Loss: {loss.item():.4f}")
            if loss < loss_best:
                loss_best = loss
    torch.save(score_net.state_dict(), os.path.join(save_dir, "score_net.pt"))

            
def test(save_dir, device='cuda'):
    net = ScoreNet(input_dim=2, output_dim=2, embed_dim=32).to(device)
    net.load_state_dict(torch.load(os.path.join(save_dir, "score_net.pt")))
    net.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sde = VP_SDE(rescale=True).to(device)
    samples = sde.predictor_corrector_sample(net, (1000, 2), device, n_lang_steps=0)
    print("score function at t=0: ", net(torch.tensor([[0.0 ,0.0]], device=device), torch.zeros(1, device=device)))
    # for i in range(100):
    #     samples = sde.langevin_step(net, samples, t = torch.ones(1000, device=device) * sde.eps, snr=0.16)
    samples = samples.cpu().numpy()
    plt.figure(figsize=(6, 6))
    # plt.hist(samples, bins=100, density=True, alpha=0.5, color='blue')
    plt.scatter(samples[:, 0], samples[:, 1], s=10)
    plt.axis('equal')
    plt.savefig(os.path.join(save_dir, "diffusion_samples.pdf"))
    plt.show()
    plt.close()
    print("Saved samples to", os.path.join(save_dir, "diffusion_samples.pdf"))

def main():
    dir_base = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(dir_base, "results", "diffusion")
    os.makedirs(save_dir, exist_ok=True)

    # train(save_dir)
    test(save_dir)

if __name__ == "__main__":
    main()

