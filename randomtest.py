import torch
import matplotlib.pyplot as plt

def VPSDE():
    # generate uniform random numbers
    n_samples = 10000
    n_dim = 2
    x = torch.rand(n_samples, n_dim) + 5
    x_umiform = x.clone()

    BETA = 1.0

    t0 = 0.0
    t1 = 10.0
    delta_t = 0.01
    t = torch.linspace(t0, t1, int((t1 - t0) / delta_t) + 1)
    for _ in range(len(t)):
        x = x - 0.5 * BETA * x * delta_t + torch.sqrt(torch.tensor(delta_t * BETA)) * torch.randn_like(x)

    x_normal = torch.randn_like(x)

    # plot the results
    plt.figure()
    plt.plot(x_umiform[:, 0], x_umiform[:, 1], 'o', label='Initial')
    plt.plot(x[:, 0], x[:, 1], 'o', label='Final')
    plt.plot(x_normal[:, 0], x_normal[:, 1], 'o', label='Normal')
    plt.legend()
    plt.show()


def main():
    pass

if __name__ == "__main__":
    VPSDE()
    # main()