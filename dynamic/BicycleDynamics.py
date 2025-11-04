class BicycleDynamics(nn.Module):
    """Known bicycle model one-step update, L output by network"""
    def __init__(self, tau=TAU):
        super().__init__()
        self.tau = tau
    def forward(self, z, a, L):
        # z: (B,4) = [x,y,theta,v]; a: (B,2) = [delta, accel]; L: (B,1)
        x,y,theta,v = z[:,0], z[:,1], z[:,2], z[:,3]
        delta = a[:,0]; accel = a[:,1]
        x_next = x + v * torch.cos(theta) * self.tau
        y_next = y + v * torch.sin(theta) * self.tau
        theta_next = theta + (v / (L.squeeze(1))) * torch.tan(delta) * self.tau
        v_next = v + accel * self.tau
        return torch.stack([x_next, y_next, theta_next, v_next], dim=1)
