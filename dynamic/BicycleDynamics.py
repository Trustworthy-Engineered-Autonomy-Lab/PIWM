class BicycleDynamics(nn.Module):
    """Known bicycle model one-step update, with optional cornering speed loss."""
    def __init__(self, tau=TAU, turn_loss_gain=0.0):
        super().__init__()
        self.tau = tau
        # Multiplies lateral acceleration magnitude, v^2 * |tan(delta) / L|.
        # A zero default preserves the original ideal bicycle-model behavior.
        self.turn_loss_gain = turn_loss_gain

    def forward(self, z, a, L):
        # z: (B,4) = [x,y,theta,v]; a: (B,2) = [delta, accel]; L: (B,1)
        x,y,theta,v = z[:,0], z[:,1], z[:,2], z[:,3]
        delta = a[:,0]; accel = a[:,1]
        curvature = torch.tan(delta) / L.squeeze(1)
        turn_decel = self.turn_loss_gain * v.square() * curvature.abs()
        x_next = x + v * torch.cos(theta) * self.tau
        y_next = y + v * torch.sin(theta) * self.tau
        theta_next = theta + v * curvature * self.tau
        v_next = v + (accel - turn_decel) * self.tau
        return torch.stack([x_next, y_next, theta_next, v_next], dim=1)
