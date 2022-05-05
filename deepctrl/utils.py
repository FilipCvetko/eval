import numpy as np
import torch


def verification(out, pert_out, threshold=0.0):
    """
    return the ratio of qualified samples.
    """
    if isinstance(out, torch.Tensor):
        return 1.0 * torch.sum(pert_out - out < threshold) / out.shape[0]
    else:
        return 1.0 * np.sum(pert_out - out < threshold) / out.shape[0]


def get_perturbed_input(input_tensor, pert_coeff):
    """
    X = X + pert_coeff*rand*X
    return input_tensor + input_tensor*pert_coeff*torch.rand()
    """
    device = input_tensor.device
    return input_tensor + torch.abs(input_tensor) * pert_coeff * torch.rand(
        input_tensor.shape, device=device
    )


def custom_slope_loss(
    x, x_pert, y, y_pert, rule_idx=2, epsilon=10e-6, desired_slope=10
):
    delta_x = x_pert[:, rule_idx] - x[:, rule_idx] + epsilon
    delta_y = torch.squeeze(y_pert - y)

    slope = torch.div(delta_y, delta_x)

    return torch.mean(torch.abs(desired_slope - slope))
