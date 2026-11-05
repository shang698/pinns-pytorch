import torch


def gradient(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def nls_fields(model, points):
    xt = points.detach().clone().requires_grad_(True)
    uv = model(xt)
    du = gradient(uv[:, :1], xt)
    dv = gradient(uv[:, 1:2], xt)
    return xt, uv, du, dv


def schrodinger_residual(model, points):
    xt, uv, du, dv = nls_fields(model, points)
    u, v = uv[:, :1], uv[:, 1:2]
    uxx, vxx = gradient(du[:, :1], xt)[:, :1], gradient(dv[:, :1], xt)[:, :1]
    amplitude_squared = u.square() + v.square()
    return torch.cat(
        (
            du[:, 1:2] + 0.5 * vxx + amplitude_squared * v,
            dv[:, 1:2] - 0.5 * uxx - amplitude_squared * u,
        ),
        dim=1,
    )


def componentwise_dx(outputs, x):
    """Compute each stage derivative with the two-pass autograd construction."""
    dummy = torch.ones_like(outputs, requires_grad=True)
    vjp = torch.autograd.grad(outputs, x, dummy, create_graph=True)[0]
    return torch.autograd.grad(vjp, dummy, torch.ones_like(vjp), create_graph=True)[0]


def allen_cahn_initial(model, points, dt, weights):
    x = points.detach().clone().requires_grad_(True)
    all_stages = model(x)  # q internal stages plus the final-time solution
    internal = all_stages[:, :-1]
    ux = componentwise_dx(internal, x)
    uxx = componentwise_dx(ux, x)
    rhs = 5 * internal - 5 * internal.pow(3) + 0.0001 * uxx
    return all_stages - dt * (rhs @ weights.T)


def allen_cahn_boundary(model, points):
    x = points.detach().clone().requires_grad_(True)
    all_stages = model(x)
    return all_stages, componentwise_dx(all_stages, x)
