import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

train_X = torch.rand(10, 2, dtype=torch.double) * 2

# print(train_X)
# explicit output dimension -- Y is 10 x 1
train_Y = 1 - (train_X - 0.5).norm(dim=-1, keepdim=True)
train_Y += 0.1 * torch.rand_like(train_Y)

gp = SingleTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    input_transform=Normalize(d=2),
    outcome_transform=Standardize(m=1),
)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)