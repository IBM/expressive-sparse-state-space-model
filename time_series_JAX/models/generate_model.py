"""
This module provides a function to generate a model based on a model name and hyperparameters.
It supports various types of models, including Neural CDEs, RNNs, S5 and the PD-SSM model.

Function:
- `create_model`: Generates and returns a model instance along with its state (if applicable)
  based on the provided model name and hyperparameters.

Parameters for `create_model`:
- `model_name`: A string specifying the model architecture to create. 'pdssm'
- `data_dim`: The input data dimension.
- `logsig_dim`: The dimension of the log-signature used in NRDE and Log-NCDE models.
- `logsig_depth`: The depth of the log-signature used in NRDE and Log-NCDE models.
- `intervals`: The intervals used in NRDE and Log-NCDE models.
- `label_dim`: The output label dimension.
- `hidden_dim`: The hidden state dimension for the model.
- `num_blocks`: The number of blocks (layers) in models like LRU or S5 or PDSSM.
- `vf_depth`: The depth of the vector field network for CDE models.
- `vf_width`: The width of the vector field network for CDE models.
- `classification`: A boolean indicating whether the task is classification (True) or regression (False).
- `output_step`: The step interval for outputting predictions in sequence models.
- `ssm_dim`: The state-space model dimension for S5/PDSSM models.
- `ssm_blocks`: The number of SSM blocks in S5 models.
- `solver`: The ODE solver used in CDE models, with a default of `diffrax.Heun()`.
- `stepsize_controller`: The step size controller used in CDE models, with a default of `diffrax.ConstantStepSize()`.
- `dt0`: The initial time step for the solver.
- `max_steps`: The maximum number of steps for the solver.
- `scale`: A scaling factor applied to the vf initialisation in CDE models.
- `lambd`: A regularisation parameter used in Log-NCDE models.
- `key`: A JAX PRNG key for random number generation.

Returns:
- A tuple containing the created model and its state (if applicable).

Raises:
- `ValueError`: If required hyperparameters for the specified model are not provided or if an
  unknown model name is passed.
"""

import equinox as eqx
import jax.random as jr
from models.PDSSM import PDSSM


def create_model(
    model_name, data_dim, label_dim, hidden_dim, num_blocks=None,
    classification=True,  output_step=1,  ssm_dim=None, *,
    key, **kwargs):

    if model_name == "pdssm":
        if num_blocks is None:
            raise ValueError("Must specify num_blocks for PDSSM.")
        pdssm = PDSSM(
            num_blocks,
            data_dim,
            ssm_dim,
            hidden_dim,
            label_dim,
            classification,
            output_step,
            key=key,
        )
        state = eqx.nn.State(pdssm)
        return pdssm, state
    else:
        raise ValueError(f"Unknown model name: {model_name}")
