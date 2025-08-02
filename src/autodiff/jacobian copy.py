import jax
import jax.numpy as jnp
from jax import flatten_util
import flax
import functools
###################################
# Extension for hidden layer jacobian

# ---------- A. dig the flax module out of wrapper.apply_test ----------
import inspect
from functools import partial
from typing import Any, Dict

# --------------------------------------------------------
# 1)  dig the real flax.linen.Module out of wrapper.apply_test
# --------------------------------------------------------
import flax.linen as nn


def _hidden_module_from_wrapper(wrapper) -> nn.Module:
    """
    Inspect closure cells in `wrapper.apply_test` and return the first
    flax.linen.Module we find.  Works for all your wrap_model variants.
    """
    func = wrapper.apply_test
    # unwrap (jax.jit / functools.partial) layers if present
    while hasattr(func, "func"):
        func = func.func

    # look through non-locals captured in the closure
    for cell_obj in inspect.getclosurevars(func).nonlocals.values():
        if isinstance(cell_obj, nn.Module):
            return cell_obj

    raise RuntimeError("Could not locate a Flax Module inside wrapper")


# --------------------------------------------------------
# 2)  helper to pull out the *true* parameter PyTree
# --------------------------------------------------------
def _extract_param_pytree(params_dict: Dict[str, Any]):
    """
    All wrappers store learnable params under params_dict['params'].
    The simplest wrapper (wrap_model) nests twice:

        {'params': {'params': <real pytree>}, 'batch_stats': None}

    Others (with batch stats, dropout, …) already store the correct structure:
        {'params': <real pytree>, 'batch_stats': … }

    This function always returns the *inner* pytree that actually contains
    arrays (Conv_0/kernel, Dense_1/bias, …).
    """
    p = params_dict["params"]
    # detect the double nesting
    if isinstance(p, dict) and "params" in p and isinstance(p["params"], dict):
        p = p["params"]
    return p


# --------------------------------------------------------
# 3)  make a function params → hidden_activations
# --------------------------------------------------------
def _make_mid_apply(wrapper, params_dict, layer_dim: int):
    flax_net = _hidden_module_from_wrapper(wrapper)

    # keep extra (static) collections if they exist
    static_vars = {}
    for k in ("batch_stats", "attention_mask", "relative_position_index"):
        if k in params_dict and params_dict[k] is not None:
            static_vars[k] = params_dict[k]

    def model_apply(p, x):
        """p is the real param pytree; x is [B, H, W, C]."""
        variables = {"params": p, **static_vars}
        hidden, _ = flax_net.apply(variables, x, return_hidden=True)
        return hidden  # shape [B, layer_dim]

    return model_apply


# --------------------------------------------------------
# 4)  public builder:  jac_fn(x) → [layer_dim, num_params]
# --------------------------------------------------------
def get_hidden_jacobian(params_dict, wrapper, layer_dim: int = 120):
    """
    Returns a JITed function:

        jac_fn(example)  ->  jnp.array  [layer_dim, num_params]

    which is the Jacobian of the 120-d hidden layer w.r.t. *all* parameters.
    """
    # 4.1  pull out the true parameters (PyTree of arrays)
    params_pytree = _extract_param_pytree(params_dict)

    # 4.2  build mid-layer forward
    model_apply = _make_mid_apply(wrapper, params_dict, layer_dim)

    # 4.3  flatten helper
    ravel = lambda pyt: flatten_util.ravel_pytree(pyt)[0]

    # 4.4  define & jit the jacobian fn
    @jax.jit
    def jac_fn(example):
        # ensure batch dimension
        if example.ndim == 3:  # [H,W,C]
            example_b = example[None, ...]  # [1,H,W,C]
        else:
            example_b = example

        # f(p) = hidden(p, example_b)  -> shape [1, layer_dim]
        f = partial(model_apply, x=example_b)

        # pytree_jac has same structure as params_pytree
        pytree_jac = jax.jacrev(f)(params_pytree)

        # stack per-unit gradients into 2-D matrix
        rows = [
            ravel(jax.tree_map(lambda w: w[0, i], pytree_jac))
            for i in range(layer_dim)
        ]
        return jnp.stack(rows, 0)

    return jac_fn


##################################
# hidden vector-product version of Jacobian
# TODO: get_hidden_jacobian_vector_product
# TODO: get_hidden_jacobianT_vector_product


###################################
# Jacobian products without trees #

def get_jacobian_vector_product(
        params_dict,
        model: flax.linen.Module,
        data_array: jax.Array = None,
        single_datapoint = False,
    ):
    if single_datapoint:
        data_array = jnp.expand_dims(data_array, 0)
    
    params = params_dict['params']
    if model.has_attentionmask:
        attention_mask = params_dict['attention_mask']
        relative_position_index = params_dict['relative_position_index']
        model_on_data = lambda p: model.apply_test(p, attention_mask, relative_position_index, data_array)
    elif model.has_batch_stats:
        batch_stats = params_dict['batch_stats']
        model_on_data = lambda p: model.apply_test(p, batch_stats, data_array)
    else:
        model_on_data = lambda p: model.apply_test(p, data_array)
    devectorize_fun = flatten_util.ravel_pytree(params)[1]

    @jax.jit
    def jacobian_vector_product(vector):
        # parameter space -> data times output space
        tree = devectorize_fun(vector)
        _, J_tree = jax.jvp(model_on_data, (params,), (tree,))
        return J_tree.reshape(-1)
    # batch×output_dim
    return jacobian_vector_product
    
def get_jacobianT_vector_product(
        params_dict,
        model: flax.linen.Module,
        data_array: jax.Array = None,
        single_datapoint = False,
    ):
    if single_datapoint:
        data_array = jnp.expand_dims(data_array, 0)
    B = data_array.shape[0]
    
    params = params_dict['params']
    if model.has_attentionmask:
        attention_mask = params_dict['attention_mask']
        relative_position_index = params_dict['relative_position_index']
        model_on_data = lambda p: model.apply_test(p, attention_mask, relative_position_index, data_array)
    elif model.has_batch_stats:
        batch_stats = params_dict['batch_stats']
        model_on_data = lambda p: model.apply_test(p, batch_stats, data_array)
    else:
        model_on_data = lambda p: model.apply_test(p, data_array)
    _, model_on_data_vjp = jax.vjp(model_on_data, params)
    vectorize_fun = lambda tree: flatten_util.ravel_pytree(tree)[0]

    @jax.jit
    def jacobianT_vector_product(vector):
        # data times output space -> parameter space 
        vector = vector.reshape((B, -1))
        Jt_vector = model_on_data_vjp(vector)[0]
        return vectorize_fun(Jt_vector)
    
    return jacobianT_vector_product



#######################################
# Instatiate full jacobian explicitly #

def get_jacobian_explicit(params_dict, model, output_dim=None):
    vectorize_fun = lambda x : flatten_util.ravel_pytree(x)[0]

    params = params_dict['params']
    if model.has_attentionmask:
        attention_mask = params_dict['attention_mask']
        relative_position_index = params_dict['relative_position_index']
        model_apply = lambda data, p: model.apply_test(p, attention_mask, relative_position_index, data)
    elif model.has_batch_stats:
        batch_stats = params_dict['batch_stats']
        model_apply = lambda data, p: model.apply_test(p, batch_stats, data)
    else:
        model_apply = lambda data, p: model.apply_test(p, data)

    @jax.jit
    def jacobian(query_data):
        query_data = jnp.expand_dims(query_data, 0)
        model_on_data = functools.partial(model_apply, query_data)
        #pytree_jacob = jax.jacfwd(fun)(params)
        pytree_jacob = jax.jacrev(model_on_data)(params)
        # return the jacobian as a output_dim x num_param matrix
        # where p is the number of params
        jacob_array = jnp.asarray([vectorize_fun(jax.tree_map(
            lambda x: x[:, i, :], pytree_jacob)) for i in range(output_dim)]) 
        return jacob_array
    return jacobian