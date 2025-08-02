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
def get_hidden_jacobian_layer(
    params_dict: Dict[str, Any],
    wrapper,
    layer_dim: int = 120,
    target_layer: str = None
):
    """
    Returns a JITed function:
        jac_fn(example)  ->  jnp.array [layer_dim, num_params_in_target]
    which is ∂z/∂θ^{(target_layer)}.  If target_layer is None, returns
    the full-Jacobian (all params) as before.
    """
    # --- pull out the real pytree of parameters ---------------
    params_pytree = _extract_param_pytree(params_dict)

    # --- build the hidden-layer forward pass ---------------
    model_apply = _make_mid_apply(wrapper, params_dict, layer_dim)

    # --- helper to flatten a pytree into one 1D array --------
    ravel = lambda pyt: flatten_util.ravel_pytree(pyt)[0]

    # --- If no target_layer, fall back to the old full Jacobian
    if target_layer is None:
        @jax.jit
        def jac_fn_full(example):
            # same as your existing code
            example_b = example[None, ...] if example.ndim == 3 else example
            f = partial(model_apply, x=example_b)
            pytree_jac = jax.jacrev(f)(params_pytree)
            rows = [
                ravel(jax.tree_map(lambda w: w[0, i], pytree_jac))
                for i in range(layer_dim)
            ]
            return jnp.stack(rows, 0)
        return jac_fn_full

    # --- Otherwise, split params into (θ^{(ℓ)}, θ^{(-ℓ)}) ----
    if target_layer not in params_pytree:
        raise KeyError(f"Layer {target_layer!r} not found in params.")
    # extract the small subtree we care about:
    theta_layer     = params_pytree[target_layer]
    # the rest stays fixed
    theta_rest_pyt  = {k:v for k,v in params_pytree.items() if k != target_layer}

    # --- define f_sub : θ^{(ℓ)} -> hidden z -------------------
    def f_sub(theta_layer_only, x):
        # reassemble full pytree by inserting the new layer params
        full = dict(theta_rest_pyt)
        full[target_layer] = theta_layer_only
        return model_apply(full, x)   # returns [B, layer_dim]

    @jax.jit
    def jac_fn_layer(example):
        # ensure batch dim
        example_b = example[None, ...] if example.ndim == 3 else example
        # curry out x
        f      = partial(f_sub, x=example_b)
        # jacrev only on the layer we passed in
        pyt_j  = jax.jacrev(f)(theta_layer)  # same tree‐structure as theta_layer
        # now flatten and stack rows:
        # each leaf in pyt_j has shape [1, layer_dim, ...of leaf...]
        rows = []
        for i in range(layer_dim):
            # for each leaf, extract its gradient for unit i, 
            # then flatten across all leaves in theta_layer
            leaf_flat = ravel(jax.tree_map(lambda w: w[0, i], pyt_j))
            rows.append(leaf_flat)
        return jnp.stack(rows, 0)  # [layer_dim, num_params_in_layer]

    return jac_fn_layer



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

############ Hidden version of jacobian ############
def get_hidden_jacobianT_vector_product(
    params_dict,
    wrapper: flax.linen.Module,
    data_array: jax.Array = None,
    single_datapoint: bool = False,
    layer_dim: int = 120,
    target_layer: str = None
):
    """
    Returns a function JT(v) = (∂z / ∂θ^{(ℓ)})^T · v,
    where z is the `layer_dim`‐vector of hidden activations.
    If target_layer is None, θ^{(ℓ)} == all params.
    """

    # 1a) batch‐ify the input
    if single_datapoint:
        data_array = jnp.expand_dims(data_array, 0)   # [1, ...]
    B = data_array.shape[0]

    # 1b) extract the true pytree (and possibly sub‐pytree)
    full_pytree = _extract_param_pytree(params_dict)
    if target_layer is None:
        pytree_to_diff = full_pytree
    else:
        # make sure the key exists
        if target_layer not in full_pytree:
            raise KeyError(f"layer {target_layer!r} not in params")
        pytree_to_diff = full_pytree[target_layer]

    # 1c) build a flax‐Net that returns only the hidden layer z(θ; x)
    model_apply = _make_mid_apply(wrapper, params_dict, layer_dim)

    def hidden_on_data(p_subtree):
        # Re-assemble the full pytree before applying
        if target_layer is None:
            full = p_subtree
        else:
            full = { **{k:v for k,v in full_pytree.items() if k != target_layer},
                     target_layer: p_subtree }
        # hidden: [B, layer_dim]
        return model_apply(full, data_array)

    # 1d) do a single vjp pullback
    _, vjp_fun = jax.vjp(hidden_on_data, pytree_to_diff)

    # 1e) our vector→tree→flat helper
    def vectorize(tree):
        return flatten_util.ravel_pytree(tree)[0]

    @jax.jit
    def hidden_jacT(v):
        # v: flat array of length B×layer_dim
        v = v.reshape((B, layer_dim))           # [B, d]
        grad_subtree = vjp_fun(v)[0]            # pytree shaped like pytree_to_diff
        return vectorize(grad_subtree)          # flat [n_params_subtree]

    return hidden_jacT

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