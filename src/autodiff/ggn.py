import jax
import jax.numpy as jnp
import flax
import functools
from jax import flatten_util
from jax import make_jaxpr
from tqdm import tqdm

import time


#####################################
# Generalize Gauss Newtown products #

def get_ggn_vector_product(
        params_dict,
        model: flax.linen.Module,
        data_array: jax.Array,
        single_datapoint = False,
        likelihood_type: str = "regression"
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
    def ggn_tree_product(tree):
        _, J_tree = jax.jvp(model_on_data, (params,), (tree,))
        pred, model_on_data_vjp = jax.vjp(model_on_data, params)
        if likelihood_type == "regression":
            HJ_tree = J_tree
        elif likelihood_type == "classification":
            pred = jax.nn.softmax(pred, axis=1)
            pred = jax.lax.stop_gradient(pred)
            D = jax.vmap(jnp.diag)(pred)
            H = jnp.einsum("bo, bi->boi", pred, pred)
            H = D - H
            HJ_tree = jnp.einsum("boi, bi->bo", H, J_tree)
        elif likelihood_type == "binary_multiclassification":
            #pred = jax.nn.sigmoid(pred)
            #HJ_tree = jnp.einsum("bo, bo->bo", pred - pred**2, J_tree)
            HJ_tree = J_tree
        else:
            raise ValueError(f"Likelihood {likelihood_type} not supported. Use either 'regression' or 'classification'.")
        JtHJ_tree = model_on_data_vjp(HJ_tree)[0]
        return JtHJ_tree
    @jax.jit
    def ggn_vector_product(v):
        tree = devectorize_fun(v)
        ggn_tree = ggn_tree_product(tree)
        ggn_v = flatten_util.ravel_pytree(ggn_tree)[0]
        return jnp.array(ggn_v)
    return ggn_vector_product



def get_ggn_vector_product_dataloader(
        params_dict,
        model: flax.linen.Module,
        dataloader,
        likelihood_type: str = "regression"
    ):
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
    devectorize_fun = flatten_util.ravel_pytree(params)[1]
    flatten_param = jnp.array(flatten_util.ravel_pytree(params)[0])

    @jax.jit
    def ggn_tree_product_batch(X, tree):
        #model_on_data = lambda p: model_apply(p, X)
        model_on_data = functools.partial(model_apply, X)
        _, J_tree = jax.jvp(model_on_data, (params,), (tree,))
        pred, model_on_data_vjp = jax.vjp(model_on_data, params)
        if likelihood_type == "regression":
            HJ_tree = J_tree
        elif likelihood_type == "classification":
            pred = jax.nn.softmax(pred, axis=1)
            pred = jax.lax.stop_gradient(pred)
            D = jax.vmap(jnp.diag)(pred)
            H = jnp.einsum("bo, bi->boi", pred, pred)
            H = D - H
            HJ_tree = jnp.einsum("boi, bi->bo", H, J_tree)
        elif likelihood_type == "binary_multiclassification":
            #pred = jax.nn.sigmoid(pred)
            #HJ_tree = jnp.einsum("bo, bo->bo", pred - pred**2, J_tree)
            HJ_tree = J_tree
        else:
            raise ValueError(f"Likelihood {likelihood_type} not supported. Use either 'regression' or 'classification'.")
        JtHJ_tree = model_on_data_vjp(HJ_tree)[0]
        return JtHJ_tree
    
    @jax.jit
    def ggn_vector_product_batch(X, v):
        tree = devectorize_fun(v)
        ggn_tree = ggn_tree_product_batch(X, tree)
        ggn_v = flatten_util.ravel_pytree(ggn_tree)[0]
        return jnp.asarray(ggn_v)

    batch = next(iter(dataloader))
    x_init = jnp.asarray(batch[0].numpy())

    start = time.time()
    ggn_vector_product_batch(jnp.ones_like(x_init), 2*jnp.ones_like(flatten_param))
    print(f"One BATCH GGN vp took {time.time()-start} seconds")
    start = time.time()
    ggn_vector_product_batch(x_init, flatten_param)
    print(f"Again...... it took {time.time()-start} seconds")
    start = time.time()
    ggn_vector_product_batch(x_init, jnp.ones_like(flatten_param))
    print(f"Aaand again...... it took {time.time()-start} seconds")

    def ggn_vector_product_dataloader(v):
        """
        The loop in this function will never be jitted,
        even inside a jit(), so we should be fine with
        file system access etc.
        """
        result = jnp.zeros_like(v)
        for batch in tqdm(dataloader, desc="Computing GGN vp over dataloader"):
            #print("batch")
            X = jnp.asarray(batch[0].numpy())
            # start = time.time()
            result_batch = ggn_vector_product_batch(X, v)
            # print(X.shape, X.dtype)
            result += result_batch
            # print(f".... inside the loop it took {time.time()-start} seconds")
        return result
    
    # result_shape = jax.ShapeDtypeStruct(flatten_param.shape, flatten_param.dtype)
    # def ggn_vector_product(v):
    #     return jax.pure_callback(ggn_vector_product_dataloader, result_shape, v)

    return ggn_vector_product_dataloader
    # return jax.jit(ggn_vector_product)


#####################################
# Layer-specific GGN products #

def _filter_params_by_layers(params, target_layers):
    """
    Extract a subset of parameters matching target layer names.

    Args:
        params: Parameter pytree (nested dict)
        target_layers: List of layer name prefixes to include. Can match nested paths with '/'.
                      Examples: ['LinearNorm_0'], ['ResNet50Backbone_0/BottleneckStage_3']

    Returns:
        filtered_params: Pytree with only specified layers
        rest_params: Pytree with remaining layers (frozen)
    """
    if target_layers is None:
        return params, {}

    from flax import traverse_util

    # Flatten the params to handle nested structures
    flat_params = traverse_util.flatten_dict(params, sep='/')

    filtered_flat = {}
    rest_flat = {}

    for key_path, value in flat_params.items():
        # Check if any target layer matches this parameter path
        matched = False
        for layer_name in target_layers:
            if key_path.startswith(layer_name):
                filtered_flat[key_path] = value
                matched = True
                break

        if not matched:
            rest_flat[key_path] = value

    # Unflatten back to nested dicts
    filtered = traverse_util.unflatten_dict(filtered_flat, sep='/')
    rest = traverse_util.unflatten_dict(rest_flat, sep='/')

    return filtered, rest


def _merge_params(filtered_params, rest_params):
    """Merge filtered and rest parameter pytrees back together."""
    from flax import traverse_util

    # Flatten both to handle nested structures properly
    flat_filtered = traverse_util.flatten_dict(filtered_params, sep='/')
    flat_rest = traverse_util.flatten_dict(rest_params, sep='/')

    # Merge
    merged_flat = {**flat_rest, **flat_filtered}

    # Unflatten back
    return traverse_util.unflatten_dict(merged_flat, sep='/')


def get_ggn_vector_product_with_layer_filter(
        params_dict,
        model: flax.linen.Module,
        data_array: jax.Array,
        single_datapoint = False,
        likelihood_type: str = "regression",
        target_layers = None
    ):
    """
    Get GGN vector product that only operates on specified layers.

    Args:
        params_dict: Full parameter dictionary
        model: Flax model
        data_array: Data to compute GGN over
        single_datapoint: Whether data_array is a single datapoint
        likelihood_type: Type of likelihood (regression/classification)
        target_layers: List of layer names to include (e.g., ['LinearNorm_0', 'BottleneckStage_3'])
                      If None, uses all layers (equivalent to get_ggn_vector_product)

    Returns:
        ggn_vector_product: Function that computes GGN·v for vector v
    """
    if single_datapoint:
        data_array = jnp.expand_dims(data_array, 0)

    params = params_dict['params']

    # Filter parameters
    filtered_params, rest_params = _filter_params_by_layers(params, target_layers)

    # Set up model evaluation
    if model.has_attentionmask:
        attention_mask = params_dict['attention_mask']
        relative_position_index = params_dict['relative_position_index']
        model_on_data = lambda p: model.apply_test(p, attention_mask, relative_position_index, data_array)
    elif model.has_batch_stats:
        batch_stats = params_dict['batch_stats']
        model_on_data = lambda p: model.apply_test(p, batch_stats, data_array)
    else:
        model_on_data = lambda p: model.apply_test(p, data_array)

    # Create vectorize/devectorize functions for filtered params only
    _, devectorize_fun = flatten_util.ravel_pytree(filtered_params)

    @jax.jit
    def ggn_tree_product(filtered_tree):
        # Merge with frozen rest params
        full_params = _merge_params(filtered_tree, rest_params)

        # Forward pass with JVP on filtered params
        def model_on_filtered(fp):
            return model_on_data(_merge_params(fp, rest_params))

        _, J_tree = jax.jvp(model_on_filtered, (filtered_tree,), (filtered_tree,))
        pred, model_on_data_vjp = jax.vjp(model_on_filtered, filtered_tree)

        if likelihood_type == "regression":
            HJ_tree = J_tree
        elif likelihood_type == "classification":
            pred = jax.nn.softmax(pred, axis=1)
            pred = jax.lax.stop_gradient(pred)
            D = jax.vmap(jnp.diag)(pred)
            H = jnp.einsum("bo, bi->boi", pred, pred)
            H = D - H
            HJ_tree = jnp.einsum("boi, bi->bo", H, J_tree)
        elif likelihood_type == "binary_multiclassification":
            HJ_tree = J_tree
        else:
            raise ValueError(f"Likelihood {likelihood_type} not supported.")

        JtHJ_tree = model_on_data_vjp(HJ_tree)[0]
        return JtHJ_tree

    @jax.jit
    def ggn_vector_product(v):
        tree = devectorize_fun(v)
        ggn_tree = ggn_tree_product(tree)
        ggn_v = flatten_util.ravel_pytree(ggn_tree)[0]
        return jnp.array(ggn_v)

    return ggn_vector_product


def get_ggn_vector_product_dataloader_with_layer_filter(
        params_dict,
        model: flax.linen.Module,
        dataloader,
        likelihood_type: str = "regression",
        target_layers = None
    ):
    """
    Get GGN vector product over dataloader that only operates on specified layers.

    Args:
        params_dict: Full parameter dictionary
        model: Flax model
        dataloader: DataLoader to iterate over
        likelihood_type: Type of likelihood (regression/classification)
        target_layers: List of layer names to include (e.g., ['LinearNorm_0', 'BottleneckStage_3'])
                      If None, uses all layers

    Returns:
        ggn_vector_product_dataloader: Function that computes GGN·v over full dataset
    """
    params = params_dict['params']

    # Filter parameters
    filtered_params, rest_params = _filter_params_by_layers(params, target_layers)

    # Set up model evaluation
    if model.has_attentionmask:
        attention_mask = params_dict['attention_mask']
        relative_position_index = params_dict['relative_position_index']
        model_apply = lambda data, p: model.apply_test(p, attention_mask, relative_position_index, data)
    elif model.has_batch_stats:
        batch_stats = params_dict['batch_stats']
        model_apply = lambda data, p: model.apply_test(p, batch_stats, data)
    else:
        model_apply = lambda data, p: model.apply_test(p, data)

    _, devectorize_fun = flatten_util.ravel_pytree(filtered_params)
    flatten_param = jnp.array(flatten_util.ravel_pytree(filtered_params)[0])

    @jax.jit
    def ggn_tree_product_batch(X, filtered_tree):
        def model_on_filtered(fp):
            return model_apply(X, _merge_params(fp, rest_params))

        _, J_tree = jax.jvp(model_on_filtered, (filtered_tree,), (filtered_tree,))
        pred, model_on_data_vjp = jax.vjp(model_on_filtered, filtered_tree)

        if likelihood_type == "regression":
            HJ_tree = J_tree
        elif likelihood_type == "classification":
            pred = jax.nn.softmax(pred, axis=1)
            pred = jax.lax.stop_gradient(pred)
            D = jax.vmap(jnp.diag)(pred)
            H = jnp.einsum("bo, bi->boi", pred, pred)
            H = D - H
            HJ_tree = jnp.einsum("boi, bi->bo", H, J_tree)
        elif likelihood_type == "binary_multiclassification":
            HJ_tree = J_tree
        else:
            raise ValueError(f"Likelihood {likelihood_type} not supported.")

        JtHJ_tree = model_on_data_vjp(HJ_tree)[0]
        return JtHJ_tree

    @jax.jit
    def ggn_vector_product_batch(X, v):
        tree = devectorize_fun(v)
        ggn_tree = ggn_tree_product_batch(X, tree)
        ggn_v = flatten_util.ravel_pytree(ggn_tree)[0]
        return jnp.asarray(ggn_v)

    # Warm up
    batch = next(iter(dataloader))
    x_init = jnp.asarray(batch[0].numpy())

    start = time.time()
    ggn_vector_product_batch(jnp.ones_like(x_init), 2*jnp.ones_like(flatten_param))
    print(f"One BATCH GGN vp (layer-filtered) took {time.time()-start} seconds")

    def ggn_vector_product_dataloader(v):
        result = jnp.zeros_like(v)
        for batch in tqdm(dataloader, desc="Computing layer-filtered GGN vp over dataloader"):
            X = jnp.asarray(batch[0].numpy())
            result_batch = ggn_vector_product_batch(X, v)
            result += result_batch
        return result

    return ggn_vector_product_dataloader
