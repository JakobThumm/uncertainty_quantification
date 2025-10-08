import jax
import jax.numpy as jnp
from src.models import compute_num_params
from src.datasets.utils import get_subset_loader
from src.autodiff.ggn import get_ggn_vector_product, get_ggn_vector_product_dataloader
from src.autodiff.hessian import get_hessian_vector_product, get_hessian_vector_product_dataloader
from src.autodiff.jacobian import get_jacobian_vector_product, get_jacobianT_vector_product
from src.lanczos.low_memory import low_memory_lanczos
from src.estimators.frobenius import get_frobenius_norm, get_frobenius_norm_sequential, get_frobenius_norm_difference_sequential
from src.sketches import No_sketch, Dense_sketch, SRFT_sketch
import numpy as np
import time
import os
import cloudpickle
import hashlib


def _get_cache_base_key(args_dict, trainset_size, n_params):
    """Generate base cache key for all intermediate computations"""
    cache_params = {
        'dataset': args_dict['ID_dataset'],
        'model': args_dict.get('model'),
        'run_name': args_dict.get('run_name'),
        'trainset_size': trainset_size,
        'n_params': n_params,
        'model_seed': args_dict.get('model_seed'),
        'use_hessian': args_dict.get('use_hessian', False),
    }

    param_str = str(sorted(cache_params.items()))
    cache_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]

    base_key = (
        f"{args_dict['ID_dataset']}"
        f"_{args_dict.get('model', 'unknown')}"
        f"_n{trainset_size}"
        f"_{cache_hash}"
    )
    return base_key


def _save_ggn_vector_product(cache_dir, base_key, ggn_vector_product, args_dict, trainset_size):
    """Save GGN vector product (as a pickled function)"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{base_key}_ggn.cloudpickle")

    cache_data = {
        'ggn_vector_product': ggn_vector_product,
        'trainset_size': trainset_size,
        'use_hessian': args_dict.get('use_hessian', False),
        'likelihood': args_dict['likelihood'],
        'serialize_ggn_on_batches': args_dict.get('serialize_ggn_on_batches', False),
    }

    with open(cache_path, 'wb') as f:
        cloudpickle.dump(cache_data, f)

    print(f"Saved GGN vector product to {cache_path}")


def _load_ggn_vector_product(cache_dir, base_key):
    """Load GGN vector product from cache"""
    cache_path = os.path.join(cache_dir, f"{base_key}_ggn.cloudpickle")

    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"GGN cache file not found: {cache_path}")

    with open(cache_path, 'rb') as f:
        cache_data = cloudpickle.load(f)

    print(f"Loaded GGN vector product from {cache_path}")
    return cache_data['ggn_vector_product']


def _save_sketch_op(cache_dir, base_key, sketch_op, args_dict, n_params):
    """Save sketch operator"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{base_key}_sketch.cloudpickle")

    cache_data = {
        'sketch_op': sketch_op,
        'sketch': args_dict.get('sketch'),
        'sketch_size': args_dict.get('sketch_size'),
        'sketch_seed': args_dict.get('sketch_seed'),
        'sketch_padding': args_dict.get('sketch_padding'),
        'sketch_density': args_dict.get('sketch_density'),
        'n_params': n_params,
    }

    with open(cache_path, 'wb') as f:
        cloudpickle.dump(cache_data, f)

    print(f"Saved sketch operator to {cache_path}")


def _load_sketch_op(cache_dir, base_key):
    """Load sketch operator from cache"""
    cache_path = os.path.join(cache_dir, f"{base_key}_sketch.cloudpickle")

    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Sketch cache file not found: {cache_path}")

    with open(cache_path, 'rb') as f:
        cache_data = cloudpickle.load(f)

    print(f"Loaded sketch operator from {cache_path}")
    return cache_data['sketch_op']


def _save_eigenpairs(cache_dir, base_key, eigenvec, eigenval, args_dict):
    """Save eigenvectors and eigenvalues"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{base_key}_eigenpairs.cloudpickle")

    cache_data = {
        'eigenvec': np.array(eigenvec),
        'eigenval': np.array(eigenval),
        'lanczos_lm_iter': args_dict['lanczos_lm_iter'],
        'lanczos_seed': args_dict['lanczos_seed'],
        'n_eigenvec_lm': args_dict.get('n_eigenvec_lm'),
    }

    with open(cache_path, 'wb') as f:
        cloudpickle.dump(cache_data, f)

    print(f"Saved eigenpairs to {cache_path}")


def _load_eigenpairs(cache_dir, base_key):
    """Load eigenvectors and eigenvalues from cache"""
    cache_path = os.path.join(cache_dir, f"{base_key}_eigenpairs.cloudpickle")

    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Eigenpairs cache file not found: {cache_path}")

    with open(cache_path, 'rb') as f:
        cache_data = cloudpickle.load(f)

    print(f"Loaded eigenpairs from {cache_path}")
    return cache_data['eigenvec'], cache_data['eigenval']


def _save_score_functions(cache_dir, base_key, score_fun, eigenval, approx_quadratic_form, quadratic_form, args_dict):
    """Save score functions and eigenvalues"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{base_key}_score_functions.cloudpickle")

    cache_data = {
        'score_fun': score_fun,
        'eigenval': eigenval,
        'approx_quadratic_form': approx_quadratic_form,
        'quadratic_form': quadratic_form,
        'lanczos_lm_iter': args_dict['lanczos_lm_iter'],
        'lanczos_hm_iter': args_dict.get('lanczos_hm_iter', 0),
        'use_eigenvals': args_dict.get('use_eigenvals', False),
        'prior_std': args_dict.get('prior_std', 0.1),
    }

    with open(cache_path, 'wb') as f:
        cloudpickle.dump(cache_data, f)

    print(f"Saved score functions to {cache_path}")


def load_score_functions(cache_dir, base_key):
    """Load score functions and eigenvalues from cache"""
    cache_path = os.path.join(cache_dir, f"{base_key}_score_functions.cloudpickle")

    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Score functions cache file not found: {cache_path}")

    with open(cache_path, 'rb') as f:
        cache_data = cloudpickle.load(f)

    print(f"Loaded score functions from {cache_path}")
    return (
        cache_data['score_fun'],
        cache_data['eigenval'],
        cache_data['approx_quadratic_form'],
        cache_data['quadratic_form']
    )


def _get_or_compute_ggn(params_dict, model, train_loader, args_dict, trainset_size, n_params, cache_dir, base_key):
    """Get GGN vector product from cache or compute it"""
    load_ggn = args_dict.get('load_ggn_vector_product', False)

    if load_ggn:
        try:
            print("Loading GGN vector product from cache...")
            ggn_vector_product = _load_ggn_vector_product(cache_dir, base_key)
            print("Successfully loaded GGN vector product")
            return ggn_vector_product
        except FileNotFoundError as e:
            print(f"Failed to load GGN: {e}")
            print("Computing GGN vector product from scratch...")

    # Compute GGN vector product
    if not args_dict["serialize_ggn_on_batches"]:
        data_array = jnp.asarray([train_loader.dataset[i][0] for i in range(trainset_size)])
        if not args_dict["use_hessian"]:
            ggn_vector_product = get_ggn_vector_product(
                params_dict, model, data_array=data_array,
                likelihood_type=args_dict["likelihood"]
            )
        else:
            print("Using the Hessian instead of the GGN")
            ggn_vector_product = get_hessian_vector_product(
                params_dict, model,
                data_array=(data_array, jnp.asarray([data[1] for data in train_loader.dataset])),
                likelihood_type=args_dict["likelihood"]
            )
    else:
        train_loader = get_subset_loader(
            train_loader, trainset_size,
            batch_size=args_dict["train_batch_size"],
            drop_last=True
        )
        if not args_dict["use_hessian"]:
            ggn_vector_product = get_ggn_vector_product_dataloader(
                params_dict, model, train_loader,
                likelihood_type=args_dict["likelihood"]
            )
        else:
            print("Using the Hessian instead of the GGN")
            ggn_vector_product = get_hessian_vector_product_dataloader(
                params_dict, model, train_loader,
                likelihood_type=args_dict["likelihood"]
            )

    # Warm up / JIT compile GGN
    start = time.time()
    ggn_vector_product(jax.random.normal(jax.random.PRNGKey(0), shape=(n_params,)))
    print(f"One GGN vp took {time.time()-start} seconds")

    # Save GGN if cache_dir is specified
    if cache_dir:
        _save_ggn_vector_product(cache_dir, base_key, ggn_vector_product, args_dict, trainset_size)

    return ggn_vector_product


def _get_or_compute_sketch(args_dict, n_params, cache_dir, base_key):
    """Get sketch operator from cache or compute it"""
    load_sketch = args_dict.get('load_sketch_op', False)

    if load_sketch:
        try:
            print("Loading sketch operator from cache...")
            sketch_op = _load_sketch_op(cache_dir, base_key)
            print("Successfully loaded sketch operator")
            return sketch_op
        except FileNotFoundError as e:
            print(f"Failed to load sketch: {e}")
            print("Computing sketch operator from scratch...")

    print("Creating sketch operator...")
    key_sketch = jax.random.PRNGKey(args_dict["sketch_seed"])
    if args_dict["sketch"] is None:
        sketch_op = No_sketch()
    elif args_dict["sketch"] == "srft":
        print(f"Use srft sketch with num params {n_params} and padding {args_dict['sketch_padding']} --> fake num params = {args_dict['sketch_padding']+n_params} must NOT have prime factors >127 (thanks JAX fft)")
        sketch_op = SRFT_sketch(key_sketch, n_params, args_dict['sketch_size'], padding=args_dict['sketch_padding'])
    elif args_dict["sketch"] == "dense":
        print(f"Use dense sketch with {'optimal' if args_dict['sketch_density'] is None else args_dict['sketch_density']} density")
        sketch_op = Dense_sketch(key_sketch, n_params, args_dict['sketch_size'], density=args_dict['sketch_density'])
    else:
        raise ValueError(f"Sketch '{args_dict['sketch']}' not supported. Use either 'srtf', 'dense' or None.")

    if cache_dir:
        _save_sketch_op(cache_dir, base_key, sketch_op, args_dict, n_params)

    print("Successfully created sketch operator.")
    return sketch_op


def _get_or_compute_eigenpairs(ggn_vector_product, sketch_op, args_dict, n_params, trainset_size, cache_dir, base_key):
    """Get eigenpairs from cache or compute them via Lanczos"""
    load_eigenpairs = args_dict.get('load_eigenpairs', False)

    if load_eigenpairs:
        try:
            print("Loading eigenpairs from cache...")
            eigenvec, eigenval = _load_eigenpairs(cache_dir, base_key)
            eigenvec = jnp.asarray(eigenvec)
            eigenval = jnp.asarray(eigenval)
            print(f"Successfully loaded {len(eigenval)} eigenvalues")
            print(f"  Eigenvals = {eigenval[:5]} ... {eigenval[-5:]}")
            return eigenvec, eigenval
        except FileNotFoundError as e:
            print(f"Failed to load eigenpairs: {e}")
            print("Computing eigenpairs from scratch...")

    print("Computing eigenpairs using Lanczos...")
    start = time.time()
    key_lanczos = jax.random.PRNGKey(args_dict["lanczos_seed"])
    eigenvec, eigenval = low_memory_lanczos(key_lanczos, ggn_vector_product, n_params, args_dict["lanczos_lm_iter"], sketch_op)
    print(f"Lanczos {args_dict['lanczos_lm_iter']} iterations, dataset size {trainset_size}, with {n_params} params model -> took {time.time()-start:.3f} seconds")
    print(f"returned {len(eigenval)} eigenvals = {eigenval[:5]} ... {eigenval[-5:]}")

    # Orthogonalize and select the first (good) 'n_eigenvec' vectors
    start = time.time()
    print("Doing PCA...")
    U, S, _ = np.linalg.svd(eigenvec @ jnp.diag(eigenval), full_matrices=False)
    if args_dict['n_eigenvec_lm'] < len(S):
        threshold = sorted(S, reverse=True)[args_dict['n_eigenvec_lm']]
        eigenvec = U[:, S > threshold]
        eigenval = S[S > threshold]
    else:
        eigenvec = U
        eigenval = S
    eigenvec = jnp.asarray(eigenvec)
    eigenval = jnp.asarray(eigenval)
    print(f"PCA took {time.time()-start:.3f} seconds")

    if cache_dir:
        _save_eigenpairs(cache_dir, base_key, eigenvec, eigenval, args_dict)

    return eigenvec, eigenval


def low_memory_lanczos_score_fun(
        model,
        params_dict,
        train_loader,
        args_dict,
        use_eigenvals : bool = True
    ):
    # Validate cache dependencies
    load_ggn = args_dict.get('load_ggn_vector_product', False)
    load_sketch = args_dict.get('load_sketch_op', False)
    load_eigenpairs = args_dict.get('load_eigenpairs', False)

    if load_sketch and not load_ggn:
        raise ValueError("--load_sketch_op requires --load_ggn_vector_product")
    if load_eigenpairs and not (load_ggn and load_sketch):
        raise ValueError("--load_eigenpairs requires both --load_ggn_vector_product and --load_sketch_op")

    # Setup parameters
    cache_dir = args_dict.get('cache_dir')
    trainset_size = int(0.9*args_dict["subsample_trainset"])
    n_params = compute_num_params(params_dict["params"])
    prior_scale = 1. / (2 * trainset_size * args_dict['prior_std']**2)

    # Get base cache key
    base_key = None
    if cache_dir or load_ggn:
        base_key = _get_cache_base_key(args_dict, trainset_size, n_params)
        print(f"Cache base key: {base_key}")

    # Get or compute GGN, sketch, and eigenpairs
    ggn_vector_product = _get_or_compute_ggn(
        params_dict, model, train_loader, args_dict, trainset_size, n_params, cache_dir, base_key
    )
    sketch_op = _get_or_compute_sketch(args_dict, n_params, cache_dir, base_key)
    eigenvec, eigenval = _get_or_compute_eigenpairs(
        ggn_vector_product, sketch_op, args_dict, n_params, trainset_size, cache_dir, base_key
    )

    # Define score functions
    @jax.jit
    def approx_ggn_vector_product(vector):
        return sketch_op.T @ jnp.einsum("ab, b, cb, c-> a", eigenvec, eigenval, eigenvec, sketch_op @ vector)

    if use_eigenvals:
        scale = jnp.sqrt(eigenval / (eigenval + prior_scale))
        @jax.jit
        def inv_sqrt_approx_ggn_vector_product(vector):
            return ((sketch_op @ vector).T @ eigenvec) * scale
    else:
        @jax.jit
        def inv_sqrt_approx_ggn_vector_product(vector):
            return (sketch_op @ vector).T @ eigenvec

    @jax.vmap
    @jax.jit
    def score_fun(datapoint):
        jacobianT_vector_product = get_jacobianT_vector_product(params_dict, model, datapoint, single_datapoint=True)
        variance = get_frobenius_norm_difference_sequential(
            jacobianT_vector_product,
            inv_sqrt_approx_ggn_vector_product,
            dim_in = args_dict["output_dim"]
        )
        return variance * args_dict['prior_std']**2

    @jax.vmap
    @jax.jit
    def quadratic_form(datapoint):
        jacobian_vector_product = get_jacobian_vector_product(params_dict, model, datapoint, single_datapoint=True)
        jacobianT_vector_product = get_jacobianT_vector_product(params_dict, model, datapoint, single_datapoint=True)
        real_quadratic_form = jax.jit(lambda vector: jacobian_vector_product(ggn_vector_product(jacobianT_vector_product(vector))))
        qf = get_frobenius_norm(
            real_quadratic_form,
            dim_in = args_dict["output_dim"],
            sequential = True
        )
        return qf

    @jax.vmap
    @jax.jit
    def approx_quadratic_form(datapoint):
        jacobian_vector_product = get_jacobian_vector_product(params_dict, model, datapoint, single_datapoint=True)
        jacobianT_vector_product = get_jacobianT_vector_product(params_dict, model, datapoint, single_datapoint=True)
        fake_quadratic_form = jax.jit(lambda vector: jacobian_vector_product(approx_ggn_vector_product(jacobianT_vector_product(vector))))
        approx_qf = get_frobenius_norm(
            fake_quadratic_form,
            dim_in = args_dict["output_dim"],
        )
        return approx_qf

    return score_fun, eigenval, approx_quadratic_form, quadratic_form
