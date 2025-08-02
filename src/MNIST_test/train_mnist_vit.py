import os
import struct
import numpy as onp
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state, checkpoints

# --- Model Definition ---
class PatchEmbedding(nn.Module):
    patch_size: int
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            name="proj"
        )(x)
        b, h, w, d = x.shape
        x = x.reshape(b, h*w, d)
        return x

class EncoderBlock(nn.Module):
    dim: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, *, deterministic):
        # Self-attention
        y = nn.LayerNorm(name="ln1")(x)
        # Flax's MultiHeadDotProductAttention expects num_heads, qkv_features, out_features
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            out_features=self.dim,
            dropout_rate=self.dropout_rate,
            name="attn"
        )(y, y, deterministic=deterministic)
        x = x + y

        # MLP
        y = nn.LayerNorm(name="ln2")(x)
        y = nn.Dense(self.mlp_dim, name="mlp_fc1")(y)
        y = nn.gelu(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
        y = nn.Dense(self.dim, name="mlp_fc2")(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
        return x + y

class ViT(nn.Module):
    img_size: int = 28
    patch_size: int = 7
    in_channels: int = 1
    num_classes: int = 10
    dim: int = 64
    depth: int = 4
    num_heads: int = 1
    mlp_dim: int = 128
    dropout_rate: float = 0.1

    def setup(self):
        self.patch_embed = PatchEmbedding(
            patch_size=self.patch_size,
            embed_dim=self.dim)
        num_patches = (self.img_size // self.patch_size)**2

        self.cls_token = self.param(
            'cls_token', nn.initializers.normal(stddev=1.0),
            (1, 1, self.dim))
        self.pos_embed = self.param(
            'pos_embed', nn.initializers.normal(stddev=1.0),
            (1, 1 + num_patches, self.dim))

        self.encoders = [
            EncoderBlock(self.dim, self.num_heads, self.mlp_dim, self.dropout_rate)
            for _ in range(self.depth)
        ]
        self.classifier = nn.Sequential([
            nn.LayerNorm(),
            nn.Dense(self.num_classes)
        ])

    def __call__(self, x, *, train: bool = False):
        b = x.shape[0]
        x = self.patch_embed(x)
        cls = jnp.tile(self.cls_token, (b, 1, 1))
        x = jnp.concatenate([cls, x], axis=1)
        x = x + self.pos_embed[:, :x.shape[1], :]

        for block in self.encoders:
            x = block(x, deterministic=not train)

        cls_final = x[:, 0]
        logits = self.classifier(cls_final)
        return logits

# --- MNIST Data Loading from Local IDX Files ---
def load_mnist_idx(path, kind='train'):
    """
    Load MNIST data from `path` directory containing:
      - train-images-idx3-ubyte
      - train-labels-idx1-ubyte
      - t10k-images-idx3-ubyte
      - t10k-labels-idx1-ubyte
    """
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte')
    # Read labels
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = onp.fromfile(lbpath, dtype=onp.uint8)
    # Read images
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = onp.fromfile(imgpath, dtype=onp.uint8).reshape(num, rows, cols)
    return images, labels


def get_datasets(batch_size, data_dir):
    # Load raw numpy arrays
    train_images, train_labels = load_mnist_idx(data_dir, 'train')
    test_images, test_labels   = load_mnist_idx(data_dir, 't10k')

    # Normalize and reshape
    train_images = train_images.astype(onp.float32) / 255.0
    test_images  = test_images.astype(onp.float32)  / 255.0

    # Batch generators
    def data_generator(images, labels, shuffle=True):
        idx = onp.arange(len(images))
        if shuffle:
            onp.random.shuffle(idx)
        for i in range(0, len(images), batch_size):
            batch_idx = idx[i:i+batch_size]
            batch_imgs = images[batch_idx][..., None]  # (B,H,W,1)
            batch_lbl = labels[batch_idx]
            yield {
                'image': jnp.array(batch_imgs),
                'label': jnp.array(batch_lbl)
            }

    train_gen = lambda: data_generator(train_images, train_labels, shuffle=True)
    test_gen  = lambda: data_generator(test_images, test_labels, shuffle=False)
    return train_gen, test_gen

# --- Training Utilities ---
class TrainState(train_state.TrainState):
    pass


def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
    return optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()

@jax.jit
def train_step(state, batch, rng):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'], rngs={'dropout': rng}, train=True)
        loss = cross_entropy_loss(logits, batch['label'])
        return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    acc = jnp.mean(jnp.argmax(logits, -1) == batch['label'])
    return state, {'loss': loss, 'accuracy': acc}

@jax.jit
def eval_step(params, batch):
    logits = ViT().apply({'params': params}, batch['image'], train=False)
    loss = cross_entropy_loss(logits, batch['label'])
    acc  = jnp.mean(jnp.argmax(logits, -1) == batch['label'])
    return {'loss': loss, 'accuracy': acc}

# --- Main Training Loop ---
def main():
    # Hyperparameters
    num_epochs    = 10
    batch_size    = 128
    learning_rate = 1e-3
    data_dir      = '/home/skyle/Desktop/uq_benchmark/datasets/MNIST/raw'

    # Data generators
    train_gen, test_gen = get_datasets(batch_size, data_dir)

    # Initialize model & optimizer
    rng = jax.random.PRNGKey(0)
    dropout_rng, init_rng = jax.random.split(rng)
    dummy_input = jnp.ones((1,28,28,1), jnp.float32)
    model = ViT()
    params = model.init({'params': init_rng, 'dropout': dropout_rng}, dummy_input, train=True)['params']
    tx = optax.adam(learning_rate)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Training loop
    for epoch in range(1, num_epochs+1):
        # Train
        train_metrics = {'loss': [], 'accuracy': []}
        for batch in train_gen():
            rng, step_rng = jax.random.split(rng)
            state, m = train_step(state, batch, step_rng)
            train_metrics['loss'].append(m['loss'])
            train_metrics['accuracy'].append(m['accuracy'])
        train_loss = onp.mean(onp.array(train_metrics['loss']))
        train_acc  = onp.mean(onp.array(train_metrics['accuracy']))

        # Eval
        eval_metrics = {'loss': [], 'accuracy': []}
        for batch in test_gen():
            m = eval_step(state.params, batch)
            eval_metrics['loss'].append(m['loss'])
            eval_metrics['accuracy'].append(m['accuracy'])
        test_loss = onp.mean(onp.array(eval_metrics['loss']))
        test_acc  = onp.mean(onp.array(eval_metrics['accuracy']))

        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc*100:.2f}%, "
              f"Test Loss: {test_loss:.4f}, "
              f"Test Acc: {test_acc*100:.2f}%")

    # Save checkpoint
    checkpoints.save_checkpoint("./checkpoints", state.params, step=num_epochs)

if __name__ == '__main__':
    main()
