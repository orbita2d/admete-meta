# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "jax[cuda12]~=0.6.0",
#    "flax~=0.10.6",
#    "optax",
#    "orbax",
#    "pandas",
#    "numpy",
#    "scipy",
#    "pyarrow",
#    "tqdm",
#   ]
# ///

import argparse
from functools import partial
import inspect
import logging
from pathlib import Path
import shutil
from typing import Literal
import jax
from jax import numpy as jnp
import numpy as np
import flax
from flax import nnx
import orbax.checkpoint as ocp
import optax
import pandas as pd
import numpy as np
from jax import numpy as jnp
import pyarrow as pa
import pyarrow.dataset as pds
from tqdm.auto import tqdm
from jax import random
from flax.nnx import initializers


logger = logging.getLogger(__name__)

N_FEATURES = 64 * 5
        

class AdmeteModel(nnx.Module):
    def __init__(self, rngs):
        AccumulatorSize = 256
        HiddenSize = 72
        self.accumulator_baseline = nnx.Linear(
            2*(N_FEATURES+64),
            AccumulatorSize,
            rngs=rngs,
            use_bias=False
        )
        # self.accumulator_delta = nnx.LinearGeneral(
        #     (N_FEATURES, 64, 2), # for each player for each feature, for each king location
        #     AccumulatorSize,
        #     rngs=rngs,
        #     axis=(-1, -2, -3),
        #     use_bias=False,
        #     # kernel_init=initializers.zeros, # no initial delta, we learn it from scratch
        # )
        self.accumulator_batchnorm = nnx.BatchNorm(AccumulatorSize, rngs=rngs, use_bias=True, use_scale=False) # apply the bias after the batchnorm
        self.hidden_layer = nnx.Linear(
            AccumulatorSize,
            HiddenSize,
            rngs=rngs,
            use_bias=False
        )
        self.hidden_batchnorm = nnx.BatchNorm(HiddenSize, rngs=rngs, use_bias=True, use_scale=False)
        self.final_layer = nnx.Linear(
            HiddenSize,
            1,
            rngs=rngs,
            use_bias=True
    )
        self.layers = [self.hidden_layer, self.final_layer] # makes it easier to reason about the model when analysing the parameters

    def __call__(self, us, them, k_us, k_them, train=False, zero_delta=False):
        assert us.shape[1] == N_FEATURES
        assert us.shape == them.shape

        x1 = self.accumulator_baseline(jnp.concatenate([us, jax.nn.one_hot(k_us, 64), them, jax.nn.one_hot(k_them, 64)], axis=-1))  
        # def delta_part(us, them, k_us, k_them):
        #     lk = self.accumulator_delta.kernel[:, k_us, 0].T @ us
        #     rk = self.accumulator_delta.kernel[:, k_them, 1].T @ them
        #     return lk + rk
        # if not zero_delta:
        #     x2 = nnx.vmap(delta_part, in_axes=(0, 0, 0, 0), out_axes=0)(us, them, k_us, k_them)
        #     x = x1 + x2
        # else:
        #     x = x1
        x = x1

        x = self.accumulator_batchnorm(x, use_running_average=not train)
        x = nnx.relu(x)
        x = self.hidden_layer(x)
        x = self.hidden_batchnorm(x, use_running_average=not train)
        x = nnx.relu(x)
        x = self.final_layer(x)
        # batched= nnx.vmap(forward, in_axes=(0, 0, 0, 0), out_axes=0)(us, them, k_us, k_them)
        return jnp.reshape(x, (-1,)) # flatten the batch dimension


def normalise_eval(eval: jax.Array, logistic_scaling: float) -> jax.Array:
    return nnx.sigmoid(eval/logistic_scaling)

def loss_fn(pred, labels):
    # return jnp.mean((pred-labels)**2)
    # binary cross entropy (equivalent to KL divergence)
    eps = 1e-7
    assert pred.shape == labels.shape
    labels = jnp.clip(labels, eps, 1-eps) # avoid log(0)
    pred = jnp.clip(pred, eps, 1-eps) # avoid log(0)
    return -jnp.mean(labels*jnp.log(pred) + (1-labels)*jnp.log(1-pred)) 

def regularize(model: AdmeteModel, regularization: float = 0., zero_delta: bool = False) -> jax.Array:
    # l2 norm over the accumulator deltas
    if zero_delta:
        return 0.
    l_baseline = nnx.state(model)['accumulator_baseline'].kernel.value
    l_delta = nnx.state(model)['accumulator_delta'].kernel.value
    norm_baseline = jnp.mean(jnp.square(l_baseline), axis=None)
    norm_delta = jnp.mean(jnp.square(l_delta), axis=None)
    return regularization * (norm_delta / norm_baseline)

base_params = nnx.All(nnx.Param, nnx.Not(nnx.PathContains("accumulator_delta")))
delta_params = nnx.All(nnx.Param, nnx.PathContains("accumulator_delta"))

# @nnx.jit
@partial(nnx.jit, static_argnames=('freeze_delta'))
def train_step(model:nnx.Module, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch: dict[str, jax.Array], regularization: float, freeze_delta: bool = False):
    def loss(model: AdmeteModel, regularization: float):
        pred = model(batch['f_us'], batch['f_them'], batch['k_us'], batch['k_them'], train=True, zero_delta=freeze_delta)
        # loss = loss_fn(nnx.sigmoid(pred), batch['label']) + regularize(model, regularization, zero_delta=freeze_delta)
        loss = loss_fn(nnx.sigmoid(pred), batch['label'])
        return loss
    
    if freeze_delta:
        diff_state = nnx.DiffState(0, base_params) # only update the base parameters, not the delta parameters (much faster)
    else:
        diff_state = 0
    
    grads = nnx.grad(loss, argnums=diff_state)(model, regularization)
    loss_ex_reg = loss(model, 0.)
    optimizer.update(grads)
    metrics.update(loss=loss_ex_reg)

@nnx.jit
def eval_step(model:nnx.Module, metrics: nnx.MultiMetric, batch: dict[str, jax.Array]):
    pred = nnx.sigmoid(model(batch['f_us'], batch['f_them'], batch['k_us'], batch['k_them']))
    label = batch['label']
    loss_val = loss_fn(pred, label)

    metrics.update(
        # loss=loss_vals, # let the multi-metric handle the averaging (so we don't have weird cases from non-uniform batch sizes)
        loss=loss_val,
        invloss=loss_fn(1-pred, label),
        winning_ratio=jnp.mean(pred > 0.8),
        losing_ratio=jnp.mean(pred < 0.2),
    )

@jax.jit
def unpack_features(features: jax.Array) -> dict[str, jax.Array]:
    """
    Unpack the features into a dictionary.
    """
    pieces_values_us = [
    1, # white pawn
    2,
    3,
    4,
    5, # white queen
    ]
    pieces_values_them = [
    9, # black pawn
    10, 
    11,
    12,
    13, # black queen
    ]
    flipped = jnp.flip(features.reshape(-1, 8, 8), axis=1).reshape(-1, 64)

    f_us = jnp.concat([jnp.array(features == i, dtype=jnp.int8) for i in pieces_values_us], axis=1)
    f_them = jnp.concat([jnp.array(flipped == i, dtype=jnp.int8) for i in pieces_values_them], axis=1)
    # king square is the index of the king piece (6 or 13), get the index
    k_us = jnp.argmax(features == 6, axis=1)
    k_them = jnp.argmax(flipped == 14, axis=1)
    return {
        'f_us': f_us,
        'f_them': f_them,
        'k_us': k_us,
        'k_them': k_them,
    }


class FileSource:
    def __init__(self, path: Path, random_seed: int, logistic_scaling: float):
        feature_length = 64
        schema = pa.schema([
            pa.field('features', pa.list_(pa.int8(), feature_length)),
            pa.field('eval', pa.int32()),
            pa.field('setType', pa.string()), # 0 = training, 1 = test, 2 = validation
        ])

        partition = pds.partitioning(pa.schema([
            pa.field('setType', pa.string(), nullable=False),
        ]))

        self.max_rows = 24 * (1<<30) // (feature_length + 4) # 24 GiB, 64 features + 4 eval + setType
        self.ds = pds.dataset(path, partitioning=partition, schema=schema)
        logger.info(f"Initialised {self.ds.count_rows()} rows from {path}")
        logger.info(f"Schema: {self.ds.schema}")
        logger.info(f"Test samples:       {self.samples("test"):>10,}")
        logger.info(f"Train samples:      {self.samples("train"):>10,}")
        logger.info(f"Validation samples: {self.samples("validation"):>10,}")
        logger.info(f"Total samples:      {self.ds.count_rows():>10,}")
        logger.info(f"Max rows:           {self.max_rows:>10,}")
        self.path = path
        self.random_seed = random_seed
        self.logistic_scaling = logistic_scaling
        test_data = self.test_data()
        logger.info(f"Loaded {test_data.num_rows} test samples into memory")
        self.test_jax = self.load_to_device(test_data)
        logger.info(f"Loaded {test_data.num_rows} test samples into device {self.test_jax['features'].device.id} ({self.test_jax['features'].device.device_kind})")
        del test_data
        # show device info
        self._training_data = None
        self.training_data()
        logger.info(f"Loaded {self._training_data.num_rows} training samples into memory")
        
    def __len__(self):
        return self.ds.count_rows()
    
    def samples(self, dataset: Literal["test", "train", "validation"]):
        assert dataset in ["test", "train", "validation"], f"Dataset must be one of 'test', 'train', 'validation', not {dataset}."
        max_rows = min(self.max_rows, self.ds.count_rows())
        return int(self.ds.count_rows(filter=pds.field("setType") == f"setType={dataset}") / self.ds.count_rows() * max_rows)
    
    def load_to_device(self, table: pa.Table):
        # load the dataset into jax
        features = jnp.from_dlpack(table.column('features').combine_chunks().values).reshape(-1, 64)
        arr = {
            'features': features,
            'label': normalise_eval(jnp.from_dlpack(table.column('eval').combine_chunks()), self.logistic_scaling),
            }
        arr = jax.device_put(arr, device=jax.devices()[0])
        return arr

    
    def batched(self, batch_size: int, *, dataset=Literal["test", "train", "validation"], repeat: bool = True):
        if dataset == "test":
            return self._batched_test(batch_size)
        return self._batched_lazy(batch_size, dataset=dataset, repeat=repeat)
        
    def training_data(self):
        if self._training_data is None:
            self._training_data = self.ds.head(self.samples("train"), filter=pds.field("setType") == "setType=train")
        return self._training_data
    
    def test_data(self):
        tab = self.ds.head(self.samples("test"), filter=pds.field("setType") == "setType=test")
        print(f"Length: {tab.num_rows}\nExpected: {self.samples('test')}\nDS Count: {self.ds.count_rows(filter=pds.field('setType') == 'setType=test')}")
        return tab
    
    def clear_in_memory(self):
        self._training_data = None
        self._test_data = None
    
    def validation_data(self):
        return self.ds.head(self.samples("validation"), filter=pds.field("setType") == "setType=validation")

    def _batched_lazy(self, batch_size: int, *, dataset: Literal["test", "train", "validation"], repeat: bool = True):
        assert isinstance(batch_size, int), f"Batch size must be an integer, not {type(batch_size)}."
        assert batch_size > 0, f"Batch size must be positive, not {batch_size}."
        assert isinstance(repeat, bool), f"Repeat must be a boolean, not {type(repeat)}."
        
        if dataset == "train":
            chunk = self.training_data()
        elif dataset == "test":
            raise NotImplementedError("Test data is not lazy loaded.")
        elif dataset == "validation":
            chunk = self.validation_data()
        else:
            raise ValueError(f"Unknown dataset {dataset}, must be one of 'test', 'train', 'validation'.")

        while True:
            chunk_len = chunk.num_rows
            
            for i in range(0, chunk_len, batch_size):
                if i + batch_size > chunk_len:
                    logger.debug(f"End of chunk, breaking")
                    break
                # get the next batch
                b = chunk.slice(offset=i, length=batch_size)
                arr = self.load_to_device(b)
                res = {
                    **unpack_features(arr['features']),
                    'label': arr['label'],
                }
                assert res["label"].shape == (batch_size,), f"Label shape {res['label'].shape} does not match batch size {batch_size}."
                yield res

            if not repeat:
                break

    def _batched_test(self, batch_size: int):
        assert isinstance(batch_size, int), f"Batch size must be an integer, not {type(batch_size)}."
        assert batch_size > 0, f"Batch size must be positive, not {batch_size}."
        
        chunk_len = self.samples("test")
        len_2 = len(self.test_jax['features'])
        assert chunk_len == len_2, f"Chunk length {chunk_len:,} does not match test_jax length {len_2:,}."
        
        for i in range(0, chunk_len, batch_size):
            if i + batch_size > chunk_len:
                logger.debug(f"End of chunk, breaking")
                break
            # get the next batch
            res =  {
                **unpack_features(self.test_jax['features'][i:i+batch_size]),
                'label': self.test_jax['label'][i:i+batch_size],
            }
            assert res["label"].shape == (batch_size,), f"Label shape {res['label'].shape} does not match batch size {batch_size}."
            yield res
    
def main() -> None:
    arg_parser = argparse.ArgumentParser(description="Train a neural network to predict evals from feature vectors.")
    arg_parser.add_argument("train", type=str, help="Path to the training data.")
    arg_parser.add_argument("output", type=str, help="Path to the output file.")
    arg_parser.add_argument("iterations", type=int, help="Number of iterations to train for.")
    arg_parser.add_argument("--regularization", type=float, default=0.0, help="L2 regularization strength.")
    args = arg_parser.parse_args()

    train_file = Path(args.train)
    assert train_file.exists(), f"Train file {train_file} does not exist."

    checkpoint_path = Path(args.output)
    assert checkpoint_path.exists(), f"Checkpoint path {checkpoint_path} does not exist."
    dated_checkpoint_path = checkpoint_path / f"{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    dated_checkpoint_path.mkdir()
    assert dated_checkpoint_path.exists(), f"Failed to create checkpoint path {checkpoint_path}."
    checkpoint_path = dated_checkpoint_path.resolve()

    iterations = args.iterations
    assert isinstance(iterations, int), f"Iterations must be an integer, not {type(iterations)}."
    assert iterations > 0, f"Iterations must be positive, not {iterations}."

    logger.setLevel(logging.INFO)
    # logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.FileHandler(checkpoint_path/"training.log"))
    # set the format to include the time (based on perf timer)

    logger.info(f"""
        Train file: {train_file}
        Output path: {checkpoint_path}
        Iterations: {iterations}
    """)
    # version info of the libraries
    logger.info(f"JAX version: {jax.__version__}")
    logger.info(f"Flax version: {flax.__version__}")
    logger.info(f"Optax version: {optax.__version__}")
    logger.info(f"Orbax version: {ocp.__version__}")
    logger.info(f"NumPy version: {np.__version__}")
    logger.info(f"PyArrow version: {pa.__version__}")
    logger.info(f"devices: {jax.devices()}")
    for device in jax.devices():
        logger.info(f"Device: {device}, {device.device_kind}, {device.id}, {device.platform}, {device.host_id}")

    script_path = Path(__file__).resolve()
    shutil.copy2(script_path, dated_checkpoint_path / script_path.name)
    
    logistic_scaling = 400.
    logger.info(f"Logistic scaling: {logistic_scaling:.0f}cp")
    regularization = jnp.exp(args.regularization) # exponentiate so I don't have to write 0.000001 like a caveman
    logger.info(f"Regularization: {regularization:.5g}")

    seed = 314159

    data_loader = FileSource(train_file, seed, logistic_scaling)
    logger.info(f"Data loader initialised: {len(data_loader)} samples, {data_loader.samples("test")} test, {data_loader.samples("train")} train") 

    batch_size = 1<<14
    gradient_accumulation_steps = 1
    checkpoint_freq = 1<<15
    steps_per_epoch = (len(data_loader)-1) // batch_size + 1
    validation_freq = 1<<11


    model = AdmeteModel(rngs=nnx.Rngs(seed))
    total_steps = iterations // gradient_accumulation_steps
    warmup_steps = total_steps // 10  # 10% warmup
    end_steps = total_steps // 10  # 10% end
    peak_lr = 4e-3
    end_lr = 2e-6

    schedule = optax.join_schedules([
        optax.linear_schedule(0, peak_lr, warmup_steps),
        optax.cosine_decay_schedule(peak_lr, total_steps - warmup_steps - end_steps, end_lr),
        optax.constant_schedule(end_lr),
    ], boundaries=[warmup_steps, total_steps - end_steps])

    opt = optax.adam(schedule)
    opt = optax.MultiSteps(opt, gradient_accumulation_steps) # Accumulate gradients for a few steps before updating the model, (we can't fit the whole batch in memory)
    optimizer = nnx.Optimizer(model, opt)    
    # optimizer_base = nnx.Optimizer(model, opt, wrt=base_params) # optimizer for the base parameters only

    train_metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average('loss'),
    )
    test_metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average('loss'),
        invloss=nnx.metrics.Average('invloss'),
        winning_ratio=nnx.metrics.Average('winning_ratio'),
        losing_ratio=nnx.metrics.Average('losing_ratio'),
    )

    metrics_history:dict[str, list[float]] = {}
    for metric in train_metrics.compute().keys():
        metrics_history[f'train_{metric}'] = []
    for metric in test_metrics.compute().keys():
        metrics_history[f'test_{metric}'] = []

    checkpointer = ocp.StandardCheckpointer()

    with open(checkpoint_path / "config.txt", "w") as f:
        f.write(f"Train file: {train_file}\n")
        f.write(f"Output path: {checkpoint_path}\n")
        f.write(f"Iterations: {iterations}\n")
        # f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Gradient accumulation steps: {gradient_accumulation_steps}\n")
        f.write(f"LR schedule: {schedule}\n")
        f.write(f"Logistic scaling: {logistic_scaling:.0f}\n")
        f.write(f"Regularization: {regularization:.5g}\n")
        f.write(f"Validation frequency: {validation_freq}\n")
        f.write(f"Checkpoint frequency: {checkpoint_freq}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Training samples: {data_loader.samples("train")}\n")
        f.write(f"Training samples: {data_loader.samples("test")}\n")
        f.write(f"Model:\n")
        f.write(f"{inspect.getsource(AdmeteModel)}\n")

    with open(checkpoint_path / "metrics.csv", "w") as f:
        f.write("step,epoch")
        for metric in metrics_history.keys():
            f.write(f",{metric}")
        f.write("\n")

    # Train
    logger.info(f"Training for {iterations} iterations")
    for step, batch in tqdm(enumerate(data_loader.batched(batch_size, dataset="train")), total=iterations):
        logger.debug(f"Step {step:>{int(np.log10(iterations)+1)}}")
        # freeze_delta = step < iterations // 5 # freeze the delta parameters for the 20%
        # opt = optimizer_base if freeze_delta else optimizer
        train_step(model, optimizer, train_metrics, batch, regularization, freeze_delta=False)
        if ((step % validation_freq == 0) or (step == iterations-1)):
            epoch = step / steps_per_epoch
            for metric, value in train_metrics.compute().items():  # Compute the metrics.
                metrics_history[f'train_{metric}'].append(float(value))  # Record the metrics.
            train_metrics.reset()  # Reset the metrics for the test set.
            # test batch
            for test_batch in data_loader.batched(batch_size, dataset="test", repeat=False):
                eval_step(model, test_metrics, test_batch)
            for metric, value in test_metrics.compute().items():
                metrics_history[f'test_{metric}'].append(float(value))
            test_metrics.reset()

            logger.info(f"Epoch {epoch:>{int(np.log10((iterations-1)/steps_per_epoch+1)+6)}.5f} / step {step:>{int(np.log10(iterations)+1)}}: Train loss: {metrics_history['train_loss'][-1]:.5f}, Test loss: {metrics_history['test_loss'][-1]:.5f}")
            # save metrics
            with open(checkpoint_path / "metrics.csv", "a") as f:
                f.write(f"{step},{epoch:.5f}")
                for metric in metrics_history.keys():
                    f.write(f",{metrics_history[metric][-1]}")
                f.write("\n")
        if (step != 0) and step % checkpoint_freq == 0:
            logger.info(f"Checkpointing at step {step}")
            _, state = nnx.split(model)
            checkpointer.save(str(checkpoint_path/f"state_{step}"), state)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
        if step >= iterations-1:
            break
    logger.info("Training complete")
    # run the validation set for hyperparameter tuning
    test_metrics.reset()
    data_loader.clear_in_memory() # we're going to load the validation set into memory, we don't need the training set anymore, so let's free up the memory
    for test_batch in data_loader.batched(batch_size, dataset="validation", repeat=False):
        eval_step(model, test_metrics, test_batch)
    for metric, value in test_metrics.compute().items():
        logger.info(f"Validation {metric}: {value:.5f}")
    
    # Save model
    state = nnx.state(model)

    checkpointer.save(str(checkpoint_path/"state"), state)
    logger.info(f"Model saved to {checkpoint_path}")
    
    checkpointer.wait_until_finished()
    logging.shutdown()

if __name__ == "__main__":
    main()
