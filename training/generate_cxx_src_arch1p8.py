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

import importlib
from pathlib import Path
import sys
import importlib
import sys
import orbax.checkpoint as ocp

import orbax.checkpoint as ocp
from flax import nnx
from jax import numpy as jnp

output_version = '2025-09-27_11-59-17' # 0.633750 separate kings weights, R=1, 200K
outputs_loc = Path('~/share/junebug/chess/training/checkpoints/').expanduser()

outputs = outputs_loc / output_version


def load_training_module(output_path):
    """Load the training.py module from the checkpoint directory"""
    training_path = output_path / 'training.py'
    spec = importlib.util.spec_from_file_location("training_checkpoint", training_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["training_checkpoint"] = module
    spec.loader.exec_module(module)
    return module


training = load_training_module(outputs)

def fold_batchnorm_layer(layer, batchnorm):
    """
    Fold the batch normalization layer into the previous layer.
    """
    if batchnorm is None:
        return layer.kernel, layer.bias
    
    kernel, bias = layer.kernel, layer.bias
    if bias is None:
        bias = jnp.zeros(kernel.shape[1])
    scale = batchnorm.scale if batchnorm.scale is not None else 1.0
    mean = batchnorm.mean
    var = batchnorm.var
    bn_bias = batchnorm.bias if batchnorm.bias is not None else 0.0
    
    # Calculate the folded kernel and bias
    # y = x W + b
    # Y = (y - mean) * scale / sqrt(var + epsilon) + bias
    # Y = x W' + b'
    # a = scale / sqrt(var + epsilon)
    # W' = W * a
    # b' = bias + (b - mean) * a
    
    epsilon = batchnorm.epsilon
    rescale = scale / jnp.sqrt(var + epsilon)
    
    folded_kernel = kernel * rescale
    folded_bias = bn_bias + (bias - mean) * rescale
    
    return folded_kernel, folded_bias


# Create model instance with same architecture
abstract_model = training.AdmeteModel(rngs=training.nnx.Rngs(0))
graphdef, abstract_state = nnx.split(abstract_model)

checkpointer = ocp.StandardCheckpointer()
state = checkpointer.restore(str(outputs/"state"), abstract_state)
model = nnx.merge(graphdef, state)

import os
from pathlib import Path
from jinja2 import Environment, BaseLoader

acc_shift = 4
logistic_scaling = training.LOGISTIC_SCALING if hasattr(training, 'LOGISTIC_SCALING') else 400.0

# Jinja2 template for header file
HEADER_TEMPLATE = """#pragma once
#include <memory>

#include <network.hpp>

namespace Neural {

// Model generated from training run {{ output_version }}

constexpr uint8_t ACC_SHIFT = {{ acc_shift }};
constexpr size_t N_ACCUMULATED = {{ accumulator_size }};
constexpr nn_t LOGISTIC_SCALING = {{ logistic_scaling }}f;
static_assert(N_FEATURES == {{ feature_size }}, "Feature size mismatch");

namespace generated {

std::unique_ptr<FloatingAccumulatorLayer<nn_t, N_FEATURES, N_ACCUMULATED>> gen_accumulator();

{% for layer in layers %}
std::unique_ptr<LinearLayer<nn_t, {{ layer.input_size }}, {{ layer.output_size }}>> gen_layer_{{ loop.index0 }}();
{% endfor %}

} // namespace generated

typedef Accumulator<N_FEATURES, N_ACCUMULATED, ACC_SHIFT> accumulator_t;
accumulator_t get_accumulator();

typedef Network<N_FEATURES, N_ACCUMULATED, ACC_SHIFT, {{ layer_sizes }}> network_t;
network_t get_network();

} // namespace Neural
"""

# Jinja2 template for implementation file
IMPL_TEMPLATE = """#include "weights.hpp"

namespace Neural {
namespace generated {

std::unique_ptr<FloatingAccumulatorLayer<nn_t, N_FEATURES, N_ACCUMULATED>> gen_accumulator() {
    constexpr nn_t w[] = {
    {% for chunk in accumulator.weights_chunks %}
        {{ chunk }},
    {% endfor %}
    };
    constexpr nn_t b[] = {
    {% for chunk in accumulator.bias_chunks %}
        {{ chunk }},
    {% endfor %}
    };
    std::unique_ptr<FloatingAccumulatorLayer<nn_t, N_FEATURES, N_ACCUMULATED>> layer = std::make_unique<FloatingAccumulatorLayer<nn_t, N_FEATURES, N_ACCUMULATED>>(w, b);
    return layer;
}

{% for layer in layers %}
std::unique_ptr<LinearLayer<nn_t, {{ layer.input_size }}, {{ layer.output_size }}>> gen_layer_{{ loop.index0 }}() {
\tconstexpr nn_t w[] = {
{% for chunk in layer.weights_chunks %}
\t\t{{ chunk }},
{% endfor %}
\t};
\tconstexpr nn_t b[] = {
{% for chunk in layer.bias_chunks %}
\t\t{{ chunk }},
{% endfor %}
\t};
    return std::make_unique<LinearLayer<nn_t, {{ layer.input_size }}, {{ layer.output_size }}>>(w, b);
}

{% endfor %}
} // namespace generated

accumulator_t get_accumulator() {
    auto layer = generated::gen_accumulator();
    return accumulator_t(std::move(layer));
}

network_t get_network() {
    network_t net;
{% for i in range(num_layers) %}
    net.set_layer<{{ i }}>(generated::gen_layer_{{ i }}());
{% endfor %}
    return net;
}

} // namespace Neural
"""

def format_array_chunks(arr, chunk_size=8, precision=5):
    """Format array into chunks of comma-separated values."""
    chunks = []
    for i in range(0, len(arr), chunk_size):
        chunk = arr[i:min(i + chunk_size, len(arr))]
        line = ', '.join(f"{val:.{precision}f}f" for val in chunk)
        chunks.append(line)
    return chunks

def model_to_floating_cpp(model, output_dir: str, output_version: str) -> tuple[str, str]:
    # Set up Jinja2 environment
    env = Environment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=False)
    header_template = env.from_string(HEADER_TEMPLATE)
    impl_template = env.from_string(IMPL_TEMPLATE)
    
    # Extract model information
    accumulator_size = model.accumulator_baseline.kernel.shape[1]
    feature_size = model.accumulator_baseline.kernel.shape[0] // 2
    
    # Prepare accumulator data
    acc_weights, acc_bias = fold_batchnorm_layer(
        model.accumulator_baseline, 
        model.accumulator_batchnorm
    )
    accumulator_data = {
        'weights_chunks': format_array_chunks(acc_weights.flatten()),
        'bias_chunks': format_array_chunks(acc_bias)
    }
    
    # Prepare layer data
    layers_data = []
    for i, layer in enumerate(model.layers):
        batchnorm = model.hidden_batchnorm if i == 0 else None
        weights, bias = fold_batchnorm_layer(layer, batchnorm)
        
        layer_info = {
            'input_size': layer.kernel.shape[0],
            'output_size': layer.kernel.shape[1],
            'weights_chunks': format_array_chunks(weights.flatten()),
            'bias_chunks': format_array_chunks(bias)
        }
        layers_data.append(layer_info)
    
    # Prepare template context
    layer_sizes = ', '.join(str(layer.kernel.shape[1]) for layer in model.layers)
    
    header_context = {
        'output_version': output_version,
        'acc_shift': acc_shift,
        'accumulator_size': accumulator_size,
        'logistic_scaling': f"{logistic_scaling:.1f}",
        'feature_size': feature_size,
        'layers': layers_data,
        'layer_sizes': layer_sizes
    }
    
    impl_context = {
        'accumulator': accumulator_data,
        'layers': layers_data,
        'num_layers': len(model.layers)
    }
    
    # Render templates
    header_content = header_template.render(**header_context)
    impl_content = impl_template.render(**impl_context)
    
    # Write to files
    header_path = os.path.join(output_dir, "weights.hpp")
    impl_path = os.path.join(output_dir, "weights.cxx")
    
    with open(header_path, 'w') as f:
        f.write(header_content)
    
    with open(impl_path, 'w') as f:
        f.write(impl_content)
    
    return header_path, impl_path

# Usage
output = Path("./generated/")
output.mkdir(exist_ok=True)
model_to_floating_cpp(model, output, output_version)