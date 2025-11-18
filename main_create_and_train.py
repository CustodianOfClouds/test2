import json
import numpy as np
import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from layer import Layer
from neural_network import NeuralNet
from training import Training
from activation_functions import ActivationFunction

############################################################################################################
# CONFIGURATION SELECTOR - Change this number to select which model to train (1-7)
############################################################################################################
CONFIG_TO_RUN = 7

############################################################################################################
# CONFIGURATION DEFINITIONS
############################################################################################################

def get_configuration(config_num):
    """
    Returns the configuration for the specified config number.
    Each config showcases different combinations of:
    - Activation functions (ReLU, Leaky ReLU, ELU, Tanh, Sigmoid, Softmax, Linear)
    - Cost functions (MSE, MAE, Binary Cross-Entropy, Categorical Cross-Entropy)
    - Weight initializations (Normal, Xavier, He, Uniform)
    - Bias initializations (Zeros, Ones, Constant, Normal)
    """

    configs = {
        ############################################################################################################
        # CONFIG 1: RGB Red Color Classification - SHOWCASES NORMAL INITIALIZATION
        ############################################################################################################
        1: {
            'name': 'RGB Red Color Classification',
            'description': 'Classify RGB colors as "red" or "not red"',
            'architecture': '3 → 10 → 5 → 2',
            'details': 'Normal weight init (std=0.01) + MAE cost + ELU hidden layers',
            'layers': lambda: [
                Layer(3, 3, 'input'),
                Layer(3, 10, 'hidden',
                      activation_func=ActivationFunction.elu,
                      activation_params={'alpha': 1.0},
                      weight_init='normal', weight_init_params={'std': 0.01},
                      bias_init='zeros'),
                Layer(10, 5, 'hidden',
                      activation_func=ActivationFunction.elu,
                      activation_params={'alpha': 1.0},
                      weight_init='normal', weight_init_params={'std': 0.01},
                      bias_init='zeros'),
                Layer(5, 2, 'output',
                      activation_func=ActivationFunction.tanh)
            ],
            'data_file': 'color_data.json',
            'input_key': 'RGB_Values',
            'output_key': 'Is_Red',
            'learning_rate': 0.0001,
            'num_epochs': 500,
            'num_samples': 600,
            'cost_function': 'mae',  # Mean Absolute Error
            'save_file': 'model_red.json'
        },

        ############################################################################################################
        # CONFIG 2: XOR Problem - SHOWCASES RELU + XAVIER INIT
        ############################################################################################################
        2: {
            'name': 'XOR Problem',
            'description': 'Learn XOR function (proves you need hidden layers!)',
            'architecture': '2 → 6 → 4 → 2',
            'details': 'Xavier init + ReLU hidden + Tanh output + MSE',
            'layers': lambda: [
                Layer(2, 2, 'input'),
                Layer(2, 6, 'hidden',
                      activation_func=ActivationFunction.relu,
                      weight_init='xavier',
                      bias_init='zeros'),
                Layer(6, 4, 'hidden',
                      activation_func=ActivationFunction.relu,
                      weight_init='xavier',
                      bias_init='zeros'),
                Layer(4, 2, 'output',
                      activation_func=ActivationFunction.tanh)
            ],
            'data_file': 'xor_data.json',
            'input_key': 'Input_Values',
            'output_key': 'Output_Values',
            'learning_rate': 0.001,
            'num_epochs': 500,
            'num_samples': 600,
            'cost_function': 'mse',  # Mean Squared Error
            'save_file': 'model_xor.json'
        },

        ############################################################################################################
        # CONFIG 3: Sine Wave Classification - SHOWCASES SWISH + UNIFORM XAVIER INIT
        ############################################################################################################
        3: {
            'name': 'Sine Wave Classification',
            'description': 'Classify points as above or below y = sin(x)',
            'architecture': '2 → 16 → 12 → 8 → 2',
            'details': 'Uniform Xavier + Swish (alpha=1.2) + Sigmoid output + Binary CE',
            'layers': lambda: [
                Layer(2, 2, 'input'),
                Layer(2, 16, 'hidden',
                      activation_func=ActivationFunction.swish,
                      activation_params={'alpha': 1.2},
                      weight_init='uniform_xavier',
                      bias_init='constant', bias_init_params={'value': 0.1}),
                Layer(16, 12, 'hidden',
                      activation_func=ActivationFunction.swish,
                      activation_params={'alpha': 1.2},
                      weight_init='uniform_xavier',
                      bias_init='constant', bias_init_params={'value': 0.1}),
                Layer(12, 8, 'hidden',
                      activation_func=ActivationFunction.swish,
                      activation_params={'alpha': 1.2},
                      weight_init='uniform_xavier',
                      bias_init='zeros'),
                Layer(8, 2, 'output',
                      activation_func=ActivationFunction.sigmoid)  # Sigmoid for Binary CE
            ],
            'data_file': 'sine_data.json',
            'input_key': 'Input_Values',
            'output_key': 'Output_Values',
            'learning_rate': 0.0001,
            'num_epochs': 5000,
            'num_samples': 600,
            'cost_function': 'binary_crossentropy',
            'save_file': 'model_sine.json'
        },

        ############################################################################################################
        # CONFIG 4: Checkerboard Pattern - SHOWCASES MISH + HE INIT
        ############################################################################################################
        4: {
            'name': 'Checkerboard Pattern',
            'description': 'Classify grid squares as black or white (chess board)',
            'architecture': '2 → 20 → 16 → 12 → 2',
            'details': 'He init + Mish activation + Tanh output + MSE',
            'layers': lambda: [
                Layer(2, 2, 'input'),
                Layer(2, 20, 'hidden',
                      activation_func=ActivationFunction.mish,
                      weight_init='he',
                      bias_init='zeros'),
                Layer(20, 16, 'hidden',
                      activation_func=ActivationFunction.mish,
                      weight_init='he',
                      bias_init='zeros'),
                Layer(16, 12, 'hidden',
                      activation_func=ActivationFunction.mish,
                      weight_init='he',
                      bias_init='zeros'),
                Layer(12, 2, 'output',
                      activation_func=ActivationFunction.tanh)
            ],
            'data_file': 'checkerboard_data.json',
            'input_key': 'Input_Values',
            'output_key': 'Output_Values',
            'learning_rate': 0.0003,
            'num_epochs': 5000,
            'num_samples': 600,
            'cost_function': 'mse',
            'save_file': 'model_checkerboard.json'
        },

        ############################################################################################################
        # CONFIG 5: Quadrant Classification - SHOWCASES GELU + SELU
        ############################################################################################################
        5: {
            'name': 'Quadrant Classification (MULTI-CLASS)',
            'description': 'Classify which quadrant a point is in (4 classes)',
            'architecture': '2 → 12 → 10 → 4',
            'details': 'GELU + SELU mixed + Tanh output + MAE cost',
            'layers': lambda: [
                Layer(2, 2, 'input'),
                Layer(2, 12, 'hidden',
                      activation_func=ActivationFunction.gelu,
                      weight_init='xavier',
                      bias_init='normal', bias_init_params={'std': 0.01}),
                Layer(12, 10, 'hidden',
                      activation_func=ActivationFunction.selu,
                      activation_params={'alpha': 1.67326324, 'scale': 1.05070098},
                      weight_init='xavier',
                      bias_init='zeros'),
                Layer(10, 4, 'output',
                      activation_func=ActivationFunction.tanh)  # 4 classes
            ],
            'data_file': 'quadrant_data.json',
            'input_key': 'Input_Values',
            'output_key': 'Output_Values',
            'learning_rate': 0.0005,
            'num_epochs': 1000,
            'num_samples': 600,
            'cost_function': 'mae',  # Mean Absolute Error for multi-class
            'save_file': 'model_quadrant.json'
        },

        ############################################################################################################
        # CONFIG 6: House Price Regression - SHOWCASES LINEAR OUTPUT + PARAMETRIC ELU
        ############################################################################################################
        6: {
            'name': 'House Price Regression (LINEAR OUTPUT)',
            'description': 'Predict house prices (regression with unbounded outputs)',
            'architecture': '3 → 12 → 10 → 1',
            'details': 'Parametric ELU + He init + Linear output + MSE',
            'layers': lambda: [
                Layer(3, 3, 'input'),
                Layer(3, 12, 'hidden',
                      activation_func=ActivationFunction.parametric_elu,
                      activation_params={'alpha': 1.5, 'beta': 1.0},
                      weight_init='he',
                      bias_init='zeros'),
                Layer(12, 10, 'hidden',
                      activation_func=ActivationFunction.parametric_elu,
                      activation_params={'alpha': 1.5, 'beta': 1.0},
                      weight_init='he',
                      bias_init='zeros'),
                Layer(10, 1, 'output',
                      activation_func=ActivationFunction.linear)  # Linear for unbounded regression
            ],
            'data_file': 'linear_regression_data.json',
            'input_key': 'Input_Values',
            'output_key': 'Output_Values',
            'learning_rate': 0.001,
            'num_epochs': 1000,
            'num_samples': 700,
            'cost_function': 'mse',
            'save_file': 'model_linear_regression.json'
        },

        ############################################################################################################
        # CONFIG 7: Iris Flower Classification - SHOWCASES SOFTMAX + CATEGORICAL CE (REQUIRED)
        ############################################################################################################
        7: {
            'name': 'Iris Flower Classification (SOFTMAX + CATEGORICAL CE)',
            'description': 'Classify iris flowers into 3 species',
            'architecture': '4 → 14 → 10 → 3',
            'details': 'Leaky ReLU (alpha=0.02) + He init + Softmax + Categorical CE',
            'layers': lambda: [
                Layer(4, 4, 'input'),
                Layer(4, 14, 'hidden',
                      activation_func=ActivationFunction.leaky_relu,
                      activation_params={'alpha': 0.02},
                      weight_init='he',
                      bias_init='zeros'),
                Layer(14, 10, 'hidden',
                      activation_func=ActivationFunction.leaky_relu,
                      activation_params={'alpha': 0.02},
                      weight_init='he',
                      bias_init='zeros'),
                Layer(10, 3, 'output',
                      activation_func=ActivationFunction.softmax)  # SOFTMAX REQUIRED
            ],
            'data_file': 'iris_data.json',
            'input_key': 'Input_Values',
            'output_key': 'Output_Values',
            'learning_rate': 0.001,
            'num_epochs': 1000,
            'num_samples': 700,
            'cost_function': 'categorical_crossentropy',  # REQUIRED for softmax
            'save_file': 'model_iris.json'
        },
    }

    if config_num not in configs:
        raise ValueError(f"Invalid configuration number: {config_num}. Must be 1-7.")

    return configs[config_num]


############################################################################################################
# MAIN EXECUTION
############################################################################################################

# Get the selected configuration
config = get_configuration(CONFIG_TO_RUN)

# Print configuration info
print("=" * 70)
print(f"TRAINING: {config['name']}")
print("=" * 70)
print(f"Description: {config['description']}")
print(f"Architecture: {config['architecture']}")
print(f"Details: {config['details']}")
print("=" * 70)
print()

# Build the neural network
neural_net = NeuralNet()
for layer in config['layers']():
    neural_net.add_layer(layer)

# Load training data
data_file = os.path.join(os.path.dirname(__file__), "data", config['data_file'])
with open(data_file, "r") as file:
    data = json.load(file)

input_data = np.array(data[config['input_key']])
target_data = np.array(data[config['output_key']])

print(f"Loaded {len(input_data)} training samples from {config['data_file']}")
print(f"Learning rate: {config['learning_rate']}")
print(f"Epochs: {config['num_epochs']}")
print(f"Cost function: {config['cost_function']}")
print()

# Create a Training object
training = Training(neural_net,
                   learning_rate=config['learning_rate'],
                   clip_value=5,
                   cost_function=config['cost_function'])

# Train the neural network
training.train(input_data, target_data,
              epochs=config['num_epochs'],
              samples_per_epoch=config['num_samples'])

# Save the neural net
save_file = os.path.join(os.path.dirname(__file__), "models", config['save_file'])
neural_net.save(save_file)

print()
print("=" * 70)
print(f"Training complete! Model saved to {config['save_file']}")
print("=" * 70)
