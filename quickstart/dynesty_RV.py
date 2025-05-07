import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import traceback

# Create output directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Load the results
with open('HD189567_HARPS/dynesty/results.p', 'rb') as f:
    results = pickle.load(f)

# Load the model container
with open('HD189567_HARPS/dynesty/model_container.p', 'rb') as f:
    mc = pickle.load(f)

# Extract samples
samples = results.samples
print(f"Samples shape: {samples.shape}")

# Calculate weights from logwt
logwt = results.logwt
weights = np.exp(logwt - logwt.max())
weights /= weights.sum()

# Get the maximum likelihood sample
max_idx = np.argmax(results.logl)
theta_max = results.samples[max_idx]

# Get the weighted median parameters
theta_median = np.zeros(samples.shape[1])
for i in range(samples.shape[1]):
    # Calculate weighted median
    sorted_idx = np.argsort(samples[:, i])
    sorted_weights = weights[sorted_idx]
    cumulative_weights = np.cumsum(sorted_weights)
    cumulative_weights /= cumulative_weights[-1]  # Normalize
    theta_median[i] = np.interp(0.5, cumulative_weights, samples[sorted_idx, i])

# Print model container information
print("\nModel container information:")
print(f"Type: {type(mc)}")
print(f"Attributes: {dir(mc)}")

# Check bounds type
if hasattr(mc, 'bounds'):
    print(f"\nBounds type: {type(mc.bounds)}")
    if isinstance(mc.bounds, np.ndarray):
        print(f"Bounds shape: {mc.bounds.shape}")
        if len(mc.bounds.shape) == 2:
            print(f"Bounds: {mc.bounds}")

# Try different ways to get parameter names
param_names = []

# Option 1: Try mc.pam_names
if hasattr(mc, 'pam_names') and mc.pam_names:
    param_names = mc.pam_names
    print(f"Found {len(param_names)} parameters from mc.pam_names")

# Option 2: Try mc.variable_list
if not param_names and hasattr(mc, 'variable_list') and mc.variable_list:
    param_names = mc.variable_list
    print(f"Found {len(param_names)} parameters from mc.variable_list")

# Option 3: Try mc.variables
if not param_names and hasattr(mc, 'variables'):
    if hasattr(mc.variables, 'keys'):
        param_names = list(mc.variables.keys())
        print(f"Found {len(param_names)} parameters from mc.variables.keys()")

# Option 4: Check if there's a planets attribute and extract names
if not param_names and hasattr(mc, 'planets'):
    try:
        planet_params = []
        for planet in mc.planets:
            planet_params.extend([f"{planet}_P", f"{planet}_K", f"{planet}_e", f"{planet}_omega", f"{planet}_M0"])
        param_names = planet_params
        print(f"Created {len(param_names)} parameters from planet names")
    except:
        pass

# If all else fails, create generic names
if not param_names:
    param_names = [f"param_{i}" for i in range(samples.shape[1])]
    print(f"Created {len(param_names)} generic parameter names")

print("\nParameter names:", param_names)

print("\nMaximum likelihood parameters:")
for name, value in zip(param_names, theta_max):
    print(f"  {name}: {value}")

print("\nMedian parameters:")
for name, value in zip(param_names, theta_median):
    print(f"  {name}: {value}")

# Convert array to dictionary for PyORBIT
theta_dict = {name: value for name, value in zip(param_names, theta_median)}

# Get the dataset
dataset_names = list(mc.dataset_dict.keys())
print(f"\nAvailable datasets: {dataset_names}")

# Print information about the models
if hasattr(mc, 'models'):
    print("\nModels information:")
    print(f"Models type: {type(mc.models)}")
    print(f"Models keys: {list(mc.models.keys())}")

# Explore the dataset structure
print("\nExploring dataset structure:")
for dataset_name in dataset_names:
    print(f"\nDataset: {dataset_name}")
    dataset = mc.dataset_dict[dataset_name]
    print(f"  Type: {type(dataset)}")
    
    # Check if the dataset has the expected attributes
    has_x = hasattr(dataset, 'x')
    has_y = hasattr(dataset, 'y')
    has_e = hasattr(dataset, 'e')
    print(f"  Has x: {has_x}, Has y: {has_y}, Has e: {has_e}")
    
    if has_x and has_y and has_e:
        print(f"  x shape: {dataset.x.shape}")
        print(f"  y shape: {dataset.y.shape}")
        print(f"  e shape: {dataset.e.shape}")
        print(f"  x range: {np.min(dataset.x)} to {np.max(dataset.x)}")
        print(f"  y range: {np.min(dataset.y)} to {np.max(dataset.y)}")

# Create a simple plot of just the data
for dataset_name in dataset_names:
    try:
        # Get the observed data
        dataset = mc.dataset_dict[dataset_name]
        x_data = dataset.x
        y_data = dataset.y
        e_data = dataset.e
        
        # Create a simple plot
        plt.figure(figsize=(10, 6))
        plt.errorbar(x_data, y_data, yerr=e_data, fmt='o', alpha=0.7)
        plt.xlabel('Time (days)')
        plt.ylabel('RV (m/s)')
        plt.title(f'HD189567 {dataset_name} Data')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'plots/HD189567_{dataset_name}_data.png', dpi=300)
        print(f"Data plot saved as plots/HD189567_{dataset_name}_data.png")
        
    except Exception as e:
        print(f"Error creating data plot for {dataset_name}: {str(e)}")
        traceback.print_exc()

# Try to create model plots for each dataset
for dataset_name in dataset_names:
    try:
        print(f"\nProcessing dataset for model: {dataset_name}")
        
        # Get the observed data
        dataset = mc.dataset_dict[dataset_name]
        x_data = dataset.x
        y_data = dataset.y
        e_data = dataset.e
        
        print(f"Data loaded: {len(x_data)} points")
        
        # Find the correct model key
        model_key = None
        for key in mc.models.keys():
            if dataset_name in key or key in dataset_name:
                model_key = key
                break
        
        if model_key is None:
            print(f"Could not find model for {dataset_name}. Available models: {list(mc.models.keys())}")
            continue
        
        print(f"Using model key: {model_key}")
        
        # Initialize the model
        model = mc.models[model_key]
        print(f"Model type: {type(model)}")
        print(f"Model attributes: {dir(model)}")
        
        # Check if model has the model method
        if not hasattr(model, 'model'):
            print(f"Model does not have 'model' method. Available methods: {[m for m in dir(model) if not m.startswith('_')]}")
            continue
        
        # Generate model using median parameters
        model_x = np.linspace(np.min(x_data), np.max(x_data), 1000)
        
        # Evaluate the model
        print("Evaluating model...")
        try:
            # Try calling model with dictionary
            y_model = model.model(theta_dict, x_data)
            y_curve = model.model(theta_dict, model_x)
        except Exception as e:
            print(f"Error with dictionary parameters: {str(e)}")
            # Try calling model with array
            try:
                y_model = model.model(theta_median, x_data)
                y_curve = model.model(theta_median, model_x)
            except Exception as e2:
                print(f"Error with array parameters: {str(e2)}")
                raise
        
        # Create the plot
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(2, 1, height_ratios=[3, 1])
        
        # Top panel: data and model
        ax1 = plt.subplot(gs[0])
        ax1.errorbar(x_data, y_data, yerr=e_data, fmt='o', alpha=0.5, label='Data')
        ax1.plot(model_x, y_curve, 'r-', alpha=0.8, label='Model')
        ax1.set_ylabel('RV (m/s)')
        ax1.set_title(f'HD189567 {dataset_name} Model')
        ax1.legend()
        
        # Bottom panel: residuals
        ax2 = plt.subplot(gs[1])
        ax2.errorbar(x_data, y_data - y_model, yerr=e_data, fmt='o', alpha=0.5)
        ax2.axhline(y=0, color='r', linestyle='-', alpha=0.8)
        ax2.set_xlabel('Time (days)')
        ax2.set_ylabel('Residuals (m/s)')
        
        plt.tight_layout()
        plt.savefig(f'plots/HD189567_{dataset_name}_model.png', dpi=300)
        print(f"RV model plot saved as plots/HD189567_{dataset_name}_model.png")
        
    except Exception as e:
        print(f"Error creating plot for {dataset_name}: {str(e)}")
        traceback.print_exc()

# Create a corner plot for the parameters
try:
    import corner
    
    # Create corner plot
    fig = corner.corner(
        samples,
        weights=weights,
        labels=param_names,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3f"
    )
    
    plt.savefig('plots/HD189567_dynesty_corner.png', dpi=300, bbox_inches='tight')
    print("Corner plot saved as plots/HD189567_dynesty_corner.png")
    
except Exception as e:
    print(f"Error creating corner plot: {str(e)}")
    traceback.print_exc()
