from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers.pid import Controller
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

class PIDOptimizer:
    def __init__(self, model_path, data_path, train_routes, val_routes):
        self.model_path = model_path
        self.data_path = data_path
        self.train_routes = train_routes
        self.val_routes = val_routes
        self.optimization_history = []
    
    def objective(self, x):
        """Objective function to minimize"""
        p, i, d = x
        print(f"\nTrying parameters: P={p:.3f}, I={i:.3f}, D={d:.3f}")
        
        # Evaluate on both training and validation sets
        train_stats = evaluate_pid_params(self.model_path, self.data_path, p, i, d, self.train_routes, num_routes=50)
        val_stats = evaluate_pid_params(self.model_path, self.data_path, p, i, d, self.val_routes, num_routes=50)
        
        # Calculate combined cost with regularization
        train_val_diff = abs(train_stats['total_cost']['mean'] - val_stats['total_cost']['mean'])
        regularization = 0.3 * train_val_diff
        
        # Add stability penalty
        stability_penalty = 0.2 * (train_stats['total_cost']['std'] + val_stats['total_cost']['std'])
        
        total_cost = val_stats['total_cost']['mean'] + regularization + stability_penalty
        
        # Store optimization history
        self.optimization_history.append({
            'params': x.copy(),
            'cost': total_cost,
            'train_cost': train_stats['total_cost']['mean'],
            'val_cost': val_stats['total_cost']['mean'],
            'train_std': train_stats['total_cost']['std'],
            'val_std': val_stats['total_cost']['std']
        })
        
        print("\nTraining Statistics:")
        print(f"Mean cost: {train_stats['total_cost']['mean']:.2f} ± {train_stats['total_cost']['std']:.2f}")
        print(f"Min: {train_stats['total_cost']['min']:.2f}, Max: {train_stats['total_cost']['max']:.2f}")
        print(f"Median: {train_stats['total_cost']['median']:.2f}")
        
        print("\nValidation Statistics:")
        print(f"Mean cost: {val_stats['total_cost']['mean']:.2f} ± {val_stats['total_cost']['std']:.2f}")
        print(f"Min: {val_stats['total_cost']['min']:.2f}, Max: {val_stats['total_cost']['max']:.2f}")
        print(f"Median: {val_stats['total_cost']['median']:.2f}")
        
        print(f"\nCost difference: {train_val_diff:.2f}, Regularization: {regularization:.2f}")
        print(f"Stability penalty: {stability_penalty:.2f}")
        print(f"Final cost: {total_cost:.2f}")
        
        return total_cost

def split_routes(data_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split routes into train, validation, and test sets"""
    all_routes = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    np.random.shuffle(all_routes)
    
    n_routes = len(all_routes)
    n_train = int(train_ratio * n_routes)
    n_val = int(val_ratio * n_routes)
    
    train_routes = all_routes[:n_train]
    val_routes = all_routes[n_train:n_train + n_val]
    test_routes = all_routes[n_train + n_val:]
    
    return train_routes, val_routes, test_routes

def evaluate_pid_params(model_path, data_path, p, i, d, routes, num_routes=None, use_all_routes=False):
    """Evaluate PID controller with given parameters on specified routes"""
    model = TinyPhysicsModel(model_path, debug=False)
    controller = Controller()
    controller.p = p
    controller.i = i
    controller.d = d
    
    # Use all routes if specified, otherwise use a subset
    if use_all_routes:
        eval_routes = routes
    elif num_routes is not None:
        eval_routes = np.random.choice(routes, min(num_routes, len(routes)), replace=False)
    else:
        eval_routes = routes
    
    # Create progress bar for route evaluation
    pbar = tqdm(eval_routes, desc="Evaluating routes", leave=True)
    
    # Store individual route costs for statistics
    route_costs = []
    route_lataccel_costs = []
    route_jerk_costs = []
    
    for route_file in pbar:
        route_path = os.path.join(data_path, route_file)
        sim = TinyPhysicsSimulator(model, route_path, controller=controller, debug=False)
        costs = sim.rollout()
        
        # Store individual costs
        route_costs.append(costs['total_cost'])
        route_lataccel_costs.append(costs['lataccel_cost'])
        route_jerk_costs.append(costs['jerk_cost'])
        
        # Update progress bar with current statistics
        pbar.set_postfix({
            'avg_cost': f"{np.mean(route_costs):.2f}",
            'std_cost': f"{np.std(route_costs):.2f}",
            'min_cost': f"{np.min(route_costs):.2f}",
            'max_cost': f"{np.max(route_costs):.2f}"
        })
    
    # Calculate statistics
    stats = {
        'total_cost': {
            'mean': np.mean(route_costs),
            'std': np.std(route_costs),
            'min': np.min(route_costs),
            'max': np.max(route_costs),
            'median': np.median(route_costs)
        },
        'lataccel_cost': {
            'mean': np.mean(route_lataccel_costs),
            'std': np.std(route_lataccel_costs),
            'min': np.min(route_lataccel_costs),
            'max': np.max(route_lataccel_costs),
            'median': np.median(route_lataccel_costs)
        },
        'jerk_cost': {
            'mean': np.mean(route_jerk_costs),
            'std': np.std(route_jerk_costs),
            'min': np.min(route_jerk_costs),
            'max': np.max(route_jerk_costs),
            'median': np.median(route_jerk_costs)
        }
    }
    
    return stats

def optimize_pid(model_path, data_path, method='differential_evolution'):
    """Find optimal PID parameters using scipy.optimize"""
    # Split routes
    train_routes, val_routes, test_routes = split_routes(data_path)
    print(f"Training on {len(train_routes)} routes")
    print(f"Validating on {len(val_routes)} routes")
    print(f"Testing on {len(test_routes)} routes")
    
    # Define parameter bounds
    bounds = [
        (0.05, 1.0),    # P bounds
        (0.001, 0.2),   # I bounds
        (-0.3, 0.1)     # D bounds
    ]
    
    # Create optimizer instance
    optimizer = PIDOptimizer(model_path, data_path, train_routes, val_routes)
    
    # Initial guess (middle of bounds)
    x0 = np.array([0.5, 0.1, -0.1])
    
    print(f"\nStarting optimization using {method}...")
    
    if method == 'differential_evolution':
        result = differential_evolution(
            optimizer.objective,
            bounds=bounds,
            maxiter=50,
            popsize=15,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            workers=1  # Disable parallel processing for now
        )
    else:
        result = minimize(
            optimizer.objective,
            x0=x0,
            bounds=bounds,
            method=method,
            options={'maxiter': 100}
        )
    
    best_params = result.x
    best_cost = result.fun
    
    print("\nOptimization Results:")
    print(f"Best P: {best_params[0]:.3f}")
    print(f"Best I: {best_params[1]:.3f}")
    print(f"Best D: {best_params[2]:.3f}")
    print(f"Best Cost: {best_cost:.4f}")
    
    # Evaluate best parameters on test set
    print("\nEvaluating best parameters on test set...")
    test_stats = evaluate_pid_params(model_path, data_path, best_params[0], best_params[1], best_params[2], 
                                   test_routes, use_all_routes=True)
    
    print("\nFinal Test Results:")
    print(f"Mean Total Cost: {test_stats['total_cost']['mean']:.4f} ± {test_stats['total_cost']['std']:.4f}")
    print(f"Min: {test_stats['total_cost']['min']:.4f}, Max: {test_stats['total_cost']['max']:.4f}")
    print(f"Median: {test_stats['total_cost']['median']:.4f}")
    
    print(f"\nLateral Acceleration Cost: {test_stats['lataccel_cost']['mean']:.4f} ± {test_stats['lataccel_cost']['std']:.4f}")
    print(f"Jerk Cost: {test_stats['jerk_cost']['mean']:.4f} ± {test_stats['jerk_cost']['std']:.4f}")
    
    return best_params, optimizer.optimization_history

def plot_results(optimization_history):
    """Plot optimization results"""
    # Extract data from history
    params = np.array([h['params'] for h in optimization_history])
    costs = np.array([h['cost'] for h in optimization_history])
    train_costs = np.array([h['train_cost'] for h in optimization_history])
    val_costs = np.array([h['val_cost'] for h in optimization_history])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Cost vs Iteration
    ax1 = fig.add_subplot(221)
    ax1.plot(costs, 'b-', label='Total Cost')
    ax1.plot(train_costs, 'g--', label='Training Cost')
    ax1.plot(val_costs, 'r--', label='Validation Cost')
    ax1.plot(np.minimum.accumulate(costs), 'k:', label='Best Cost')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost')
    ax1.set_title('Cost vs Iteration')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Parameter Evolution
    ax2 = fig.add_subplot(222)
    ax2.plot(params[:, 0], 'b-', label='P')
    ax2.plot(params[:, 1], 'g-', label='I')
    ax2.plot(params[:, 2], 'r-', label='D')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Parameter Value')
    ax2.set_title('Parameter Evolution')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: P vs I scatter with cost as color
    ax3 = fig.add_subplot(223)
    scatter = ax3.scatter(params[:, 0], params[:, 1], c=costs, cmap='viridis')
    plt.colorbar(scatter, ax=ax3, label='Cost')
    ax3.set_xlabel('P')
    ax3.set_ylabel('I')
    ax3.set_title('P vs I (Cost as Color)')
    
    # Plot 4: P vs D scatter with cost as color
    ax4 = fig.add_subplot(224)
    scatter = ax4.scatter(params[:, 0], params[:, 2], c=costs, cmap='viridis')
    plt.colorbar(scatter, ax=ax4, label='Cost')
    ax4.set_xlabel('P')
    ax4.set_ylabel('D')
    ax4.set_title('P vs D (Cost as Color)')
    
    plt.tight_layout()
    plt.savefig('pid_optimization_results.png')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the physics model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data")
    parser.add_argument("--method", type=str, default='differential_evolution',
                      choices=['differential_evolution', 'Nelder-Mead', 'L-BFGS-B', 'SLSQP'],
                      help="Optimization method to use")
    args = parser.parse_args()
    
    print(f"Starting optimization using {args.method}...")
    best_params, optimization_history = optimize_pid(args.model_path, args.data_path, method=args.method)
    
    # Plot results
    plot_results(optimization_history) 