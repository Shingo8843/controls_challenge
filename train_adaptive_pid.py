from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers.my_pid import Controller
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

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

def evaluate_adaptive_params(model_path, data_path, p_lr, i_lr, d_lr, p_bounds, i_bounds, d_bounds, routes, num_routes=None, use_all_routes=False):
    """Evaluate adaptive PID controller with given parameters on specified routes"""
    model = TinyPhysicsModel(model_path, debug=False)
    controller = Controller()
    
    # Set adaptive parameters
    controller.p_lr = p_lr
    controller.i_lr = i_lr
    controller.d_lr = d_lr
    controller.p_bounds = p_bounds
    controller.i_bounds = i_bounds
    controller.d_bounds = d_bounds
    
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

def optimize_adaptive_pid(model_path, data_path, method='differential_evolution', num_workers=1):
    """Find optimal adaptive PID parameters using scipy.optimize"""
    # Split routes
    train_routes, val_routes, test_routes = split_routes(data_path)
    print(f"Training on {len(train_routes)} routes")
    print(f"Validating on {len(val_routes)} routes")
    print(f"Testing on {len(test_routes)} routes")
    
    # Define parameter bounds for optimization - tighter bounds around good values
    bounds = [
        (0.00001, 0.0005),  # p_lr bounds - reduced upper bound
        (0.000001, 0.00005),  # i_lr bounds - reduced upper bound
        (0.000001, 0.00005),  # d_lr bounds - reduced upper bound
        (0.2, 0.3),  # p_bounds[0] - tighter around optimal P
        (0.25, 0.35),  # p_bounds[1] - tighter around optimal P
        (0.08, 0.12),  # i_bounds[0] - tighter around optimal I
        (0.09, 0.13),  # i_bounds[1] - tighter around optimal I
        (0.005, 0.02),  # d_bounds[0] - tighter around optimal D
        (0.01, 0.03)   # d_bounds[1] - tighter around optimal D
    ]
    
    # Store optimization history
    optimization_history = []
    best_cost = float('inf')
    no_improvement_count = 0
    max_no_improvement = 20  # Increased from 10 to 20
    
    def objective(x):
        """Objective function to minimize"""
        nonlocal best_cost, no_improvement_count
        
        # Unpack parameters
        p_lr, i_lr, d_lr = x[0:3]
        p_bounds = (x[3], x[4])
        i_bounds = (x[5], x[6])
        d_bounds = (x[7], x[8])
        
        print(f"\nTrying parameters:")
        print(f"Learning rates: P={p_lr:.6f}, I={i_lr:.6f}, D={d_lr:.6f}")
        print(f"P bounds: {p_bounds}")
        print(f"I bounds: {i_bounds}")
        print(f"D bounds: {d_bounds}")
        
        # Evaluate on both training and validation sets
        train_stats = evaluate_adaptive_params(model_path, data_path, p_lr, i_lr, d_lr, 
                                             p_bounds, i_bounds, d_bounds,
                                             train_routes, num_routes=100)  # Increased from 50
        val_stats = evaluate_adaptive_params(model_path, data_path, p_lr, i_lr, d_lr,
                                           p_bounds, i_bounds, d_bounds,
                                           val_routes, num_routes=100)  # Increased from 50
        
        # Calculate combined cost with regularization
        train_val_diff = abs(train_stats['total_cost']['mean'] - val_stats['total_cost']['mean'])
        regularization = 0.5 * train_val_diff  # Increased from 0.3
        
        # Add stability penalty
        stability_penalty = 0.3 * (train_stats['total_cost']['std'] + val_stats['total_cost']['std'])  # Increased from 0.2
        
        # Add median penalty to reduce impact of outliers
        median_penalty = 0.2 * (train_stats['total_cost']['median'] + val_stats['total_cost']['median'])
        
        total_cost = val_stats['total_cost']['mean'] + regularization + stability_penalty + median_penalty
        
        # Store optimization history
        optimization_history.append({
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
        print(f"Median penalty: {median_penalty:.2f}")
        print(f"Final cost: {total_cost:.2f}")
        
        # Check for improvement with smaller threshold
        if total_cost < best_cost - 0.05:  # Reduced from 0.1
            best_cost = total_cost
            no_improvement_count = 0
            print("\nNew best cost found!")
        else:
            no_improvement_count += 1
            print(f"\nNo improvement for {no_improvement_count} iterations")
        
        return total_cost
    
    # Initial guess (middle of bounds)
    x0 = np.array([0.0001, 0.00001, 0.00001, 0.25, 0.3, 0.1, 0.11, 0.01, 0.015])
    
    print(f"\nStarting optimization using {method}...")
    
    if method == 'differential_evolution':
        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=50,  # Increased from 30
            popsize=15,  # Increased from 10
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            workers=1,
            callback=lambda xk, convergence=0: print(f"\nIteration {len(optimization_history)} complete. Convergence: {convergence}")
        )
    else:
        result = minimize(
            objective,
            x0=x0,
            bounds=bounds,
            method=method,
            options={'maxiter': 100},  # Increased from 50
            callback=lambda xk: print(f"\nIteration {len(optimization_history)} complete")
        )
    
    best_params = result.x
    best_cost = result.fun
    
    print("\nOptimization Results:")
    print(f"Best P learning rate: {best_params[0]:.6f}")
    print(f"Best I learning rate: {best_params[1]:.6f}")
    print(f"Best D learning rate: {best_params[2]:.6f}")
    print(f"Best P bounds: ({best_params[3]:.3f}, {best_params[4]:.3f})")
    print(f"Best I bounds: ({best_params[5]:.3f}, {best_params[6]:.3f})")
    print(f"Best D bounds: ({best_params[7]:.3f}, {best_params[8]:.3f})")
    print(f"Best Cost: {best_cost:.4f}")
    
    # Evaluate best parameters on test set
    print("\nEvaluating best parameters on test set...")
    test_stats = evaluate_adaptive_params(model_path, data_path, 
                                        best_params[0], best_params[1], best_params[2],
                                        (best_params[3], best_params[4]),
                                        (best_params[5], best_params[6]),
                                        (best_params[7], best_params[8]),
                                        test_routes, use_all_routes=True)
    
    print("\nFinal Test Results:")
    print(f"Mean Total Cost: {test_stats['total_cost']['mean']:.4f} ± {test_stats['total_cost']['std']:.4f}")
    print(f"Min: {test_stats['total_cost']['min']:.4f}, Max: {test_stats['total_cost']['max']:.4f}")
    print(f"Median: {test_stats['total_cost']['median']:.4f}")
    
    print(f"\nLateral Acceleration Cost: {test_stats['lataccel_cost']['mean']:.4f} ± {test_stats['lataccel_cost']['std']:.4f}")
    print(f"Jerk Cost: {test_stats['jerk_cost']['mean']:.4f} ± {test_stats['jerk_cost']['std']:.4f}")
    
    return best_params, optimization_history

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
    
    # Plot 2: Learning Rate Evolution
    ax2 = fig.add_subplot(222)
    ax2.plot(params[:, 0], 'b-', label='P learning rate')
    ax2.plot(params[:, 1], 'g-', label='I learning rate')
    ax2.plot(params[:, 2], 'r-', label='D learning rate')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Evolution')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: P bounds evolution
    ax3 = fig.add_subplot(223)
    ax3.plot(params[:, 3], 'b-', label='P lower bound')
    ax3.plot(params[:, 4], 'r-', label='P upper bound')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('P Bound Value')
    ax3.set_title('P Bounds Evolution')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: I bounds evolution
    ax4 = fig.add_subplot(224)
    ax4.plot(params[:, 5], 'b-', label='I lower bound')
    ax4.plot(params[:, 6], 'r-', label='I upper bound')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('I Bound Value')
    ax4.set_title('I Bounds Evolution')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('adaptive_pid_optimization_results.png')
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
    best_params, optimization_history = optimize_adaptive_pid(args.model_path, args.data_path, method=args.method)
    
    # Plot results
    plot_results(optimization_history)