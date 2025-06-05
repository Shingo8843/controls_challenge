import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers.ppo_controller import Controller
import argparse
from itertools import cycle


def train_controller(model_path, data_path, num_episodes=1000, save_interval=100, checkpoint_path=None, resume=False, 
                    routes_per_episode=5, val_routes_per_check=100):
    model = TinyPhysicsModel(model_path, debug=False)
    all_routes = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    np.random.shuffle(all_routes)

    # Split dataset
    n = len(all_routes)
    train_routes = all_routes[:int(0.7 * n)]
    val_routes = all_routes[int(0.7 * n):int(0.85 * n)]
    test_routes = all_routes[int(0.85 * n):]

    print(f"Train: {len(train_routes)}, Val: {len(val_routes)}, Test: {len(test_routes)}")
    print(f"Using {routes_per_episode} routes per episode")
    print(f"Using {val_routes_per_check} routes for validation checks")

    os.makedirs('models/rl', exist_ok=True)

    # Initialize controller
    controller = Controller(load_model=resume, model_path=checkpoint_path)
    start_episode = controller.episode_count if resume else 0

    # Create a cycle iterator for training routes to ensure all routes are used
    train_route_cycle = cycle(train_routes)
    
    train_costs = []
    val_costs = []
    best_val_cost = float('inf')
    no_improve_count = 0
    patience = 50

    # Create progress bar for episodes
    episode_pbar = tqdm(range(start_episode, num_episodes), desc="Episodes", position=0)
    
    for episode in episode_pbar:
        controller.reset()
        episode_cost = 0
        route_costs = []
        
        # Create progress bar for routes within episode
        route_pbar = tqdm(range(routes_per_episode), desc=f"Episode {episode+1} Routes", position=1, leave=False)
        
        # Use multiple routes per episode
        for route_idx in route_pbar:
            route_file = next(train_route_cycle)
            sim = TinyPhysicsSimulator(model, os.path.join(data_path, route_file), controller=controller, debug=False)
            costs = sim.rollout()
            route_cost = costs["total_cost"]
            route_costs.append(route_cost)
            episode_cost += route_cost
            controller.train_on_episode()
            
            # Update route progress bar
            route_pbar.set_postfix({
                'route_cost': f'{route_cost:.4f}',
                'route': route_file
            })
        
        # Average cost across routes in this episode
        avg_episode_cost = episode_cost / routes_per_episode
        train_costs.append(avg_episode_cost)
        controller.episode_count = episode + 1

        # Update episode progress bar
        episode_pbar.set_postfix({
            'avg_cost': f'{avg_episode_cost:.4f}',
            'best_val': f'{best_val_cost:.4f}',
            'routes_used': f'{episode * routes_per_episode}/{len(train_routes)}'
        })

        # Validation every 20 episodes
        if (episode + 1) % 20 == 0:
            val_cost = 0
            # Randomly sample validation routes
            val_subset = np.random.choice(val_routes, min(val_routes_per_check, len(val_routes)), replace=False)
            val_pbar = tqdm(val_subset, desc="Validation", position=1, leave=False)
            for val_file in val_pbar:
                controller.reset()
                sim = TinyPhysicsSimulator(model, os.path.join(data_path, val_file), controller=controller, debug=False)
                val_cost += sim.rollout()["total_cost"]
                val_pbar.set_postfix({'current_cost': f'{val_cost/len(val_subset):.4f}'})
            val_cost /= len(val_subset)
            val_costs.append(val_cost)

            print(f"\n[Episode {episode+1}] Val Cost: {val_cost:.4f}, Best: {best_val_cost:.4f}")
            if val_cost < best_val_cost:
                best_val_cost = val_cost
                no_improve_count = 0
                controller.save_checkpoint('models/rl/ppo_best.pt')
                print("✅ Saved new best model.")
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print("⏹ Early stopping triggered.")
                    break

        # Save model and plot
        if (episode + 1) % save_interval == 0:
            controller.save_checkpoint(f'models/rl/ppo_ep{episode+1}.pt')
            plot_progress(train_costs, val_costs)

    # Final save
    controller.save_checkpoint('models/rl/ppo_final.pt')

    # Evaluate on test
    print("\nEvaluating on test set...")
    test_cost = evaluate_controller(controller, model, test_routes, data_path)
    return controller, train_costs, val_costs, test_cost


def evaluate_controller(controller, model, routes, data_path, plot=True):
    controller.set_training(False)
    total_cost = 0
    for idx, route_file in enumerate(routes):
        controller.reset()
        sim = TinyPhysicsSimulator(model, os.path.join(data_path, route_file), controller=controller, debug=False)
        costs = sim.rollout()
        total_cost += costs["total_cost"]

        if plot:
            plt.figure(figsize=(10, 4))
            plt.plot(sim.target_lataccel_history, 'r--', label="Target")
            plt.plot(sim.current_lataccel_history, 'b-', label="Actual")
            plt.title(f"Route {idx} - Lateral Acceleration")
            plt.xlabel("Timestep")
            plt.ylabel("Lat Accel")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"results/route_{idx}_plot.png")
            plt.close()

    avg_cost = total_cost / len(routes)
    print(f"Test Cost: {avg_cost:.4f}")
    return avg_cost


def plot_progress(train_costs, val_costs):
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(train_costs, label="Train Cost", alpha=0.4)
    if len(train_costs) > 10:
        smoothed = np.convolve(train_costs, np.ones(10)/10, mode='valid')
        plt.plot(range(10 - 1, len(train_costs)), smoothed, 'r--', label="Train Moving Avg")
    if val_costs:
        plt.plot(range(0, len(train_costs), 20), val_costs, 'g-', label="Validation Cost")
    plt.xlabel("Episode")
    plt.ylabel("Cost")
    plt.title("PPO Training Progress")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/training_curve.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to tinyphysics.onnx")
    parser.add_argument("--data_path", type=str, required=True, help="Path to /data folder")
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--routes_per_episode", type=int, default=5, help="Number of routes to use per training episode")
    parser.add_argument("--val_routes_per_check", type=int, default=100, help="Number of validation routes to use per validation check")
    args = parser.parse_args()

    print("🚗 Starting PPO Training")
    train_controller(
        model_path=args.model_path,
        data_path=args.data_path,
        num_episodes=args.num_episodes,
        save_interval=args.save_interval,
        checkpoint_path=args.checkpoint_path,
        resume=args.resume,
        routes_per_episode=args.routes_per_episode,
        val_routes_per_check=args.val_routes_per_check
    )
