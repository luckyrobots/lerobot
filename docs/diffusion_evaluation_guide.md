# Diffusion Policy Evaluation & Real-Time Visualization Guide

This comprehensive guide shows how to evaluate trained diffusion policies using the LeRobot framework with real-time visualization and inference capabilities.

## Overview

Your trained diffusion policy is located at:
```
outputs/train/2025-06-12/17-45-21_diffusion/checkpoints/020000/pretrained_model/
```

**Model Configuration:**
- Type: Diffusion Policy
- Horizon: 16 time steps
- Action Steps: 8 steps executed per policy call
- Vision Backbone: ResNet18
- Input: State (6D) + 2 Cameras (480×640×3)
- Output: 6D action vector

## Quick Start

```python
import torch
import numpy as np
import gymnasium as gym
import gym_pusht
from pathlib import Path
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

# Load your trained diffusion policy
checkpoint_path = "outputs/train/2025-06-12/17-45-21_diffusion/checkpoints/020000/pretrained_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

policy = DiffusionPolicy.from_pretrained(checkpoint_path)
policy.to(device)
policy.eval()

print(f"✅ Policy loaded successfully on {device}")
```

## 1. Basic Single Episode Evaluation

```python
def evaluate_single_episode(policy, device="cuda", seed=42):
    """Evaluate a single episode"""

    # Create environment
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        max_episode_steps=300,
        render_mode="rgb_array"
    )

    # Reset policy and environment
    policy.reset()
    observation, info = env.reset(seed=seed)

    total_reward = 0
    step_count = 0
    frames = []

    print(f"Starting episode evaluation...")

    done = False
    while not done:
        # Preprocess observation for policy
        processed_obs = {
            "observation.state": torch.from_numpy(observation["agent_pos"]).float().unsqueeze(0).to(device),
            "observation.image": torch.from_numpy(observation["pixels"]).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        }

        # Get action from diffusion policy
        with torch.inference_mode():
            action = policy.select_action(processed_obs)

        # Convert to numpy for environment
        action_np = action.squeeze(0).cpu().numpy()

        # Step environment
        observation, reward, terminated, truncated, info = env.step(action_np)

        # Track progress
        total_reward += reward
        step_count += 1
        frames.append(env.render())

        done = terminated or truncated

        if step_count % 50 == 0:
            print(f"  Step {step_count}: Reward={reward:.3f}, Total={total_reward:.3f}")

    env.close()

    print(f"Episode finished: {'SUCCESS' if terminated else 'FAILED'}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Steps: {step_count}")

    return {
        "success": terminated,
        "total_reward": total_reward,
        "steps": step_count,
        "frames": frames
    }

# Run evaluation
result = evaluate_single_episode(policy, device=device)
```

## 2. Real-Time OpenCV Visualization

```python
import cv2
import time

def evaluate_with_realtime_opencv(policy, device="cuda", fps_limit=10):
    """Real-time evaluation with OpenCV visualization"""

    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        max_episode_steps=300,
        render_mode="rgb_array"
    )

    policy.reset()
    observation, _ = env.reset()

    # Create window
    cv2.namedWindow("Diffusion Policy Evaluation", cv2.WINDOW_AUTOSIZE)

    step_count = 0
    total_reward = 0

    print("Starting real-time evaluation (Press 'q' to quit)")

    try:
        done = False
        while not done:
            frame_start = time.time()

            # Preprocess observation
            processed_obs = {
                "observation.state": torch.from_numpy(observation["agent_pos"]).float().unsqueeze(0).to(device),
                "observation.image": torch.from_numpy(observation["pixels"]).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            }

            # Get policy action
            with torch.inference_mode():
                action = policy.select_action(processed_obs)

            action_np = action.squeeze(0).cpu().numpy()

            # Step environment
            observation, reward, terminated, truncated, info = env.step(action_np)
            total_reward += reward
            step_count += 1
            done = terminated or truncated

            # Render and display
            frame = env.render()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Add overlay information
            overlay_info = [
                f"Step: {step_count}",
                f"Reward: {reward:.3f}",
                f"Total Reward: {total_reward:.2f}",
                f"Action: [{action_np[0]:.2f}, {action_np[1]:.2f}]",
                f"Status: {'SUCCESS' if terminated else 'FAILED' if truncated else 'Running'}",
                "Press 'q' to quit"
            ]

            for i, text in enumerate(overlay_info):
                color = (0, 255, 0) if i < 4 else (0, 255, 255) if i == 4 else (255, 255, 255)
                cv2.putText(frame_bgr, text, (10, 30 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Diffusion Policy Evaluation", frame_bgr)

            # FPS control
            elapsed = time.time() - frame_start
            wait_time = max(1, int(1000/fps_limit - elapsed*1000))

            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        env.close()

    print(f"\nEvaluation complete:")
    print(f"  Final reward: {total_reward:.2f}")
    print(f"  Steps: {step_count}")
    print(f"  Success: {terminated if done else False}")

# Run real-time evaluation
evaluate_with_realtime_opencv(policy, device=device, fps_limit=10)
```

## 3. Matplotlib Real-Time Dashboard

```python
import matplotlib.pyplot as plt
from collections import deque

def evaluate_with_matplotlib_dashboard(policy, device="cuda", max_steps=200):
    """Create a real-time dashboard with matplotlib"""

    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        max_episode_steps=max_steps,
        render_mode="rgb_array"
    )

    policy.reset()
    observation, _ = env.reset()

    # Setup matplotlib
    plt.ion()
    fig = plt.figure(figsize=(15, 10))

    # Create subplots
    ax1 = plt.subplot(2, 3, 1)  # Environment view
    ax2 = plt.subplot(2, 3, 2)  # Reward plot
    ax3 = plt.subplot(2, 3, 3)  # Action plot
    ax4 = plt.subplot(2, 3, 4)  # Action space
    ax5 = plt.subplot(2, 3, 5)  # State plot
    ax6 = plt.subplot(2, 3, 6)  # Performance metrics

    # Data storage
    max_history = 100
    rewards = deque(maxlen=max_history)
    actions_x = deque(maxlen=max_history)
    actions_y = deque(maxlen=max_history)
    states_x = deque(maxlen=max_history)
    states_y = deque(maxlen=max_history)
    steps = deque(maxlen=max_history)

    step_count = 0
    total_reward = 0

    print("Starting matplotlib dashboard evaluation...")

    done = False
    while not done and step_count < max_steps:
        # Get action
        processed_obs = {
            "observation.state": torch.from_numpy(observation["agent_pos"]).float().unsqueeze(0).to(device),
            "observation.image": torch.from_numpy(observation["pixels"]).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        }

        with torch.inference_mode():
            action = policy.select_action(processed_obs)

        action_np = action.squeeze(0).cpu().numpy()

        # Step environment
        observation, reward, terminated, truncated, info = env.step(action_np)

        # Update data
        rewards.append(reward)
        actions_x.append(action_np[0])
        actions_y.append(action_np[1])
        states_x.append(observation["agent_pos"][0])
        states_y.append(observation["agent_pos"][1])
        steps.append(step_count)
        total_reward += reward
        step_count += 1
        done = terminated or truncated

        # Update plots every 5 steps
        if step_count % 5 == 0:
            # Clear all axes
            for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
                ax.clear()

            # Environment view
            frame = env.render()
            ax1.imshow(frame)
            ax1.set_title(f"Environment - Step {step_count}")
            ax1.axis('off')

            # Reward history
            ax2.plot(list(steps), list(rewards))
            ax2.set_title(f"Reward (Total: {total_reward:.2f})")
            ax2.set_xlabel("Step")
            ax2.set_ylabel("Reward")
            ax2.grid(True)

            # Action history
            ax3.plot(list(steps), list(actions_x), label="Action X", color='blue')
            ax3.plot(list(steps), list(actions_y), label="Action Y", color='red')
            ax3.set_title("Action Commands")
            ax3.set_xlabel("Step")
            ax3.set_ylabel("Action Value")
            ax3.legend()
            ax3.grid(True)

            # Action space
            ax4.scatter(list(actions_x), list(actions_y), alpha=0.6, c=list(steps), cmap='viridis')
            ax4.set_title("Action Space")
            ax4.set_xlabel("Action X")
            ax4.set_ylabel("Action Y")
            ax4.grid(True)

            # State trajectory
            ax5.plot(list(states_x), list(states_y), alpha=0.7, color='green')
            ax5.scatter(list(states_x), list(states_y), alpha=0.6, c=list(steps), cmap='plasma', s=10)
            ax5.set_title("Agent Trajectory")
            ax5.set_xlabel("Position X")
            ax5.set_ylabel("Position Y")
            ax5.grid(True)

            # Performance metrics
            ax6.text(0.1, 0.8, f"Step: {step_count}", fontsize=12, transform=ax6.transAxes)
            ax6.text(0.1, 0.7, f"Total Reward: {total_reward:.2f}", fontsize=12, transform=ax6.transAxes)
            ax6.text(0.1, 0.6, f"Current Reward: {reward:.3f}", fontsize=12, transform=ax6.transAxes)
            ax6.text(0.1, 0.5, f"Action: [{action_np[0]:.2f}, {action_np[1]:.2f}]", fontsize=12, transform=ax6.transAxes)
            ax6.text(0.1, 0.4, f"Status: {'SUCCESS' if terminated else 'FAILED' if truncated else 'Running'}",
                    fontsize=12, transform=ax6.transAxes)
            ax6.set_xlim(0, 1)
            ax6.set_ylim(0, 1)
            ax6.set_title("Metrics")

            plt.tight_layout()
            plt.pause(0.01)

    plt.ioff()
    plt.show()
    env.close()

    print(f"\nDashboard evaluation complete:")
    print(f"  Success: {terminated if done else False}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Steps: {step_count}")

# Run dashboard evaluation
evaluate_with_matplotlib_dashboard(policy, device=device, max_steps=200)
```

## 4. Batch Evaluation with Statistics

```python
def evaluate_multiple_episodes(policy, n_episodes=10, device="cuda"):
    """Comprehensive batch evaluation"""

    print(f"Starting batch evaluation with {n_episodes} episodes...")

    results = []
    all_rewards = []
    all_successes = []
    all_steps = []

    for episode in range(n_episodes):
        print(f"\n--- Episode {episode + 1}/{n_episodes} ---")

        # Run single episode
        result = evaluate_single_episode(
            policy,
            device=device,
            seed=42 + episode
        )

        results.append(result)
        all_rewards.append(result["total_reward"])
        all_successes.append(result["success"])
        all_steps.append(result["steps"])

    # Compute statistics
    success_rate = np.mean(all_successes) * 100
    avg_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    avg_steps = np.mean(all_steps)

    print(f"\n{'='*50}")
    print(f"BATCH EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Episodes Evaluated: {n_episodes}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Reward Range: [{min(all_rewards):.2f}, {max(all_rewards):.2f}]")
    print(f"Average Episode Length: {avg_steps:.1f} steps")
    print(f"{'='*50}")

    return {
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "avg_steps": avg_steps,
        "all_rewards": all_rewards,
        "all_successes": all_successes,
        "results": results
    }

# Run batch evaluation
batch_results = evaluate_multiple_episodes(policy, n_episodes=5)
```

## 5. Performance Analysis

```python
def analyze_policy_performance(policy, device="cuda", n_measurements=50):
    """Analyze inference speed and memory usage"""

    print("Analyzing policy performance...")

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos")
    policy.reset()
    observation, _ = env.reset()

    # Prepare observation
    processed_obs = {
        "observation.state": torch.from_numpy(observation["agent_pos"]).float().unsqueeze(0).to(device),
        "observation.image": torch.from_numpy(observation["pixels"]).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    }

    # Warmup
    for _ in range(10):
        with torch.inference_mode():
            _ = policy.select_action(processed_obs)

    # Measure inference times
    inference_times = []

    for i in range(n_measurements):
        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()

        with torch.inference_mode():
            action = policy.select_action(processed_obs)

        if device == "cuda":
            torch.cuda.synchronize()

        inference_time = time.time() - start_time
        inference_times.append(inference_time * 1000)  # Convert to ms

    env.close()

    # Compute statistics
    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    avg_fps = 1000 / avg_time

    print(f"\n{'='*40}")
    print(f"PERFORMANCE ANALYSIS")
    print(f"{'='*40}")
    print(f"Measurements: {n_measurements}")
    print(f"Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"Min/Max inference time: {min_time:.2f} / {max_time:.2f} ms")
    print(f"Average FPS: {avg_fps:.1f}")

    if device == "cuda":
        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_cached = torch.cuda.memory_reserved() / 1024**2  # MB
        print(f"GPU Memory Allocated: {memory_allocated:.1f} MB")
        print(f"GPU Memory Cached: {memory_cached:.1f} MB")

    print(f"{'='*40}")

    return {
        "avg_inference_time_ms": avg_time,
        "std_inference_time_ms": std_time,
        "avg_fps": avg_fps,
        "measurements": inference_times
    }

# Run performance analysis
perf_results = analyze_policy_performance(policy, device=device)
```

## 6. Using LeRobot's Built-in Evaluation Script

You can also use the official LeRobot evaluation script:

```bash
python lerobot/scripts/eval.py \
    --policy.path=outputs/train/2025-06-12/17-45-21_diffusion/checkpoints/020000/pretrained_model \
    --env.type=pusht \
    --eval.n_episodes=10 \
    --eval.batch_size=1 \
    --device=cuda \
    --output_dir=outputs/official_eval
```

## 7. Complete Evaluation Script

Save this as `evaluate_diffusion_policy.py`:

```python
#!/usr/bin/env python3
"""
Complete Diffusion Policy Evaluation Script
"""

import torch
import numpy as np
import gymnasium as gym
import gym_pusht
import argparse
import time
from pathlib import Path
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

def main():
    parser = argparse.ArgumentParser(description="Evaluate Diffusion Policy")
    parser.add_argument("--checkpoint", type=str,
                       default="outputs/train/2025-06-12/17-45-21_diffusion/checkpoints/020000/pretrained_model",
                       help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--mode", type=str, default="basic",
                       choices=["basic", "opencv", "matplotlib", "batch", "performance"],
                       help="Evaluation mode")

    args = parser.parse_args()

    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"🤖 Loading Diffusion Policy from {args.checkpoint}")
    print(f"🔧 Device: {device}")

    # Load policy
    try:
        policy = DiffusionPolicy.from_pretrained(args.checkpoint)
        policy.to(device)
        policy.eval()
        print("✅ Policy loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load policy: {e}")
        return

    # Run evaluation based on mode
    if args.mode == "basic":
        print("Running basic evaluation...")
        result = evaluate_single_episode(policy, device=device)

    elif args.mode == "opencv":
        print("Running real-time OpenCV evaluation...")
        evaluate_with_realtime_opencv(policy, device=device)

    elif args.mode == "matplotlib":
        print("Running matplotlib dashboard evaluation...")
        evaluate_with_matplotlib_dashboard(policy, device=device)

    elif args.mode == "batch":
        print(f"Running batch evaluation with {args.episodes} episodes...")
        batch_results = evaluate_multiple_episodes(
            policy,
            n_episodes=args.episodes,
            device=device
        )

    elif args.mode == "performance":
        print("Running performance analysis...")
        perf_results = analyze_policy_performance(policy, device=device)

    print("🎉 Evaluation complete!")

if __name__ == "__main__":
    main()
```

## Usage Examples

1. **Basic evaluation**:
```bash
python evaluate_diffusion_policy.py --mode basic
```

2. **Real-time visualization**:
```bash
python evaluate_diffusion_policy.py --mode opencv
```

3. **Batch evaluation**:
```bash
python evaluate_diffusion_policy.py --mode batch --episodes 10
```

4. **Performance analysis**:
```bash
python evaluate_diffusion_policy.py --mode performance
```

Your trained diffusion policy is ready for comprehensive evaluation! 🚀

<!-- Change Log:
- Created comprehensive evaluation guide for diffusion policies
- Added multiple real-time visualization methods
- Included performance monitoring
- Provided complete evaluation script
- Tailored for specific model configuration -->
