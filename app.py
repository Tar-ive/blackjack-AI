import streamlit as st
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from matplotlib.patches import Patch
from tqdm import tqdm

# Set page config
st.set_page_config(page_title="Blackjack Q-Learning Agent", layout="wide")

# Blackjack Agent class
class BlackjackAgent:
    def __init__(self, learning_rate, initial_epsilon, epsilon_decay, final_epsilon, discount_factor=0.95):
        self.q_values = defaultdict(lambda: np.zeros(2))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []

    def get_action(self, obs):
        if np.random.random() < self.epsilon:
            return np.random.randint(2)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(self, obs, action, reward, terminated, next_obs):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        self.q_values[obs][action] += self.lr * temporal_difference
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

# Function to create grids for visualization
def create_grids(agent, usable_ace=False):
    state_value = defaultdict(float)
    policy = defaultdict(int)
    for obs, action_values in agent.q_values.items():
        state_value[obs] = float(np.max(action_values))
        policy[obs] = int(np.argmax(action_values))

    player_count, dealer_count = np.meshgrid(np.arange(12, 22), np.arange(1, 11))

    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    value_grid = player_count, dealer_count, value

    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return value_grid, policy_grid

# Function to create plots
def create_plots(value_grid, policy_grid, title):
    player_count, dealer_count, value = value_grid
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(title, fontsize=16)

    # Plot state values
    im1 = ax1.imshow(value, cmap='viridis', origin='lower')
    ax1.set_title(f"State values: {title}")
    ax1.set_xlabel("Player sum")
    ax1.set_ylabel("Dealer showing")
    ax1.set_xticks(range(10))
    ax1.set_yticks(range(10))
    ax1.set_xticklabels(range(12, 22))
    ax1.set_yticklabels(["A"] + list(range(2, 11)))
    plt.colorbar(im1, ax=ax1)

    # Plot policy
    im2 = ax2.imshow(policy_grid, cmap='Accent_r', origin='lower')
    ax2.set_title(f"Policy: {title}")
    ax2.set_xlabel("Player sum")
    ax2.set_ylabel("Dealer showing")
    ax2.set_xticks(range(10))
    ax2.set_yticks(range(10))
    ax2.set_xticklabels(range(12, 22))
    ax2.set_yticklabels(["A"] + list(range(2, 11)))

    # Add legend
    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    return fig

# Streamlit app
def main():
    st.title("Blackjack Q-Learning Agent")

    # Sidebar for hyperparameters
    st.sidebar.header("Hyperparameters")
    learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
    n_episodes = st.sidebar.slider("Number of Episodes", 10000, 1000000, 100000, 10000)
    start_epsilon = st.sidebar.slider("Initial Epsilon", 0.1, 1.0, 1.0, 0.1)
    final_epsilon = st.sidebar.slider("Final Epsilon", 0.01, 0.5, 0.1, 0.01)

    if st.sidebar.button("Train Agent"):
        # Create environment and agent
        env = gym.make("Blackjack-v1", sab=True)
        epsilon_decay = start_epsilon / (n_episodes / 2)
        agent = BlackjackAgent(learning_rate, start_epsilon, epsilon_decay, final_epsilon)

        # Training loop
        env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
        progress_bar = st.progress(0)
        status_text = st.empty()

        returns = []  # Store returns for each episode

        for episode in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_return = 0

            while not done:
                action = agent.get_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                agent.update(obs, action, reward, terminated, next_obs)
                done = terminated or truncated
                obs = next_obs
                episode_return += reward

            agent.decay_epsilon()
            returns.append(episode_return)

            # Update progress bar and status text
            progress = (episode + 1) / n_episodes
            progress_bar.progress(progress)
            status_text.text(f"Training progress: {progress:.2%}")

        status_text.text("Training complete!")

        # Visualize training results
        st.header("Training Results")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(1, n_episodes + 1), returns)
        ax.set_title("Returns over Training")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return")
        st.pyplot(fig)

        # Visualize moving average of returns
        window_size = min(1000, n_episodes // 10)  # Adjust window size based on number of episodes
        moving_avg = np.convolve(returns, np.ones(window_size), 'valid') / window_size
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(window_size, len(returns) + 1), moving_avg)
        ax.set_title(f"Moving Average of Returns (Window Size: {window_size})")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Average Return")
        st.pyplot(fig)

        # Visualize policy and value function
        st.header("Policy and Value Function")
        usable_ace = st.checkbox("Usable Ace")
        value_grid, policy_grid = create_grids(agent, usable_ace)
        fig = create_plots(value_grid, policy_grid, f"{'With' if usable_ace else 'Without'} usable ace")
        st.pyplot(fig)

if __name__ == "__main__":
    main()

