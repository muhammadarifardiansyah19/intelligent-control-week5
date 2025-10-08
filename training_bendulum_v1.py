import numpy as np
import gym
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import matplotlib.pyplot as plt

# --- Inisialisasi environment ---
env = gym.make("Pendulum-v1")  # simulasi kontrol sudut lengan robot sederhana

state_size = env.observation_space.shape[0]
action_size = 5  # discretize aksi (Pendulum aslinya continuous)
learning_rate = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
memory = deque(maxlen=5000)

# --- Model DQN ---
model = keras.Sequential([
    keras.layers.Dense(64, input_shape=(state_size,), activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(action_size, activation="linear")
])
model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

# --- Fungsi discretisasi aksi ---
def get_continuous_action(action_index):
    action_range = np.linspace(-2, 2, action_size)
    return np.array([action_range[action_index]])

# --- Pilih aksi ---
def select_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return np.random.choice(action_size)
    q_values = model.predict(state, verbose=0)
    return np.argmax(q_values[0])

# --- Training DQN ---
episodes = 1000
scores = []        # jumlah langkah tiap episode (score)
rewards = []       # total reward tiap episode

for episode in range(episodes): 
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for t in range(200):
        action_index = select_action(state, epsilon)
        action = get_continuous_action(action_index)

        next_state, reward, done, truncated, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        done = done or truncated

        # Normalisasi reward agar stabil
        reward /= 10.0

        memory.append((state, action_index, reward, next_state, done))
        state = next_state
        total_reward += reward

        if done:
            print(f"Episode {episode+1}/{episodes} | Score: {t+1} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.2f}")
            scores.append(t+1)         # simpan score (jumlah langkah)
            rewards.append(total_reward)
            break

    # Update model dari memory
    if len(memory) > batch_size:
        minibatch = random.sample(memory, batch_size)
        for s, a, r, s_next, done in minibatch:
            target = r
            if not done:
                target += gamma * np.amax(model.predict(s_next, verbose=0)[0])
            target_f = model.predict(s, verbose=0)
            target_f[0][a] = target
            model.fit(s, target_f, epochs=1, verbose=0)

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# --- Simpan model ---
model.save("dqn_robot_arm.keras")
print("Training selesai! Model disimpan sebagai dqn_robot_arm.keras")

# --- Plot hasil Reward dan Score ---
plt.figure(figsize=(12,5))
plt.plot(rewards, label="Total Reward per Episode", color='blue')
plt.plot(scores, label="Score (Steps) per Episode", color='green')
plt.xlabel("Episode")
plt.ylabel("Value") 
plt.title("Training DQN pada Pengendalian Lengan Robot (Pendulum-v1)") 
plt.legend() 
plt.grid(True)
plt.show()