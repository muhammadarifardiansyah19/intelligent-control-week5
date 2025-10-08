import numpy as np
import gymnasium as gym   # ✅ gunakan Gymnasium (pengganti Gym)
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import os

# === 1️⃣ Cek dan load model ===
model_path = "dqn_robot_arm.keras"

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"❌ File model '{model_path}' tidak ditemukan.\n"
        "Pastikan model hasil training disimpan di folder yang sama dengan file ini."
    )

model = keras.models.load_model(model_path)
print(f"✅ Model berhasil dimuat dari file: {model_path}\n")

# === 2️⃣ Inisialisasi environment ===
env = gym.make("Pendulum-v1", render_mode="human")
state_size = env.observation_space.shape[0]
action_size = 5  # harus sama seperti saat training

# === 3️⃣ Fungsi bantu ===
def get_continuous_action(action_index):
    """Konversi indeks aksi ke nilai torsi kontinu (-2 s.d. 2)."""
    action_range = np.linspace(-2, 2, action_size)
    return np.array([action_range[action_index]])

def select_action(state):
    """Pilih aksi terbaik berdasarkan model (tanpa epsilon)."""
    q_values = model.predict(state, verbose=0)
    return np.argmax(q_values[0])

# === 4️⃣ Testing ===
test_episodes = 100
hasil_test = []

print("=== HASIL TESTING MODEL DQN PADA PENDULUM ===")

for episode in range(test_episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    steps = 0

    for t in range(200):  # maksimal 200 langkah per episode
        action_index = select_action(state)
        action = get_continuous_action(action_index)
        next_state, reward, terminated, truncated, _ = env.step(action)

        state = np.reshape(next_state, [1, state_size])
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    hasil_test.append({
        "Episode": f"{episode + 1}/{test_episodes}",
        "Score (Steps)": steps,
        "Total Reward": total_reward
    })

    # ✅ Format tampilan ringkas: Episode X/100, Score: XX, Reward: XXX.XX
    print(f"Episode {episode+1}/{test_episodes}, Score: {steps}, Reward: {total_reward:.2f}")

env.close()

# === 5️⃣ Simpan ke Excel ===
df_hasil = pd.DataFrame(hasil_test)
excel_path = "hasil_testing_dqn.xlsx"
df_hasil.to_excel(excel_path, index=False)
print(f"\n✅ Hasil testing berhasil disimpan ke: {excel_path}")

# === 6️⃣ Visualisasi hasil ===
plt.figure(figsize=(8,4))
plt.plot(range(1, test_episodes + 1), df_hasil["Total Reward"], marker='o', color='blue', label="Reward")
plt.plot(range(1, test_episodes + 1), df_hasil["Score (Steps)"], marker='s', color='green', label="Score (Steps)")
plt.title("Hasil Testing Model DQN pada Pendulum-v1")
plt.xlabel("Episode")
plt.ylabel("Nilai")
plt.legend()
plt.grid(True)
plt.show()

print("Testing selesai ✅")
