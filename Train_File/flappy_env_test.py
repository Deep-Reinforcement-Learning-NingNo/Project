import gymnasium as gym
import flappy_bird_gymnasium
import time

# สร้าง environment
env = gym.make("FlappyBird-v0", render_mode="human")

# reset เพื่อเริ่มเกม
observation, info = env.reset()

# ลูปรันไปเรื่อย ๆ
while True:
    # เลือก action แบบสุ่ม (0 = ไม่กระโดด, 1 = กระโดด)
    action = env.action_space.sample()

    # ส่ง action เข้าไปใน env
    observation, reward, terminated, truncated, info = env.step(action)

    # แสดงค่าที่ได้
    print(f"Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

    # เช็คว่าเกมจบหรือยัง
    if terminated or truncated:
        print("Game Over!")
        time.sleep(1)
        observation, info = env.reset()

# อย่าลืมปิด environment ตอนเลิก
env.close()
