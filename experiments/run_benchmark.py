import subprocess
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="test")
    parser.add_argument("--config", default="configs/glm_test.yaml")
    args = parser.parse_args()
    # 简单调用 daily_runner
    subprocess.run(["python", "src/scheduler/daily_runner.py", "--date", "2025-06-01", "--num_tasks", "3"])