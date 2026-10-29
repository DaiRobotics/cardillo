import time
import numpy as np
import catalyst
from catalyst_conduit import Node
from catalyst_runner import run_catalyst_step

# 初始化一次，连续执行多个时间步
init = Node()
init["catalyst/scripts/script_0"] = "catalyst_pipeline.py"
catalyst.initialize(init)

try:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    cells = np.array([[0, 1, 2, 3]], dtype=np.int32)

    velocity = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )

    for step in range(2000):
        # 每个时间步改变温度，方便观察 GUI 是否更新
        temperature = np.array([300.0, 310.0, 320.0, 330.0]) + 10.0 * np.sin(step * 0.4)

        run_catalyst_step(
            points=points,
            cells=cells,
            temperature=temperature,
            velocity=velocity,
            timestep=step,
            time=step * 0.5,
        )

        print(f"Sent timestep {step}, time={step * 0.5:.1f}")
        time.sleep(1)

finally:
    catalyst.finalize(Node())
    print("Simulation finished")
