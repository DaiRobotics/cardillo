import numpy as np

tip_points = np.load(r"c:\Users\10693\Desktop\斯图\Masterarbeit\cardillo\workspace_samples_0_5.8_step_37.npy")
print(f"点数: {len(tip_points)}")
print(f"x: {tip_points[:,0].min()*1000:.2f} ~ {tip_points[:,0].max()*1000:.2f} mm")
print(f"y: {tip_points[:,1].min()*1000:.2f} ~ {tip_points[:,1].max()*1000:.2f} mm")
print(f"z: {tip_points[:,2].min()*1000:.2f} ~ {tip_points[:,2].max()*1000:.2f} mm")