import numpy as np
import matplotlib.pyplot as plt
from os import path

type_i_npy = r"C:\Users\10693\Desktop\斯图\Masterarbeit\cardillo\On the merits\workspace_samples_typeI_beta180_0_5.8_step_37_3case.npy"
basic_npy = r"C:\Users\10693\Desktop\斯图\Masterarbeit\cardillo\On the merits\workspace_samples_basic_0_5.8_step_200_3case.npy"

type_i_points = np.load(type_i_npy)
basic_points = np.load(basic_npy)

def make_cross_section(tip_points, chi_deg, slice_thickness_mm=15.0):
    chi_rad = np.deg2rad(chi_deg)
    x, y, z = tip_points[:,0], tip_points[:,1], tip_points[:,2]
    chi_coord = x*np.cos(chi_rad) + y*np.sin(chi_rad) #x 
    perp_coord = -x*np.sin(chi_rad) + y*np.cos(chi_rad) #y
    thickness_m = slice_thickness_mm * 1e-3
    mask = np.abs(perp_coord) < thickness_m/2 #half thickness
    return chi_coord[mask]*1000, z[mask]*1000

chi_deg = 124   # Type I
slice_thickness_mm = 15

chi_typei, z_typei = make_cross_section(type_i_points, chi_deg, slice_thickness_mm)
chi_basic, z_basic = make_cross_section(basic_points, chi_deg, slice_thickness_mm)

fig, ax = plt.subplots(figsize=(6, 6))#basic


ax.scatter(chi_typei, z_typei, s=2, alpha=0.3, c='purple', label='Type I (180 deg)')

ax.scatter(chi_basic, z_basic, s=8, alpha=1.0, c='black', label='basic workspace')

ax.set_xlabel('chi in mm')
ax.set_ylabel('z in mm')
ax.set_title(f'Type I vs basic, chi={chi_deg} deg')
ax.set_aspect('equal')
ax.legend()
plt.tight_layout()
plt.savefig('cross_section_typeI_vs_basic.png', dpi=300, bbox_inches='tight')
plt.show()