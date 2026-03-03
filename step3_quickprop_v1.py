import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.tri as tri
from pathlib import Path

# --- 0. Directory Setup ---
# Automatically gets the directory where this script is located
base_dir = Path(__file__).parent.absolute() if '__file__' in locals() else Path.cwd()

data_dir = base_dir / "Data_Space"
img_dir = base_dir / "Quick_Property_Scan"

# Ensure output directories exist just in case
data_dir.mkdir(parents=True, exist_ok=True)
img_dir.mkdir(parents=True, exist_ok=True)

# --- 1. Sub-regular Coefficients (Omega_k) from Takeuchi & Inoue (2010) ---
omega_table = {
    ('Ti', 'Zr'): [-14.8, 0.4, -0.1, 0.0],
    ('Ti', 'Nb'): [2.0, 0.0, 0.0, 0.0],
    ('Ti', 'Mo'): [-3.6, 0.0, 0.0, 0.0],
    ('Ti', 'Ta'): [1.4, 0.0, 0.0, 0.0],
    ('Ti', 'W'):  [-5.7, 0.0, 0.0, 0.0],
    ('Zr', 'Nb'): [3.9, -0.2, 0.0, 0.0],
    ('Zr', 'Mo'): [-6.2, 0.5, -0.1, 0.0],
    ('Zr', 'Ta'): [2.7, -0.2, 0.0, 0.0],
    ('Zr', 'W'):  [-9.1, 0.7, -0.1, 0.0],
    ('Nb', 'Mo'): [-5.7, 0.1, 0.0, 0.0],
    ('Nb', 'Ta'): [0.0, 0.0, 0.0, 0.0],
    ('Nb', 'W'):  [-8.3, 0.1, 0.0, 0.0],
    ('Mo', 'Ta'): [-4.9, -0.1, 0.0, 0.0],
    ('Mo', 'W'):  [-0.2, 0.0, 0.0, 0.0],
    ('Ta', 'W'):  [-7.4, 0.1, 0.0, 0.0]
}

elements = ['W', 'Ta', 'Mo', 'Nb', 'Zr', 'Ti']
VEC_dict = {'W': 6, 'Ta': 5, 'Mo': 6, 'Nb': 5, 'Zr': 4, 'Ti': 4}
R_dict = {'W': 1.37, 'Ta': 1.43, 'Mo': 1.39, 'Nb': 1.46, 'Zr': 1.60, 'Ti': 1.47}
atomic_no_order = ['Ti', 'Zr', 'Nb', 'Mo', 'Ta', 'W']

# --- 2. Load Dataset ---
input_file = data_dir / 'RHEA.csv'
try:
    df = pd.read_csv(input_file)
except FileNotFoundError:
    print(f"Error: {input_file.name} not found in {data_dir}. Please check your directories.")
    exit()

# Extract original columns to preserve headers later
original_cols = ['W', 'Ta', 'Mo', 'Nb', 'Zr', 'Ti', 'Comp_point', 'Unit [at% or wt%]']
for col in original_cols:
    if col not in df.columns:
        df[col] = ''

# --- 3. Calculations ---
VEC_list, delta_list, Hmix_list = [], [], []

for idx, row in df.iterrows():
    c = {el: float(row[el]) / 100.0 for el in elements}
    
    # 3a. VEC Calculation
    vec = sum(c[el] * VEC_dict[el] for el in elements)
    VEC_list.append(vec)
    
    # 3b. Delta Calculation
    r_avg = sum(c[el] * R_dict[el] for el in elements)
    delta = 100 * np.sqrt(sum(c[el] * (1 - R_dict[el]/r_avg)**2 for el in elements))
    delta_list.append(delta)
    
    # 3c. Takeuchi H_mix Calculation (Eq 10)
    hmix = 0
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            el1, el2 = elements[i], elements[j]
            c1, c2 = c[el1], c[el2]
            if c1 == 0 and c2 == 0: continue
            
            pair = tuple(sorted([el1, el2], key=lambda x: atomic_no_order.index(x)))
            if pair in omega_table:
                O = omega_table[pair]
                sign = 1 if pair[0] == el1 else -1
                
                c_i_nor = c1 / (c1 + c2)
                c_j_nor = c2 / (c1 + c2)
                diff = c_i_nor - c_j_nor
                
                term = O[0] + (sign * O[1] * diff) + (O[2] * diff**2) + (sign * O[3] * diff**3)
                hmix += 4 * term * c1 * c2
    Hmix_list.append(hmix)

df['VEC'] = VEC_list
df['Delta_pct'] = delta_list
df['H_mix_kJ_mol'] = Hmix_list

# --- 4. Pass/Fail Logic (Y/N) ---
df['VEC_Pass'] = np.where((df['VEC'] >= 5.5) & (df['VEC'] <= 6.2), 'Y', 'N')
df['Delta_Pass'] = np.where(df['Delta_pct'] < 6.07, 'Y', 'N')
df['Hmix_Pass'] = np.where((df['H_mix_kJ_mol'] >= -6.5) & (df['H_mix_kJ_mol'] <= 2.7), 'Y', 'N')

df['Total_Pass'] = np.where((df['VEC_Pass'] == 'Y') & 
                            (df['Delta_Pass'] == 'Y') & 
                            (df['Hmix_Pass'] == 'Y'), 'Y', 'N')

# --- 5. Save to CSVs Preserving Headers ---
# Save the empirical calculations
calc_cols = original_cols + ['VEC', 'Delta_pct', 'H_mix_kJ_mol']
df[calc_cols].to_csv(data_dir / "QPcalc_RHEA.csv", index=False)
print(f"Saved calculations to {data_dir.name}/QPcalc_RHEA.csv")

# Save the evaluated map data
map_cols = calc_cols + ['VEC_Pass', 'Delta_Pass', 'Hmix_Pass', 'Total_Pass']
df[map_cols].to_csv(data_dir / "QPmap_RHEA.csv", index=False)
print(f"Saved evaluated data to {data_dir.name}/QPmap_RHEA.csv")

# --- 6. Hexagonal Mapping with Boundaries ---
angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
x_verts, y_verts = np.cos(angles), np.sin(angles)

df['Hex_X'] = [np.sum(np.array([row[el]/100.0 for el in elements]) * x_verts) for idx, row in df.iterrows()]
df['Hex_Y'] = [np.sum(np.array([row[el]/100.0 for el in elements]) * y_verts) for idx, row in df.iterrows()]

# Create numerical boolean columns for drawing contour lines
df['VEC_Bool'] = np.where(df['VEC_Pass'] == 'Y', 1, 0)
df['Delta_Bool'] = np.where(df['Delta_Pass'] == 'Y', 1, 0)
df['Hmix_Bool'] = np.where(df['Hmix_Pass'] == 'Y', 1, 0)
df['Total_Bool'] = np.where(df['Total_Pass'] == 'Y', 1, 0)

# Create triangulation for continuous boundary drawing
triang = tri.Triangulation(df['Hex_X'], df['Hex_Y'])

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

maps = [
    ('VEC', 'VEC (5.5 - 6.2)', 'viridis', 'VEC_Bool'),
    ('Delta_pct', 'Atomic Mismatch $\delta$ (< 6.07%)', 'plasma', 'Delta_Bool'),
    ('H_mix_kJ_mol', 'Mixing Enthalpy (-6.5 to 2.7)', 'coolwarm', 'Hmix_Bool')
]

# Plot first 3 maps
for i, (param, title, cmap, bool_col) in enumerate(maps):
    ax = axes[i]
    hexagon = patches.Polygon(np.column_stack([x_verts, y_verts]), closed=True, fill=False, edgecolor='black', linewidth=1.5)
    ax.add_patch(hexagon)
    
    for j, el in enumerate(elements):
        ax.text(x_verts[j] * 1.15, y_verts[j] * 1.15, el, fontsize=12, ha='center', va='center', fontweight='bold')
    
    # Scatter the data
    sc = ax.scatter(df['Hex_X'], df['Hex_Y'], c=df[param], cmap=cmap, alpha=0.8, s=15)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    
    # Draw the Pass/Fail Boundary (Red Line)
    ax.tricontour(triang, df[bool_col], levels=[0.5], colors='gray', linewidths=2.5, linestyles='--')
    
    ax.set_title(title, pad=15, fontweight='bold')
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.axis('off')

# Plot 4th Master Map (Intersection)
ax_master = axes[3]
hexagon_master = patches.Polygon(np.column_stack([x_verts, y_verts]), closed=True, fill=False, edgecolor='black', linewidth=1.5)
ax_master.add_patch(hexagon_master)

for j, el in enumerate(elements):
    ax_master.text(x_verts[j] * 1.15, y_verts[j] * 1.15, el, fontsize=12, ha='center', va='center', fontweight='bold')

# Plot fails in grey, passes in gold
fails = df[df['Total_Pass'] == 'N']
passes = df[df['Total_Pass'] == 'Y']
ax_master.scatter(fails['Hex_X'], fails['Hex_Y'], c='lightgrey', alpha=0.5, s=15, label='Failed Criteria')
ax_master.scatter(passes['Hex_X'], passes['Hex_Y'], c='gold', edgecolor='darkgoldenrod', alpha=0.9, s=25, label='SWEET SPOT')

ax_master.tricontour(triang, df['Total_Bool'], levels=[0.5], colors='black', linewidths=2)
ax_master.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2)
ax_master.set_title("Master Map: Overall Sweet Spot", pad=15, fontweight='bold')
ax_master.set_xlim(-1.3, 1.3); ax_master.set_ylim(-1.3, 1.3); ax_master.axis('off')

plt.tight_layout()

# Save image to the Quick_Property_Scan directory
img_output = img_dir / "QP_Hexagon_Maps.png"
plt.savefig(img_output, dpi=300)
print(f"Saved map visualization to {img_dir.name}/QP_Hexagon_Maps.png")
