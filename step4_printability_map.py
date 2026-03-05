#!/usr/bin/env python
"""
Step 4: Printability Map Generation
====================================
Orchestrator script that calls modular, vectorized functions from Printability_Map/.

Reads data from Data_Space/ and models from Printability_Map/ET_Models/.
Designed for SLURM array jobs — each task processes a batch of compositions.

Usage:
    python step4_printability_map.py          # standalone (batch 0)
    sbatch run_slurm.sh                       # SLURM array
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from itertools import product
from pymatgen.core import Composition

# Add Printability_Map to path so we can import its modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PM_DIR = os.path.join(SCRIPT_DIR, 'Printability_Map')
sys.path.insert(0, PM_DIR)

# Import modular components
from rom_thermo import ROM_THERMO
from melt_pool import melt_pool_dimensionless
from et_models import scaled_ET, ET_NN, analytical_ET
from criteria import (
    keyholing_normalized, cooling_rate, keyholing_criteria,
    update_depth_columns, lof_criteria, balling,
)
from failure_modes import assign_failure_modes


def prepare_compositions(composition_df):
    """Parse composition dataframe into element lists, atomic/weight fractions.

    Returns element_list, atomic_arr, and a CBFV-ready DataFrame.
    """
    composition_df = composition_df.fillna(0).reset_index(drop=True)

    df_sep = composition_df.drop(['Comp_point', 'Unit [at% or wt%]', 'Unnamed: 0'], axis=1, errors='ignore')
    cols = df_sep.columns

    # Drop columns where all values are zero
    df_active = df_sep.loc[:, (df_sep != 0).any(axis=0)]
    comp_el = df_active.to_numpy() * 0.01

    # Identify active elements per row
    non_zero = df_sep.apply(lambda x: x > 0)
    element_present = non_zero.apply(lambda x: list(cols[x.values]), axis=1)

    # Extract non-zero composition ratios per row
    remove_zeros_arr = []
    for i in range(len(comp_el)):
        row_vals = comp_el[i]
        remove_zeros_arr.append(row_vals[row_vals != 0.])

    element_list = element_present.to_numpy()

    # Convert between at% and wt%
    at_per = []
    wt_per = []
    for i in range(len(composition_df)):
        comp_str = ''.join(element_list[i])
        comp_obj = Composition(comp_str)
        elemental_mass_arr = np.array([e.atomic_mass for e in comp_obj.elements])
        comp_ratio = np.around(remove_zeros_arr[i], 4)

        unit = composition_df.loc[i, 'Unit [at% or wt%]']
        if 'at' in unit:
            at_frac = comp_ratio * 100
            num = at_frac * elemental_mass_arr
            wt_frac = np.around(num / num.sum(), 4) * 100
            at_per.append(at_frac)
            wt_per.append(wt_frac)
        elif 'wt' in unit:
            wt_frac = comp_ratio * 100
            moles = wt_frac / elemental_mass_arr
            at_frac = np.around(moles / moles.sum(), 4) * 100
            at_per.append(at_frac)
            wt_per.append(wt_frac)
        else:
            raise ValueError(f'Unknown unit at row {i}: {unit}')

    atomic_arr = at_per
    return element_list, atomic_arr, wt_per


def generate_cbfv_features(element_list, atomic_arr, elem_prop_name='oliynyk'):
    """Generate Composition-Based Feature Vectors."""
    # Import from Printability_Map/cbfv
    cbfv_dir = os.path.join(PM_DIR, 'cbfv')
    sys.path.insert(0, cbfv_dir)
    import composition

    # Build formula strings
    formula_list = []
    for x in range(len(element_list)):
        parts = []
        for j in range(len(element_list[x])):
            parts.append(f"{element_list[x][j]}{atomic_arr[x][j]}")
        formula_list.append(''.join(parts))

    input_df = pd.DataFrame({'formula': formula_list, 'target': np.nan})
    feats, y, formulae, skipped = composition.generate_features(
        input_df, elem_prop=elem_prop_name,
        drop_duplicates=False, extend_features=False, sum_feat=False,
    )
    return feats


def reconstruct_element_columns(dimensionless_df):
    """Reconstruct per-element columns from Elements/Atomic_frac lists."""
    elements = dimensionless_df['Elements']
    values = dimensionless_df['Atomic_frac']

    unique_elements = sorted(set().union(*elements))

    # Build array directly instead of list-of-lists
    n = len(elements)
    data = np.zeros((n, len(unique_elements)))
    for i in range(n):
        el = elements.iloc[i]
        val = values.iloc[i]
        for j, ue in enumerate(unique_elements):
            if ue in el:
                data[i, j] = val[el.index(ue)]

    df_recon = pd.DataFrame(data, columns=unique_elements)
    df_combined = pd.concat([df_recon, dimensionless_df.reset_index(drop=True)], axis=1)
    df_combined.drop(['Elements', 'Atomic_frac'], axis=1, inplace=True)
    return df_combined


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    SECONDS = time.time()

    # --- 1. SLURM batch configuration ---
    batch_idx = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    total_batches = int(os.environ.get('TOTAL_BATCHES', 10))

    # --- 2. Data loading from Data_Space ---
    data_dir = os.path.join(SCRIPT_DIR, 'Data_Space')
    file_name = 'RHEA'

    thermo_calc_df_full = pd.read_csv(os.path.join(data_dir, 'THERMOCALC_RHEA.csv'))
    composition_df_full = pd.read_csv(os.path.join(data_dir, f'{file_name}.csv'))

    # Batch slicing
    rows_per_batch = len(composition_df_full) // total_batches
    start_row = batch_idx * rows_per_batch
    end_row = (batch_idx + 1) * rows_per_batch if batch_idx < total_batches - 1 else len(composition_df_full)

    thermo_calc_df = thermo_calc_df_full.iloc[start_row:end_row].reset_index(drop=True)
    composition_df = composition_df_full.iloc[start_row:end_row].reset_index(drop=True)

    # Free the full dataframes
    del thermo_calc_df_full, composition_df_full

    print(f"Batch {batch_idx}: rows {start_row}-{end_row} ({len(composition_df)} compositions)")

    # --- 3. Model selection ---
    e_t_model_type = 'NN'

    # --- 4. Processing parameters ---
    power_w = list(np.arange(100, 500, 1))          # 400 values
    vel_ms = list(np.arange(0.05, 2.01, 0.01))      # 196 values

    powder_thickness = [40]
    hatch_spacing = [80]
    d_laser = [80]
    powder_grain_size = [20]
    laser_wavelength_nm = [1070]
    T_amb = [288]

    dim_key_value = [1.8]
    dim_ball_value = [2.0]

    # --- 5. Build parameter grid ---
    print("Building parameter grid...")
    params = pd.DataFrame(
        list(product(power_w, vel_ms, powder_thickness, hatch_spacing,
                     d_laser, powder_grain_size, laser_wavelength_nm, T_amb)),
        columns=['Power', 'Velocity_m/s', 'Powder_thickness_um', 'hatch_spacing_um',
                 'd_laser_um', 'powder_grain_size_um', 'laser_wavelength_nm', 'amb_temp_K'],
    )
    print(f"Parameter grid: {len(params):,} rows")

    # --- 6. Composition processing ---
    print("Processing compositions...")
    element_list, atomic_arr, wt_per = prepare_compositions(composition_df)

    # --- 7. CBFV features ---
    print("Generating CBFV features...")
    feats = generate_cbfv_features(element_list, atomic_arr)
    results_df = feats.copy()
    results_df['Elements_active'] = element_list
    results_df['atomic_per'] = atomic_arr

    # --- 8. ROM thermodynamic properties ---
    print("Computing ROM thermodynamic properties...")
    results_df = ROM_THERMO(results_df)

    # --- 9. Merge with ThermoCalc data ---
    materials_features = thermo_calc_df.join(results_df)
    print("ROM_THERMO complete")
    del results_df, thermo_calc_df

    # --- 10. Melt pool dimensionless parameters (vectorized cross-product) ---
    print("Computing melt pool dimensionless parameters...")
    params, dimensionless_df = melt_pool_dimensionless(
        params, materials_features, element_list, atomic_arr
    )
    del materials_features
    print(f"Dimensionless DataFrame: {len(dimensionless_df):,} rows, {dimensionless_df.shape[1]} columns")

    # --- 11. ET Model ---
    if e_t_model_type == 'scaled':
        print("Running scaled ET model...")
        dimensionless_df = scaled_ET(dimensionless_df)

    elif e_t_model_type == 'NN':
        nn_dir = os.path.join(PM_DIR, 'ET_Models', 'ET_NN')
        print("Running NN ET model...")
        ET_result = ET_NN(dimensionless_df, nn_dir)
        dimensionless_df['length'] = ET_result['length']
        dimensionless_df['depth'] = ET_result['depth']
        dimensionless_df['width'] = ET_result['width']
        dimensionless_df['Tmax'] = ET_result['Tmax']
        dimensionless_df['Tmin'] = ET_result['Tmin']
        del ET_result

    elif e_t_model_type == 'analytical':
        print("Running analytical ET model...")
        ET_result = analytical_ET(dimensionless_df)
        dimensionless_df['length'] = ET_result['length']
        dimensionless_df['depth'] = ET_result['depth']
        dimensionless_df['width'] = ET_result['width']
        dimensionless_df['Tmax'] = ET_result['Tmax']
        dimensionless_df['Tmin'] = ET_result['Tmin']
        del ET_result
    else:
        raise ValueError(f"Unknown ET model type: {e_t_model_type}")

    print("ET model complete")

    # --- 12. Cooling rate (vectorized) ---
    print("Computing cooling rate...")
    dimensionless_df = cooling_rate(dimensionless_df)

    # --- 13. Keyholing criteria (vectorized) ---
    print("Computing keyholing criteria...")
    dimensionless_df = keyholing_normalized(dimensionless_df)
    dimensionless_df = keyholing_criteria(dimensionless_df, dim_key_value)

    # --- 14. Depth corrections (vectorized) ---
    print("Updating depth columns...")
    dimensionless_df = update_depth_columns(dimensionless_df, dim_key_value)

    # --- 15. LOF criteria (vectorized) ---
    print("Computing LOF criteria...")
    dimensionless_df = lof_criteria(dimensionless_df, dim_key_value)

    # --- 16. Balling criteria (vectorized) ---
    print("Computing balling criteria...")
    dimensionless_df = balling(dimensionless_df, T_amb=T_amb[0])

    # --- 17. Failure modes (vectorized) ---
    print("Assigning failure modes...")
    dimensionless_df = assign_failure_modes(dimensionless_df, dim_key_value, dim_ball_value)

    # --- 18. Reconstruct element columns ---
    print("Reconstructing element columns...")
    df_combined = reconstruct_element_columns(dimensionless_df)

    # --- 19. Save outputs ---
    output_dir = os.path.join(SCRIPT_DIR, 'Printability_Map_Output')
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f'Package_output_{file_name}_{e_t_model_type}_batch_{batch_idx}.csv')
    df_combined.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")

    duration = time.time() - SECONDS
    print(f"-----------------------------------")
    print(f"Batch {batch_idx} complete in {int(duration // 60)}m {int(duration % 60)}s")
    print(f"-----------------------------------")
