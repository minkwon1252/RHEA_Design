"""
Melt pool dimensionless parameter calculations.
Fully vectorized: builds the parameter×material cross product using
numpy broadcasting instead of 6 nested Python loops.
"""
import numpy as np
import pandas as pd
from itertools import product as iterproduct
from scipy.constants import R


def precompute_material_columns(results_df):
    """Add derived thermodynamic columns to results_df (vectorized)."""
    mass_kg = results_df['ROM_Molecular_Weight [g/mol]'] * 0.001

    T_liquidus_298 = results_df['PROP LT (K)'] - 298
    results_df['equil_SR_K'] = results_df['PROP LT (K)'] - results_df['PROP ST (K)']
    results_df['H_total_J'] = results_df['EQUIL Liquidus H (J)'] - results_df['EQUIL RT H (J)']
    results_df['H_total_Jmol'] = results_df['EQUIL Liquidus H (J/mol)'] - results_df['EQUIL RT H (J/mol)']
    results_df['eff_Cp_(J/molK)'] = results_df['H_total_Jmol'] / T_liquidus_298
    results_df['H_melting_J'] = results_df['EQUIL Liquidus H (J)'] - results_df['EQUIL Solidus H (J)']
    results_df['H_boiling_(J/mol)'] = 10 * R * results_df['PROP LT (K)']
    results_df['ROM_H_boiling_(J/mol)'] = 10 * R * results_df['ROM_Melting_Temp_[K]']
    results_df['H_at_boiling_(J/mol)'] = (
        results_df['EQUIL Liquidus H (J/mol)']
        + results_df['Prop Liquidus Heat capacity (J/(mol K))']
        * (results_df['ROM_Boiling_Temp_[K]'] - results_df['PROP LT (K)'])
    )
    results_df['H_after_boiling_(J/mol)'] = (
        results_df['H_at_boiling_(J/mol)'] + results_df['H_boiling_(J/mol)']
    )

    density = results_df['Prop RT Density (g/cm3)'] * 1000
    results_df['eff_Cp_JkgK'] = results_df['eff_Cp_(J/molK)'] / mass_kg
    results_df['h_s_J/m3'] = results_df['eff_Cp_JkgK'] * results_df['PROP LT (K)'] * density

    return results_df


def melt_pool_dimensionless(params, results_df, element_list, atomic_arr):
    """Build the full dimensionless parameter DataFrame via vectorized cross-product.

    Instead of 6 nested for-loops with list.append(), we:
    1. Extract unique parameter values
    2. Build the full cross-product index with np.meshgrid
    3. Use numpy broadcasting for all calculations
    """
    # Precompute material columns
    results_df = precompute_material_columns(results_df)

    # Extract unique parameter values
    v_mm = params['Velocity_m/s'] * 1000
    t_mm = params['Powder_thickness_um'] * 0.001
    h_mm = params['hatch_spacing_um'] * 0.001
    d_laser_m_series = params['d_laser_um'] * 1e-6
    r_laser_m_series = d_laser_m_series / 2

    params['LED_J/mm'] = params['Power'] / v_mm
    params['SED_J/mm2'] = params['Power'] / (v_mm * h_mm)
    params['VED_J/mm3'] = params['Power'] / (v_mm * t_mm * h_mm)
    params['generic_VED_J/mm'] = (params['powder_grain_size_um'] / params['d_laser_um']) * params['VED_J/mm3']

    power = params['Power'].unique()
    vel_ms = params['Velocity_m/s'].unique()
    r_laser = r_laser_m_series.unique()
    t_m = params['Powder_thickness_um'].unique() * 1e-6
    h_m = params['hatch_spacing_um'].unique() * 1e-6
    laser_wavelength_m = params['laser_wavelength_nm'].unique() * 1e-9

    n_mat = len(results_df)
    mat_idx = np.arange(n_mat)

    # Build cross-product indices
    # Order: P, v, beam_rad, wave, t, h, mat_idx
    grid = np.array(
        list(iterproduct(
            np.arange(len(power)),
            np.arange(len(vel_ms)),
            np.arange(len(r_laser)),
            np.arange(len(laser_wavelength_m)),
            np.arange(len(t_m)),
            np.arange(len(h_m)),
            mat_idx,
        )),
        dtype=np.int32,
    )

    n_total = len(grid)
    print(f"Total cross-product rows: {n_total:,}")

    # Extract indices
    pi = grid[:, 0]
    vi = grid[:, 1]
    ri = grid[:, 2]
    wi = grid[:, 3]
    ti = grid[:, 4]
    hi = grid[:, 5]
    mi = grid[:, 6]

    # Map indices to values (all 1-D arrays of length n_total)
    P = power[pi]
    v = vel_ms[vi]
    beam_rad = r_laser[ri]
    wave = laser_wavelength_m[wi]
    t = t_m[ti]
    h = h_m[hi]

    # Material properties (index into results_df arrays)
    h_s = results_df['h_s_J/m3'].values[mi]
    elec_resist = results_df['Prop Liquidus Electric resistivity (Ohm m)'].values[mi]
    therm_diff = results_df['Prop Liquidus Thermal diffusivity (m2/s)'].values[mi]
    therm_cond = results_df['Prop Liquidus Thermal conductivity (W/(mK))'].values[mi]
    T_liq = results_df['PROP LT (K)'].values[mi]
    T_sol = results_df['PROP ST (K)'].values[mi]
    density_RT = results_df['Prop RT Density (g/cm3)'].values[mi] * 1000
    density_liq_val = results_df['Prop Liquidus Density (g/cm3)'].values[mi] * 1000
    eff_cp = results_df['eff_Cp_JkgK'].values[mi]
    cp_liq_val = results_df['Prop Liquidus Heat capacity (J/(mol K))'].values[mi]
    latent_heat = results_df['Latent Heat Fusion (J)'].values[mi]
    surf_tens = results_df['EQUIL Liquidus Surface Tension (N/m)'].values[mi]
    surf_tens_RT = results_df['EQUIL RT Surface Tension (N/m)'].values[mi]
    dvis = results_df['EQUIL Liquidus DVIS (Pa-s)'].values[mi]
    therm_cond_RT = results_df['EQUIL RT Thermal Conductivity (W/mK)'].values[mi]
    H_after_boiling_val = results_df['H_after_boiling_(J/mol)'].values[mi]
    H_at_boiling_val = results_df['H_at_boiling_(J/mol)'].values[mi]
    H_boiling_val = results_df['H_boiling_(J/mol)'].values[mi]
    mass_kg_val = results_df['ROM_Molecular_Weight [g/mol]'].values[mi] * 0.001
    T_b_val = results_df['ROM_Boiling_Temp_[K]'].values[mi]

    # Vectorized calculations
    absorp = 0.365 * np.sqrt(elec_resist / wave)

    denom = np.pi * np.sqrt(therm_diff * v * beam_rad**3)
    H_specific = (absorp * P) / denom
    delta_H = (2**(3/4) * absorp * P) / (np.sqrt(np.pi * therm_diff * v * beam_rad**3))

    B = delta_H / (2**(3/4) * np.pi * h_s)
    dwell = beam_rad / v
    therm_diff_time = np.sqrt((therm_diff * beam_rad) / v**2)
    p_val = therm_diff / (v * beam_rad)
    T_surface = (absorp * P) / (np.pi * density_RT * eff_cp * np.sqrt(therm_diff * v * beam_rad**3))

    P_dimless = (absorp * P) / (beam_rad * therm_cond * (T_liq - 298))
    v_dimless = (v * beam_rad) / therm_diff
    t_dimless = (2 * t) / beam_rad
    h_dimless = h / beam_rad

    VED_dimless = P_dimless / (v_dimless * t_dimless * h_dimless)
    SED_dimless = P_dimless / (v_dimless * h_dimless)
    LED_dimless = P_dimless / v_dimless

    Ma = -(surf_tens - surf_tens_RT) * (t * (T_liq - 298) / (therm_diff * dvis))

    # Build composition/atomic arrays
    comp_col = [element_list[j] for j in mi]
    atomic_col = [atomic_arr[j] for j in mi]

    # Build DataFrame
    dimensionless_df = pd.DataFrame({
        'Elements': comp_col,
        'Atomic_frac': atomic_col,
        'Power': P,
        'Velocity_m/s': v,
        'Hatch_spacing_m': h,
        'Powder_thick_m': t,
        'Beam_radium_m': beam_rad,
        'Beam_diameter_m': beam_rad * 2,
        'thermal_cond_liq': therm_cond,
        'Cp_J/kg': eff_cp,
        'Laser_WaveLength_m': wave,
        'T_solidus': T_sol,
        'Latent_Heat': latent_heat,
        'Density_liq_kg/m3': density_liq_val,
        'Cp_liq': cp_liq_val,
        'thermal_diff_liq': therm_diff,
        'surf_tens_liq': surf_tens,
        'thermal_cond_RT': therm_cond_RT,
        'T_liquidus': T_liq,
        'Density_kg/m3': density_RT,
        'Absorptivity': absorp,
        'H_specific_J/m3': H_specific,
        'h_s_J/m3': h_s,
        'H_normalized': H_specific / h_s,
        'Marangoni Number': Ma,
        'H_boiling_(J/mol)': H_boiling_val,
        'H_after_boiling': H_after_boiling_val,
        'H_at_boiling': H_at_boiling_val,
        'mass_kg': mass_kg_val,
        'T_b': T_b_val,
        'B': B,
        'p': p_val,
        'dwell_time_s': dwell,
        'Thermal_diffusion_time_s': therm_diff_time,
        'T_surface_K': T_surface,
        'dimensioness_P': P_dimless,
        'dimensioness_v': v_dimless,
        'dimensionless_t': t_dimless,
        'dimensionless_h': h_dimless,
        'LED_dimensionless': LED_dimless,
        'SED_dimensionless': SED_dimless,
        'VED_dimensionless': VED_dimless,
    })

    # Energy densities
    dimensionless_df['LED_J/mm'] = P / (v * 1000)
    dimensionless_df['SED_J/mm2'] = P / ((h * 1000) * (v * 1000))
    dimensionless_df['VED_J/mm3'] = P / ((h * 1000) * (v * 1000) * (t * 1000))

    del grid, pi, vi, ri, wi, ti, hi  # free memory
    return params, dimensionless_df
