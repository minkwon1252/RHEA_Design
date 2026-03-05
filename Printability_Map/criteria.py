"""
Printability criteria calculations: keyholing, LOF, balling, cooling rate.
All functions are fully vectorized — no row-by-row Python loops.
"""
import numpy as np
import pandas as pd


def GS_depth_vectorized(df):
    """Vectorized Gusarov-Smurov keyhole depth estimate."""
    A = df['Absorptivity'].values
    P = df['Power'].values
    v = df['Velocity_m/s'].values
    T_b = df['T_b'].values
    thermal_diff = df['thermal_diff_liq'].values
    beam_size = df['Beam_radium_m'].values

    return (A * P) / (2 * np.pi * T_b) * np.log((beam_size + thermal_diff / v) / beam_size)


def keyholing_normalized(dimensionless_df, T_0=298):
    """Vectorized keyholing normalized enthalpy calculation."""
    P = dimensionless_df['Power'].values
    absorp = dimensionless_df['Absorptivity'].values
    T_liq = dimensionless_df['T_liquidus'].values
    Cp = dimensionless_df['Cp_J/kg'].values
    therm_diff = dimensionless_df['thermal_diff_liq'].values
    density = dimensionless_df['Density_liq_kg/m3'].values
    vel = dimensionless_df['Velocity_m/s'].values
    beam_rad = dimensionless_df['Beam_radium_m'].values

    Ke = (absorp * P) / ((T_liq - T_0) * np.pi * density * Cp * np.sqrt(therm_diff * vel * beam_rad**3))
    dimensionless_df['Ke'] = Ke
    dimensionless_df['norm_key_depth'] = 0.4 * (Ke - 1.4)
    dimensionless_df['Var_Key_deph'] = 0.36 * (Ke**0.86)

    therm_diff_length = np.sqrt((therm_diff * beam_rad) / vel)
    dimensionless_df['Thermal_Diffusion_Length'] = therm_diff_length
    dimensionless_df['Norm_Diffusion_Length'] = therm_diff_length / beam_rad

    return dimensionless_df


def cooling_rate(dimensionless_df, T_0=298):
    """Vectorized cooling rate calculation."""
    P = dimensionless_df['Power'].values
    absorp = dimensionless_df['Absorptivity'].values
    k = dimensionless_df['thermal_cond_liq'].values
    T_sol = dimensionless_df['T_solidus'].values
    T_liq = dimensionless_df['T_liquidus'].values
    vel = dimensionless_df['Velocity_m/s'].values

    Qp = P * absorp
    cr = 2 * np.pi * k * (T_sol - T_0) * (T_liq - T_0) * (vel / Qp)
    dimensionless_df['Cooling_rate'] = cr

    return dimensionless_df


def keyholing_criteria(dimensionless_df, dim_key_value):
    """Vectorized keyholing criteria checks."""
    T_boil = dimensionless_df['T_b'].values
    T_m = dimensionless_df['T_liquidus'].values
    H_norm = dimensionless_df['H_normalized'].values
    Ke = dimensionless_df['Ke'].values
    d = dimensionless_df['depth'].values
    w = dimensionless_df['width'].values

    criteria = (np.pi * T_boil) / T_m
    dimensionless_df['Keyholing_KH2'] = np.where(H_norm > criteria, 1.0, 0.0)
    dimensionless_df['Keyholing_KH3'] = np.where(Ke >= 6.0, 1.0, 0.0)

    for key_value in dim_key_value:
        dimensionless_df[f'Keyholing_{key_value}'] = np.where(d >= w / key_value, 1.0, 0.0)

    return dimensionless_df


def update_depth_columns(dimensionless_df, dim_key_value):
    """Vectorized depth correction for keyholing conditions."""
    gs_d = GS_depth_vectorized(dimensionless_df)
    depth = dimensionless_df['depth'].values

    # KH2 corrected depth
    kh2 = dimensionless_df['Keyholing_KH2'].values == 1.0
    dimensionless_df['depth_KH2_corrected'] = np.where(kh2, gs_d, depth)

    # KH1 corrected depth per dim_key_value
    for value in dim_key_value:
        kh1 = dimensionless_df[f'Keyholing_{value}'].values == 1.0
        dimensionless_df[f'depth_KH1_corr_{value}'] = np.where(kh1, gs_d, depth)

    # KH3 corrected depth
    kh3 = dimensionless_df['Keyholing_KH3'].values == 1.0
    dimensionless_df['depth_KH3'] = np.where(kh3, gs_d, depth)

    return dimensionless_df


def lof_criteria(dimensionless_df, dim_key_value):
    """Vectorized lack-of-fusion criteria."""
    w = dimensionless_df['width'].values
    h = dimensionless_df['Hatch_spacing_m'].values
    t = dimensionless_df['Powder_thick_m'].values

    # LOF2 with KH2 corrected depth
    d_kh2 = dimensionless_df['depth_KH2_corrected'].values
    with np.errstate(divide='ignore', invalid='ignore'):
        criteria_kh2 = (h / w)**2 + t / (t + d_kh2)
    dimensionless_df['LOF2_KH2'] = np.where(np.isfinite(criteria_kh2), np.where(criteria_kh2 >= 1, 1, 0), np.nan)

    # LOF2 with KH1 corrected depths
    for value in dim_key_value:
        d_kh1 = dimensionless_df[f'depth_KH1_corr_{value}'].values
        with np.errstate(divide='ignore', invalid='ignore'):
            criteria_kh1 = (h / w)**2 + t / (t + d_kh1)
        dimensionless_df[f'LOF2_KH1_{value}'] = np.where(
            np.isfinite(criteria_kh1), np.where(criteria_kh1 >= 1, 1, 0), np.nan
        )

    # LOF2 with KH3 depth
    d_kh3 = dimensionless_df['depth_KH3'].values
    with np.errstate(divide='ignore', invalid='ignore'):
        criteria_kh3 = (h / w)**2 + t / (t + d_kh3)
    dimensionless_df['LOF2_KH3'] = np.where(np.isfinite(criteria_kh3), np.where(criteria_kh3 >= 1, 1, 0), np.nan)

    return dimensionless_df


def balling(dimensionless_df, T_amb=288):
    """Vectorized balling criterion."""
    To = dimensionless_df['T_liquidus'].values
    Tf = dimensionless_df['T_solidus'].values
    L = dimensionless_df['Latent_Heat'].values
    p = dimensionless_df['Density_liq_kg/m3'].values
    c = dimensionless_df['Cp_liq'].values
    A = dimensionless_df['thermal_diff_liq'].values
    k = dimensionless_df['thermal_cond_liq'].values
    s = dimensionless_df['surf_tens_liq'].values
    ka = dimensionless_df['thermal_cond_RT'].values

    Ta = T_amb
    a = 100e-06  # meters

    tau_1 = ((a**2 * k) / (3 * A * ka)) * np.log((To - Ta) / (Tf - Ta))
    tau_2 = ((a**2 * k) / (3 * A * ka)) * (1 + ka / (2 * k)) * (L / (c * (Tf - Ta)))
    t_solid = 2 * (tau_1 + tau_2)
    t_spread = (p * a**3 / s)**0.5
    radius_b = a * 2.4 * (1 - np.exp(-0.9 * t_solid * (p * a**3 / s)**-0.5))

    dimensionless_df['Tau1'] = tau_1
    dimensionless_df['Tau2'] = tau_2
    dimensionless_df['Solidification Time (\u03BCs)'] = t_solid * 1e06
    dimensionless_df['Spreading Time (\u03BCs)'] = t_spread * 1e06
    dimensionless_df['Solidification/Spread Time'] = t_solid / t_spread
    dimensionless_df['Base Radius (\u03BCm)'] = radius_b * 1e06
    dimensionless_df['Base/Initial Radius'] = radius_b / a
    dimensionless_df['Balling'] = np.where(radius_b / a < 1.2599, 1.0, 0.0)

    return dimensionless_df
