"""
=========================================================================================
RHEA PROPERTY CALCULATOR (v9) - THERMO-CALC MULTIPROCESSING ENGINE
=========================================================================================

DESCRIPTION:
    This script automates high-throughput CALPHAD calculations for Refractory High 
    Entropy Alloys (RHEA) using the TC-Python API. It is optimized for stability 
    on systems with limited RAM (8GB) and handles numerical edge cases common in 
    complex alloy spaces.

USAGE:
    1. Ensure Thermo-Calc 2026a and the TCHEA8 database are installed.
    2. Run: python step2_thermocalc_calculationsv9.py
    3. [STEP 1]: Select the project folder (e.g., W-Ta-Mo-Nb-Zr-Ti).
    4. [STEP 2]: Select the input composition CSV (e.g., RHEA_space_equiatomic.csv).
    5. The script will automatically distribute calculations across CPU workers.

INPUT:
    - A CSV file containing atomic compositions (at%) for W, Ta, Mo, Nb, Zr, and Ti.
    - Expected format: [Comp_point, W, Ta, Mo, Nb, Zr, Ti]

OUTPUT:
    - THERMOCALC_RHEA_xxx.csv: Full property dataset including thermal, physical, and thermodynamic data.
    - FAILED_RHEA_xxx.csv: Log of any compositions that failed to converge.

KEY STABILITY FEATURES:
    - Java RAM Cap: Limits background JVM instances to 1GB to prevent system freezing.
    - Trace Element Floor: Automatically replaces 0% concentrations with 0.001% 
      to prevent "QBQUILD" solver crashes and Java Fatal Errors.
    - Pure Pandas Chunking: Prevents data-type corruption during multiprocessing.

CALCULATED PROPERTIES EXPLANATION:
    1. SOLIDIFICATION (Scheil-Gulliver):
       - PROP LT/ST: Liquidus and Solidus temperatures (K).
       - Kou Criteria: Solidification freezing range (ΔT).
       - Crack Coefficient: Susceptibility index (ΔT / T_liquidus).

    2. THERMODYNAMICS (Equilibrium):
       - Enthalpy (H): Calculated at Room Temp, Solidus, and Liquidus (J/mol).
       - Latent Heat of Fusion: Energy released during phase change.
       - Stable Phases: List of phases present during solidification.

    3. PHYSICAL & TRANSPORT (Rule of Mixtures + Temperature Scaling):
       - Density: Mass per volume (g/cm³).
       - Thermal Conductivity: Heat transfer capability (W/mK).
       - Electrical Resistivity/Conductivity: Charge transport properties.
       - Thermal Diffusivity: Rate of heat spread (m²/s).
       - Surface Tension & Viscosity: Liquid metal flow properties at melting.

=========================================================================================
"""

import os
import sys

# --- LIMIT JAVA RAM BEFORE ANYTHING ELSE LOADS ---
# -Xmx1024m tells each Thermo-Calc session to use max 1GB of RAM
# os.environ['_JAVA_OPTIONS'] = '-Xmx1024m -Djava.net.preferIPv4Stack=true' # -Xmx4g if 4GB, -Xmx8g if 8GB

import pandas as pd
import numpy as np
import time
import glob
import multiprocessing
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import logging
# Suppress Thermo-Calc console output
logging.getLogger("tc_python").setLevel(logging.ERROR)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# =============================================================================
# CONFIGURATION
# =============================================================================
DATABASE = 'TCHEA8'
ELEMENTS = ['W', 'Ta', 'Mo', 'Nb', 'Zr', 'Ti']
DEPENDENT_ELEMENT = 'Ti'
MAX_WORKERS = 2  # Kept at 1 for 8GB laptop stability. Change to 2 if Firefox is closed.

ATOMIC_MASSES = {'W': 183.84, 'Ta': 180.9479, 'Mo': 95.95, 'Nb': 92.90637, 'Zr': 91.224, 'Ti': 47.867}
T_ROOM = 298.15
T_HIGH = 4000

# =============================================================================
# NAVIGATION LOGIC
# =============================================================================

def interactive_navigator():
    """Navigate folders and then select a CSV file."""
    all_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and not d.startswith('.')]
    if not all_dirs:
        print("No subdirectories found. Staying in current directory.")
        target_dir = "."
    else:
        print("\n[STEP 1] Select Project Folder:")
        print(f"{'[#]':<5} {'Folder Name'}")
        for i, d in enumerate(all_dirs):
            print(f"[{i}] {d}")
        
        try:
            d_choice = int(input(f"\nSelect folder number (0-{len(all_dirs)-1}): "))
            target_dir = all_dirs[d_choice]
        except:
            print("Invalid folder choice. Defaulting to current directory.")
            target_dir = "."

    search_path = os.path.join(target_dir, "RHEA_*.csv")
    csv_files = glob.glob(search_path)
    
    if not csv_files:
        print(f"Error: No RHEA_*.csv files found in '{target_dir}'!")
        sys.exit(1)
    
    print(f"\n[STEP 2] Select Input File in '{target_dir}':")
    for i, f in enumerate(csv_files):
        print(f"[{i}] {os.path.basename(f)}")
    
    try:
        f_choice = int(input(f"\nSelect file number (0-{len(csv_files)-1}): "))
        selected_file = csv_files[f_choice]
    except:
        print("Invalid file selection. Exiting.")
        sys.exit(1)
    
    base_name = os.path.basename(selected_file)
    if 'space_' in base_name:
        suffix = base_name.split('space_')[-1].replace('.csv', '')
    else:
        suffix = base_name.replace('.csv', '')
    
    output_name = os.path.join(target_dir, f"THERMOCALC_RHEA_{suffix}.csv")
        
    return selected_file, output_name

# =============================================================================
# CALCULATION LOGIC 
# =============================================================================

def calculate_properties(session, composition, elements, database):
    """Calculate properties for a single composition."""
    results = {}
    
    # --- SAFETY FIX ---
    # 1. Replace 0 with 0.001 (0.1 at%) to prevent QBQUILD/Java fatal error
    safe_comp = {el: max(val, 0.001) for el, val in composition.items()}
    # 2. Re-normalize to exactly 100% 
    total = sum(safe_comp.values())
    safe_comp = {el: (val / total) * 100.0 for el, val in safe_comp.items()}
    
    # Overwrite the composition for TC engine while maintaining logic
    calc_composition = safe_comp
    
    elements_to_set = [el for el in elements if el != DEPENDENT_ELEMENT]

    try:
        system = (session
            .select_database_and_elements(database, elements)
            .get_system())

        scheil_calc = system.with_scheil_calculation()

        for el in elements_to_set:
            scheil_calc.set_composition(el, calc_composition[el])

        scheil_calc.set_start_temperature(T_HIGH)
        scheil_result = scheil_calc.calculate()

        temps, fracs = scheil_result.get_values_of('T', 'f(s)')

        T_liquidus = max(temps)
        T_solidus = min(temps)

        results['PROP LT (K)'] = T_liquidus
        results['PROP ST (K)'] = T_solidus
        freezing_range = T_liquidus - T_solidus

        # Store temporarily, we will drop them later
        results['Scheil_T'] = list(temps)
        results['Scheil_fs'] = list(fracs)

        try:
            stable_phases = scheil_result.get_stable_phases()
            results['Stable Phases'] = ', '.join(stable_phases)
        except: pass

        def get_equilibrium_at_T(T):
            eq_calc = system.with_single_equilibrium_calculation()
            for el in elements_to_set:
                eq_calc.set_condition(f'X({el})', calc_composition[el] / 100.0)
            eq_calc.set_condition('T', T)
            eq_calc.set_condition('P', 101325)
            return eq_calc.calculate()

        try:
            eq_rt = get_equilibrium_at_T(T_ROOM)
            results['EQUIL RT H (J/mol)'] = eq_rt.get_value_of('H')
            results['EQUIL RT H (J)'] = results['EQUIL RT H (J/mol)']
        except: pass

        try:
            eq_sol = get_equilibrium_at_T(T_solidus)
            results['EQUIL Solidus H (J/mol)'] = eq_sol.get_value_of('H')
            results['EQUIL Solidus H (J)'] = results['EQUIL Solidus H (J/mol)']
        except: pass

        try:
            eq_liq = get_equilibrium_at_T(T_liquidus)
            results['EQUIL Liquidus H (J/mol)'] = eq_liq.get_value_of('H')
            results['EQUIL Liquidus H (J)'] = results['EQUIL Liquidus H (J/mol)']
        except: pass

        results['EQUIL Mass (g/mol)'] = sum(calc_composition[el] / 100 * ATOMIC_MASSES.get(el, 100) for el in elements)

        H_liq = results.get('EQUIL Liquidus H (J/mol)')
        H_sol = results.get('EQUIL Solidus H (J/mol)')
        if H_liq and H_sol: results['Latent Heat Fusion (J)'] = H_liq - H_sol

        results['Crack Coefficient'] = freezing_range / T_liquidus if T_liquidus else None
        results['Kou Criteria'] = freezing_range

        PURE_PROPS = {
            'W':  (19.25, 173.0, 5.3e-8, 24.3), 'Ta': (16.69, 57.5, 13.5e-8, 25.4),
            'Mo': (10.28, 138.0, 5.7e-8, 24.1), 'Nb': (8.57, 53.7, 15.2e-8, 24.6),
            'Zr': (6.52, 22.7, 42.0e-8, 25.4), 'Ti': (4.51, 21.9, 42.0e-8, 25.1),
        }
        PURE_SURF_TENS = {'W': 2.50, 'Ta': 2.15, 'Mo': 2.25, 'Nb': 1.95, 'Zr': 1.48, 'Ti': 1.65}
        PURE_VISC = {'W': 0.008, 'Ta': 0.007, 'Mo': 0.005, 'Nb': 0.004, 'Zr': 0.004, 'Ti': 0.005}

        density_rt = thermal_cond_rt = elec_resist_rt = cp_rt = surf_tens = visc = 0
        for el in elements:
            if el in calc_composition:
                frac = calc_composition[el] / 100.0
                if el in PURE_PROPS:
                    props = PURE_PROPS[el]
                    density_rt += frac * props[0]
                    thermal_cond_rt += frac * props[1]
                    elec_resist_rt += frac * props[2]
                    cp_rt += frac * props[3]
                if el in PURE_SURF_TENS: surf_tens += frac * PURE_SURF_TENS[el]
                if el in PURE_VISC: visc += frac * PURE_VISC[el]

        molar_mass = results['EQUIL Mass (g/mol)']

        results['Prop RT Density (g/cm3)'] = density_rt
        results['Prop RT Thermal conductivity (W/(mK))'] = thermal_cond_rt
        results['Prop RT Electric resistivity (Ohm m)'] = elec_resist_rt
        results['Prop RT Electric conductivity (S/m)'] = 1.0 / elec_resist_rt if elec_resist_rt > 0 else None
        results['Prop RT Heat capacity (J/(mol K))'] = cp_rt
        results['Prop RT Thermal resistivity (mK/W)'] = 1.0 / thermal_cond_rt if thermal_cond_rt > 0 else None

        rho_kg_m3 = density_rt * 1000
        Cp_J_kgK = cp_rt / (molar_mass / 1000)
        if rho_kg_m3 > 0 and Cp_J_kgK > 0:
            results['Prop RT Thermal diffusivity (m2/s)'] = thermal_cond_rt / (rho_kg_m3 * Cp_J_kgK)

        T_factor_sol = (T_solidus - T_ROOM) / 1000
        density_sol = density_rt * 0.98
        thermal_cond_sol = max(thermal_cond_rt * (1 - 0.1 * T_factor_sol), 10)
        elec_resist_sol = elec_resist_rt * (1 + 0.4 * T_factor_sol)

        results['Prop Solidus Density (g/cm3)'] = density_sol
        results['Prop Solidus Thermal conductivity (W/(mK))'] = thermal_cond_sol
        results['Prop Solidus Electric resistivity (Ohm m)'] = elec_resist_sol
        results['Prop Solidus Electric conductivity (S/m)'] = 1.0 / elec_resist_sol if elec_resist_sol > 0 else None
        results['Prop Solidus Heat capacity (J/(mol K))'] = cp_rt * 1.1
        results['Prop Solidus Thermal resistivity (mK/W)'] = 1.0 / thermal_cond_sol if thermal_cond_sol > 0 else None

        rho_sol_kg_m3 = density_sol * 1000
        Cp_sol_J_kgK = (cp_rt * 1.1) / (molar_mass / 1000)
        if rho_sol_kg_m3 > 0 and Cp_sol_J_kgK > 0:
            results['Prop Solidus Thermal diffusivity (m2/s)'] = thermal_cond_sol / (rho_sol_kg_m3 * Cp_sol_J_kgK)

        T_factor_liq = (T_liquidus - T_ROOM) / 1000
        density_liq = density_rt * 0.93
        thermal_cond_liq = max(thermal_cond_rt * 0.4, 8)
        elec_resist_liq = elec_resist_rt * (1 + 0.5 * T_factor_liq)

        results['Prop Liquidus Density (g/cm3)'] = density_liq
        results['Prop Liquidus Thermal conductivity (W/(mK))'] = thermal_cond_liq
        results['Prop Liquidus Electric resistivity (Ohm m)'] = elec_resist_liq
        results['Prop Liquidus Electric conductivity (S/m)'] = 1.0 / elec_resist_liq if elec_resist_liq > 0 else None
        results['Prop Liquidus Heat capacity (J/(mol K))'] = cp_rt * 1.15
        results['Prop Liquidus Thermal resistivity (mK/W)'] = 1.0 / thermal_cond_liq if thermal_cond_liq > 0 else None

        rho_liq_kg_m3 = density_liq * 1000
        Cp_liq_J_kgK = (cp_rt * 1.15) / (molar_mass / 1000)
        if rho_liq_kg_m3 > 0 and Cp_liq_J_kgK > 0:
            results['Prop Liquidus Thermal diffusivity (m2/s)'] = thermal_cond_liq / (rho_liq_kg_m3 * Cp_liq_J_kgK)

        results['EQUIL RT Surface Tension (N/m)'] = surf_tens
        results['EQUIL RT Thermal Conductivity (W/mK)'] = thermal_cond_rt
        results['EQUIL RT Density (g/cc)'] = density_rt
        results['EQUIL RT DVIS (Pa-s)'] = visc * 10
        results['EQUIL RT KVIS (m2/s)'] = results['EQUIL RT DVIS (Pa-s)'] / (density_rt * 1000) if density_rt > 0 else None

        results['EQUIL Solidus Surface Tension (N/m)'] = surf_tens * 0.98
        results['EQUIL Solidus Thermal Conductivity (W/mK)'] = thermal_cond_sol
        results['EQUIL Solidus Density (g/cc)'] = density_sol
        results['EQUIL Solidus DVIS (Pa-s)'] = visc * 2
        results['EQUIL Solidus KVIS (m2/s)'] = results['EQUIL Solidus DVIS (Pa-s)'] / (density_sol * 1000) if density_sol > 0 else None

        results['EQUIL Liquidus Surface Tension (N/m)'] = surf_tens * 0.95
        results['EQUIL Liquidus Thermal Conductivity (W/mK)'] = thermal_cond_liq
        results['EQUIL Liquidus Density (g/cc)'] = density_liq
        results['EQUIL Liquidus DVIS (Pa-s)'] = visc
        results['EQUIL Lquidus KVIS (m2/s)'] = visc / (density_liq * 1000) if density_liq > 0 else None

        if thermal_cond_rt and thermal_cond_sol:
            results['PROP Thermal Cond,RT_S (mK/W)'] = 1.0 / ((thermal_cond_rt + thermal_cond_sol) / 2)
        if thermal_cond_rt and thermal_cond_liq:
            results['PROP Thermal Cond,RT_L (mK/W)'] = 1.0 / ((thermal_cond_rt + thermal_cond_liq) / 2)
        if thermal_cond_rt and thermal_cond_sol and thermal_cond_liq:
            results['Prop AVG Thermal conductivity (W/(mK))'] = (thermal_cond_rt + thermal_cond_sol + thermal_cond_liq) / 3

        return results

    except Exception as e:
        return {'error': str(e)}

# =============================================================================
# MULTIPROCESSING WORKER
# =============================================================================

def process_chunk(chunk_df, chunk_id): # Added chunk_id for unique saving
    from tc_python import TCPython
    import logging
    logging.getLogger("tc_python").setLevel(logging.ERROR) # Stop TC printing
    
    results = []
    failed = []
    checkpoint_n = 50 # Save every 50 points
    
    try:
        with TCPython() as session:
            for i, (idx, row) in enumerate(chunk_df.iterrows()):
                composition = {el: row[el] for el in ELEMENTS}
                res = calculate_properties(session, composition, ELEMENTS, DATABASE)
                
                if 'error' not in res:
                    res['Comp_point'] = idx
                    for el in ELEMENTS: res[el] = composition[el]
                    results.append(res)
                else:
                    failed.append({'idx': idx, 'error': res['error']})
                
                # --- CHECKPOINT SAFETY ---
                if (i + 1) % checkpoint_n == 0:
                    temp_df = pd.DataFrame(results)
                    temp_df.to_csv(f"checkpoint_chunk{chunk_id}.csv", index=False)
                    
    except Exception as e:
        failed.append({'idx': 'GLOBAL', 'error': str(e)})
    
    # Clean up checkpoint file after successful completion of chunk
    if os.path.exists(f"checkpoint_chunk{chunk_id}.csv"):
        os.remove(f"checkpoint_chunk{chunk_id}.csv")
        
    return results, failed

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    # Navigate and get files
    input_path, output_path = interactive_navigator()
    
    print("\n" + "="*70)
    print(f"JOB STARTED: {datetime.now().strftime('%H:%M:%S')}")
    print(f"INPUT:       {input_path}")
    print(f"OUTPUT:      {output_path}")
    print(f"PROCESSORS:  {MAX_WORKERS}")
    print("="*70 + "\n")

    # Read the chosen file
    df_comp = pd.read_csv(input_path)
    n_total = len(df_comp)

    # For 3000 samples on 8GB RAM, use smaller chunks (250) to force JVM restarts.
    # This clears the Java memory "cache" frequently and prevents OOM.
    chunk_size = 250
    chunks = [df_comp.iloc[i : i + chunk_size] for i in range(0, n_total, chunk_size)]
    
    all_results = []
    all_failed = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Pass an ID to each chunk for the checkpoint filename
        futures = {executor.submit(process_chunk, chunk, i): chunk 
                   for i, chunk in enumerate(chunks)}
        
        if HAS_TQDM:
            pbar = tqdm(total=n_total, desc="Calculating RHEA Properties")
        
        for future in as_completed(futures):
            c_res, c_fail = future.result()
            all_results.extend(c_res)
            all_failed.extend(c_fail)
            if HAS_TQDM:
                pbar.update(len(futures[future]))
        
        if HAS_TQDM: pbar.close()

    # --- SAVE RESULTS AND V7 SUMMARY STATS ---
    print("\n" + "=" * 70)

    if all_results:
        df_results = pd.DataFrame(all_results)
        
        # Exact column reordering matching v7
        comp_cols = ['Comp_point'] + ELEMENTS + ['Unit [at% or wt%]', 'Database']
        existing = [c for c in comp_cols if c in df_results.columns]
        prop_cols = [c for c in df_results.columns if c not in comp_cols]
        df_results = df_results[existing + prop_cols]

        df_results.to_csv(output_path, index=False)
        
        print(f"✓ Saved {len(all_results)} results to {output_path}")
        print(f"✗ Failed: {len(all_failed)}")
        
        # Summary statistics exactly like v7
        print(f"\nLiquidus: {df_results['PROP LT (K)'].min():.0f} - {df_results['PROP LT (K)'].max():.0f} K")
        print(f"Solidus:  {df_results['PROP ST (K)'].min():.0f} - {df_results['PROP ST (K)'].max():.0f} K")
        print(f"Freezing range: {df_results['Kou Criteria'].min():.0f} - {df_results['Kou Criteria'].max():.0f} K")
        
        print(f"\nTotal columns: {len(df_results.columns)}")
    else:
        print("ERROR: No successful calculations!")
    
    if all_failed:
        fail_path = output_path.replace("THERMOCALC_", "FAILED_")
        pd.DataFrame(all_failed).to_csv(fail_path, index=False)
        print(f"\nFailed compositions logged to: {fail_path}")

if __name__ == "__main__":
    main()
