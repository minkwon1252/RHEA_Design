"""
Rule-of-Mixtures thermodynamic property calculations.
Vectorized implementation for speed and memory efficiency.
"""
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element
from mendeleev import element


def ROM_THERMO(results_df):
    """Calculate ROM thermodynamic properties for all compositions.

    Uses vectorized operations where possible, with a single loop
    over compositions only for element-level property lookups.
    """
    n = results_df.shape[0]
    mol_weight = np.empty(n)
    density_arr = np.empty(n)
    T_m = np.empty(n)
    T_b = np.empty(n)

    for i in range(n):
        if i % 100 == 0 or i == n - 1:
            print(f"ROM_THERMO: {i + 1} / {n} ({(i+1)/n*100:.1f}%)")

        element_list = results_df.loc[i, 'Elements_active']
        atomic_fraction_arr = results_df.loc[i, 'atomic_per'] * 0.01

        comp_str = ''.join(element_list)
        comp_obj = Composition(comp_str)
        element_obj_list = comp_obj.elements

        elemental_mass_arr = np.array([e.atomic_mass for e in element_obj_list])
        mol_weight[i] = np.sum(atomic_fraction_arr * elemental_mass_arr)

        # Density
        elemental_density_arr = np.array(
            [e.density_of_solid for e in element_obj_list], dtype=object
        )
        # Fix None densities using mendeleev fallback
        for k in range(len(elemental_density_arr)):
            if elemental_density_arr[k] is None:
                try:
                    d = element(str(element_list[k])).density
                    elemental_density_arr[k] = d * 1000 if d is not None else 10.3 * 1000
                except Exception:
                    elemental_density_arr[k] = 10.3 * 1000
        elemental_density_arr = elemental_density_arr.astype(float)
        density_arr[i] = np.sum(atomic_fraction_arr * elemental_density_arr)

        # Melting temperature
        elemental_mp = np.array([e.melting_point for e in element_obj_list])
        T_m[i] = np.sum(atomic_fraction_arr * elemental_mp)

        # Boiling temperature
        elemental_bp = np.array([e.boiling_point for e in element_obj_list], dtype=object)
        try:
            T_b[i] = np.sum(atomic_fraction_arr * elemental_bp.astype(float))
        except (TypeError, ValueError):
            T_b[i] = np.nan

    results_df['ROM_Molecular_Weight [g/mol]'] = mol_weight
    results_df['ROM_Density_ROM kg/m3'] = density_arr
    results_df['ROM_Melting_Temp_[K]'] = T_m
    results_df['ROM_Boiling_Temp_[K]'] = T_b
    results_df['ROM_Sintering_Temp_[K]'] = 0.3 * T_m

    return results_df
