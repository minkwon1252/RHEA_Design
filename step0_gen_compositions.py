#!/usr/bin/env python
"""
Step 1: Generate RHEA Compositions
====================================
Generates diverse alloy compositions across multiple sampling strategies.

QUICK START - Sampling modes:
-----------------------------------
1. Multiples-of-10 grid (all combos summing to 100%):
       python step1_generate_compositions.py --method grid10

2. Latin Hypercube Sampling (min=0, max=50 at%):
       python step1_generate_compositions.py --method lhs --min 0 --max 50

3. LHS with Zr+Ti <= 10% constraint:
       python step1_generate_compositions.py --method lhs_constraint --min 0 --max 50

4. Equiatomic search (all 6 or subset of elements):
       python step1_generate_compositions.py --method equiatomic
       python step1_generate_compositions.py --method equiatomic --elements W Ta Mo Nb

5. Equimolar variations:
       python step1_generate_compositions.py --method equimolar

Generate ALL FIVE sampling groups at once:
       python step1_generate_compositions.py --method all

Output folder is named after the element space (e.g., W-Ta-Mo-Nb-Zr-Ti).
Files are named sequentially, e.g., RHEA_space_lhs.csv.

USAGE:
    pip install pyDOE2 numpy pandas
    python step1_generate_compositions.py [options]
"""

import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import os
from itertools import product as iterproduct, combinations

try:
    from pyDOE2 import lhs
    HAS_PYDOE = True
except ImportError:
    HAS_PYDOE = False
    print("Warning: pyDOE2 not installed. Using random sampling instead.")
    print("For better coverage, install pyDOE2: pip install pyDOE2")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Define your element system (order determines default folder name)
ELEMENTS = ['W', 'Ta', 'Mo', 'Nb', 'Zr', 'Ti']

# Atomic numbers for sorting (used in folder naming)
ATOMIC_NUMBERS = {
    'W': 74, 'Ta': 73, 'Mo': 42, 'Nb': 41, 'Cr': 24, 'V': 23, 'Fe': 26, # FCC, melting temperature order
    'Re': 75, 'Hf': 72, 'Zr': 40, 'Ti': 22, # HCP, melting temperature order
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Al': 13, # BCC, melting temperature order 
    'C' : 6, 'Si': 14, 'B' : 5, 'O' : 8, # Dopants 
}

# Default sampling params
MIN_COMPOSITION = 0.0
MAX_COMPOSITION = 50.0
N_SAMPLES = 1000
RANDOM_SEED = 13

# Elements with constrained sum (group 3)
CONSTRAINED_ELEMENTS = ['Zr', 'Ti']
CONSTRAINT_MAX_SUM = 10.0  # at%


# =============================================================================
# FOLDER / FILE NAMING
# =============================================================================

def get_folder_name(elements):
    """
    Return folder name based on elements sorted by decreasing atomic number.
    Falls back to original order if element not in ATOMIC_NUMBERS dict.

    Parameters:
    -----------
    elements : list
        List of element symbols

    Returns:
    --------
    str
        Formatted folder name (e.g., 'W-Ta-Mo-Nb-Zr-Ti')
    """
    def atomic_num(el):
        return ATOMIC_NUMBERS.get(el, 0)

    sorted_els = sorted(elements, key=atomic_num, reverse=True)
    return '-'.join(sorted_els)


def ensure_folder(elements):
    """
    Create output folder named after the element space if it doesn't exist.

    Parameters:
    -----------
    elements : list
        List of element symbols

    Returns:
    --------
    str
        Path to the created/existing folder
    """
    folder = get_folder_name(elements)
    os.makedirs(folder, exist_ok=True)
    return folder


# =============================================================================
# SAMPLING FUNCTIONS
# =============================================================================

def generate_grid10_compositions(elements):
    """
    GROUP 1: All compositions where every element's at% is a multiple of 10
    and compositions sum to 100%. Elements with 0% are included.

    Strategy: Use cartesian product of [0, 10, ..., 100] for all elements,
    then filter for arrays that sum exactly to 100.

    Parameters:
    -----------
    elements : list
        List of element symbols

    Returns:
    --------
    pd.DataFrame
        DataFrame with all valid compositions
    """
    n_elements = len(elements)
    valid_combos = []
    for vals in iterproduct(range(0, 101, 10), repeat=n_elements):
        if sum(vals) == 100:
            valid_combos.append(list(vals))

    df = pd.DataFrame(valid_combos, columns=elements, dtype=float)
    return df


def generate_lhs_compositions(n_samples, elements, min_comp=0.0, max_comp=50.0, seed=None):
    """
    GROUP 2: Latin Hypercube Sampling with no inter-element constraints.

    Strategy: Generate LHS samples in [0, 1], scale to [min_comp, max_comp],
    normalize to 100%, clip to bounds again if necessary, and re-normalize.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    elements : list
        List of element symbols
    min_comp : float
        Minimum at% per element (default: 0.0)
    max_comp : float
        Maximum at% per element (default: 50.0)
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame
        DataFrame containing LHS generated compositions
    """
    n_elements = len(elements)

    if seed is not None:
        np.random.seed(seed)

    if HAS_PYDOE:
        lhs_samples = lhs(n_elements, samples=n_samples, criterion='maximin')
    else:
        lhs_samples = np.random.rand(n_samples, n_elements)

    # Scale to [min_comp, max_comp]
    compositions = min_comp + lhs_samples * (max_comp - min_comp)

    # Normalize to sum to 100%
    row_sums = compositions.sum(axis=1, keepdims=True)
    compositions = compositions / row_sums * 100.0

    # Clip to [min_comp, max_comp] and re-normalize
    compositions = np.clip(compositions, min_comp, max_comp)
    row_sums = compositions.sum(axis=1, keepdims=True)
    compositions = compositions / row_sums * 100.0

    df = pd.DataFrame(compositions, columns=elements)
    df = df.round(4)
    return df


def generate_lhs_constrained_zrti(n_samples, elements,
                                   constrained_els=None, constraint_max=10.0,
                                   min_comp=0.0, max_comp=50.0, seed=None,
                                   max_attempts=10):
    """
    GROUP 3: LHS sampling with sum(constrained_els) <= constraint_max at%.
    Other elements have no inter-element constraint beyond normalization.

    Strategy: Sample constrained elements first in [0, constraint_max], then
    sample remaining elements to fill up to 100%.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    elements : list
        Full list of element symbols in the search space
    constrained_els : list, optional
        Elements whose sum is constrained (default: ['Zr', 'Ti'])
    constraint_max : float
        Max sum for constrained elements in at% (default: 10.0)
    min_comp : float
        Minimum at% per element (default: 0.0)
    max_comp : float
        Maximum at% per element (default: 50.0)
    seed : int, optional
        Random seed for reproducibility
    max_attempts : int
        Retry attempts if rejection sampling needed (default: 10)

    Returns:
    --------
    pd.DataFrame
        DataFrame containing constrained LHS compositions
    """
    if constrained_els is None:
        constrained_els = CONSTRAINED_ELEMENTS

    if seed is not None:
        np.random.seed(seed)

    # Identify constrained vs free element indices
    constrained_idx = [elements.index(el) for el in constrained_els if el in elements]
    free_idx = [i for i in range(len(elements)) if i not in constrained_idx]
    n_constrained = len(constrained_idx)
    n_free = len(free_idx)

    results = []
    attempts = 0

    while len(results) < n_samples and attempts < max_attempts * n_samples:
        attempts += 1

        # Sample constrained elements uniformly in [0, constraint_max]
        constrained_vals = np.random.uniform(0, constraint_max, n_constrained)
        constrained_sum = constrained_vals.sum()

        if constrained_sum > constraint_max:
            # Scale down
            constrained_vals = constrained_vals / constrained_sum * constraint_max
            constrained_sum = constraint_max

        # Remaining budget for free elements
        remaining = 100.0 - constrained_sum

        # Sample free elements
        if n_free > 0:
            if HAS_PYDOE:
                # FIX: Removed criterion='maximin' to prevent 0-size array crash on samples=1
                free_vals = lhs(n_free, samples=1).flatten()
            else:
                free_vals = np.random.rand(n_free)
            # Scale free elements to fill the remaining budget
            free_vals = free_vals / free_vals.sum() * remaining
            free_vals = np.clip(free_vals, 0, max_comp)
            # Re-normalize after clipping
            if free_vals.sum() > 0:
                free_vals = free_vals / free_vals.sum() * remaining

        # Assemble full composition
        comp = np.zeros(len(elements))
        for i, idx in enumerate(constrained_idx):
            comp[idx] = constrained_vals[i]
        for i, idx in enumerate(free_idx):
            comp[idx] = free_vals[i]

        # Validate
        if abs(comp.sum() - 100.0) < 0.1 and constrained_vals.sum() <= constraint_max + 1e-6:
            # Normalize exactly to 100
            comp = comp / comp.sum() * 100.0
            results.append(comp)

    if len(results) < n_samples:
        print(f"Warning: Only generated {len(results)} valid compositions (requested {n_samples})")

    df = pd.DataFrame(results, columns=elements)
    df = df.round(4)
    return df


def generate_equiatomic_compositions(elements, all_elements=None):
    """
    GROUP 4: Equiatomic compositions for all non-empty subsets of 'elements'.

    Strategy: Iterate through all subset combinations of size k=1 to N.
    Assign 100/k at% to elements in the subset, 0% to the rest.

    Parameters:
    -----------
    elements : list
        Elements to consider for equiatomic combinations
    all_elements : list, optional
        Full element list for column ordering (defaults to elements)

    Returns:
    --------
    pd.DataFrame
        DataFrame with one row per subset combination
    """
    if all_elements is None:
        all_elements = elements

    rows = []
    for k in range(1, len(elements) + 1):
        for subset in combinations(elements, k):
            at_pct = 100.0 / k
            row = {el: 0.0 for el in all_elements}
            for el in subset:
                row[el] = round(at_pct, 4)
            rows.append(row)

    df = pd.DataFrame(rows, columns=all_elements)
    return df


# =============================================================================
# VALIDATION
# =============================================================================

def validate_compositions(df, elements, min_comp=0.0, max_comp=100.0, tolerance=0.1):
    """
    Check if generated compositions are geometrically and physically valid.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the compositions to check
    elements : list
        List of element columns to sum
    min_comp : float
        Minimum allowable at% per element (default: 0.0)
    max_comp : float
        Maximum allowable at% per element (default: 100.0)
    tolerance : float
        Tolerance for sum to 100% check (default: 0.1)

    Returns:
    --------
    bool
        True if all compositions are valid, False if warnings were triggered
    """
    valid = True

    row_sums = df[elements].sum(axis=1)
    if not np.allclose(row_sums, 100.0, atol=tolerance):
        bad = row_sums[~np.isclose(row_sums, 100.0, atol=tolerance)]
        print(f"  WARNING: {len(bad)} rows don't sum to 100%: range [{bad.min():.2f}, {bad.max():.2f}]")
        valid = False

    if (df[elements] < min_comp - tolerance).any().any():
        print(f"  WARNING: Some values below minimum {min_comp}%")
        valid = False

    if (df[elements] > max_comp + tolerance).any().any():
        print(f"  WARNING: Some values above maximum {max_comp}%")
        valid = False

    n_dup = df.duplicated(subset=elements).sum()
    if n_dup > 0:
        print(f"  WARNING: {n_dup} duplicate compositions found")

    return valid


def save_group(df, elements, method):
    """
    Save a composition dataframe to its assigned folder.

    Parameters:
    -----------
    df : pd.DataFrame
        Compositions to save
    elements : list
        Used to determine the destination folder
    method : str
        Method name used to format the file name 'RHEA_space_[method].csv'

    Returns:
    --------
    str
        Full filepath where the csv was saved
    """
    folder = ensure_folder(elements)
    filename = f"RHEA_space_{method}.csv"
    filepath = os.path.join(folder, filename)
    df.to_csv(filepath, index=False)
    print(f"  Saved {len(df)} compositions -> {filepath}")
    return filepath


# =============================================================================
# LEGACY FUNCTIONS
# =============================================================================

def generate_equimolar_variations(elements, perturbation=5.0, n_samples=50, seed=None):
    """
    GROUP 5: Compositions centered around equimolar with random perturbations.

    Strategy: Start with 100/N for all elements, add uniform noise,
    clip below 1.0, and normalize to 100%.

    Parameters:
    -----------
    elements : list
        List of element symbols
    perturbation : float
        Maximum uniform deviation from equimolar in at% (default: 5.0)
    n_samples : int
        Number of samples to generate (default: 50)
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame
        DataFrame of equimolar-variation compositions
    """
    n_elements = len(elements)
    equimolar = 100.0 / n_elements
    if seed is not None:
        np.random.seed(seed)
    compositions = np.ones((n_samples, n_elements)) * equimolar
    perturbations = np.random.uniform(-perturbation, perturbation, (n_samples, n_elements))
    compositions += perturbations
    compositions = np.clip(compositions, 1.0, None)
    row_sums = compositions.sum(axis=1, keepdims=True)
    compositions = compositions / row_sums * 100.0
    df = pd.DataFrame(compositions, columns=elements)
    df = df.round(4)
    return df


def add_corner_compositions(df, elements, min_val=1.0):
    """
    Add corner compositions (one element dominant) to improve boundary coverage.

    Parameters:
    -----------
    df : pd.DataFrame
        Existing compositions dataframe to append to
    elements : list
        List of element symbols
    min_val : float
        Minimum value assigned to all non-dominant elements (default: 5.0)

    Returns:
    --------
    pd.DataFrame
        DataFrame with original and appended corner compositions
    """
    n_elements = len(elements)
    corners = []
    for i, dominant_el in enumerate(elements):
        corner = {}
        remaining = 100.0 - min_val * (n_elements - 1)
        for j, el in enumerate(elements):
            corner[el] = remaining if i == j else min_val
        corners.append(corner)
    corner_df = pd.DataFrame(corners)
    return pd.concat([df, corner_df], ignore_index=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Main execution block. Parses command-line arguments and orchestrates
    the composition generation, validation, and file saving processes.
    """
    parser = argparse.ArgumentParser(
        description='Generate RHEA compositions for ThermoCalc analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--method', type=str, default='lhs',
        choices=['grid10', 'lhs', 'lhs_constraint', 'equiatomic', 'all',
                 'equimolar', 'dominant'], 
        help=(
            'Sampling method:\n'
            '  grid10         - All combos with at%% as multiples of 10 (Group 1)\n'
            '  lhs            - Latin Hypercube, min=0 max=50 (Group 2)\n'
            '  lhs_constraint - LHS with elements (Zr+Ti) <= 10%% constraint (Group 3)\n'
            '  equiatomic     - All equiatomic subsets (Group 4)\n'
            '  equimolar      - Equimolar + perturbations\n'
            '  all            - Generate all FIVE groups\n'
            '  dominant       - One element dominant'
        )
    )
    parser.add_argument('-n', '--n_samples', type=int, default=N_SAMPLES,
                        help=f'Number of compositions for LHS methods (default: {N_SAMPLES})')
    parser.add_argument('--min', type=float, default=MIN_COMPOSITION,
                        help=f'Minimum at%% per element for LHS (default: {MIN_COMPOSITION})')
    parser.add_argument('--max', type=float, default=MAX_COMPOSITION,
                        help=f'Maximum at%% per element for LHS (default: {MAX_COMPOSITION})')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help=f'Random seed (default: {RANDOM_SEED})')
    parser.add_argument('--elements', nargs='+', default=ELEMENTS,
                        help=f'Element list (default: {" ".join(ELEMENTS)})')
    parser.add_argument('--equi-elements', nargs='+', default=None,
                        help='Elements for equiatomic search (subset of --elements). '
                             'Default: all --elements')
    parser.add_argument('--zrti-max', type=float, default=CONSTRAINT_MAX_SUM,
                        help=f'Max sum for Zr+Ti constraint in at%% (default: {CONSTRAINT_MAX_SUM})')
    parser.add_argument('--add-corners', action='store_true',
                        help='Add corner compositions (one element dominant) to LHS output')

    args = parser.parse_args()
    elements = args.elements

    print("=" * 70)
    print("RHEA Composition Generator")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"Elements: {elements}")
    print(f"Method:   {args.method}")
    print(f"Output folder: {get_folder_name(elements)}/")
    print("=" * 70)

    methods_to_run = [args.method] if args.method != 'all' else [
        'grid10', 'lhs', 'lhs_constraint', 'equiatomic', 'equimolar'
    ]

    for method in methods_to_run:
        print(f"\n--- Method: {method} ---")

        # ------------------------------------------------------------------ #
        # GROUP 1: Multiples of 10
        # ------------------------------------------------------------------ #
        if method == 'grid10':
            print(f"Generating all compositions where each element is a multiple of 10%...")
            df = generate_grid10_compositions(elements)
            print(f"Generated {len(df)} compositions")
            print("Validating...")
            validate_compositions(df, elements, 0.0, 100.0)
            save_group(df, elements, 'grid10')

        # ------------------------------------------------------------------ #
        # GROUP 2: LHS (min=0, max=50)
        # ------------------------------------------------------------------ #
        elif method == 'lhs':
            print(f"Generating {args.n_samples} LHS compositions "
                  f"(min={args.min}%, max={args.max}%)...")
            df = generate_lhs_compositions(
                n_samples=args.n_samples,
                elements=elements,
                min_comp=args.min,
                max_comp=args.max,
                seed=args.seed
            )
            if args.add_corners:
                print(f"Adding {len(elements)} corner compositions...")
                df = add_corner_compositions(df, elements, min_val=args.min)
            df = df.drop_duplicates(subset=elements).reset_index(drop=True)
            print(f"Generated {len(df)} unique compositions")
            print("Validating...")
            validate_compositions(df, elements, 0.0, 100.0)
            save_group(df, elements, 'lhs')

        # ------------------------------------------------------------------ #
        # GROUP 3: LHS with Zr+Ti <= constraint
        # ------------------------------------------------------------------ #
        elif method == 'lhs_constraint':
            constrained = [el for el in CONSTRAINED_ELEMENTS if el in elements]
            if not constrained:
                print(f"Warning: None of {CONSTRAINED_ELEMENTS} found in element list. "
                      f"Running unconstrained LHS instead.")
                df = generate_lhs_compositions(args.n_samples, elements,
                                               args.min, args.max, args.seed)
            else:
                print(f"Generating {args.n_samples} LHS compositions "
                      f"with {'+'.join(constrained)} <= {args.zrti_max}%...")
                df = generate_lhs_constrained_zrti(
                    n_samples=args.n_samples,
                    elements=elements,
                    constrained_els=constrained,
                    constraint_max=args.zrti_max,
                    min_comp=args.min,
                    max_comp=args.max,
                    seed=args.seed
                )
                constrained_sum = df[constrained].sum(axis=1)
                violations = (constrained_sum > args.zrti_max + 0.1).sum()
                if violations:
                    print(f"  WARNING: {violations} rows exceed {'+'.join(constrained)} <= {args.zrti_max}%")
                else:
                    print(f"  ✓ All rows satisfy {'+'.join(constrained)} <= {args.zrti_max}%")
                    print(f"  {'+'.join(constrained)} sum range: "
                          f"[{constrained_sum.min():.2f}, {constrained_sum.max():.2f}]%")

            df = df.drop_duplicates(subset=elements).reset_index(drop=True)
            print(f"Generated {len(df)} unique compositions")
            print("Validating...")
            validate_compositions(df, elements, 0.0, 100.0)
            save_group(df, elements, 'lhs_constraint')

        # ------------------------------------------------------------------ #
        # GROUP 4: Equiatomic subsets
        # ------------------------------------------------------------------ #
        elif method == 'equiatomic':
            equi_elements = args.equi_elements if args.equi_elements else elements
            invalid = [el for el in equi_elements if el not in elements]
            if invalid:
                print(f"Warning: {invalid} not in element list, ignoring.")
                equi_elements = [el for el in equi_elements if el in elements]

            n_subsets = sum(1 for k in range(1, len(equi_elements)+1)
                            for _ in combinations(equi_elements, k))
            print(f"Generating equiatomic compositions for all {n_subsets} "
                  f"non-empty subsets of {equi_elements}...")
            df = generate_equiatomic_compositions(equi_elements, all_elements=elements)
            print(f"Generated {len(df)} compositions "
                  f"(2^{len(equi_elements)}-1 = {2**len(equi_elements)-1} subsets)")
            print("Validating...")
            validate_compositions(df, elements, 0.0, 100.0)
            save_group(df, elements, 'equiatomic')

        # ------------------------------------------------------------------ #
        # GROUP 5: Equimolar variations
        # ------------------------------------------------------------------ #
        elif method == 'equimolar':
            print(f"Generating {args.n_samples} equimolar-variation compositions...")
            perturbation = args.max - (100.0 / len(elements))
            df = generate_equimolar_variations(
                elements=elements,
                perturbation=perturbation,
                n_samples=args.n_samples,
                seed=args.seed
            )
            print(f"Generated {len(df)} compositions")
            save_group(df, elements, 'equimolar')

    print(f"\n{'='*70}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.method == 'all':
        folder = get_folder_name(elements)
        print(f"All outputs saved to folder: {folder}/")
        print(f"  RHEA_space_grid10.csv")
        print(f"  RHEA_space_lhs.csv")
        print(f"  RHEA_space_lhs_constraint.csv")
        print(f"  RHEA_space_equiatomic.csv")
        print(f"  RHEA_space_equimolar.csv")


if __name__ == "__main__":
    main()
