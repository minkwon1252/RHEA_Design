"""
Failure mode classification: vectorized assignment of failure modes
based on various criteria combinations.
"""
import numpy as np


def assign_failure_modes(dimensionless_df, dim_key_value, dim_ball_value):
    """Assign all 12 failure mode criteria sets using vectorized operations.

    This replaces 12 separate row-by-row loops with vectorized np.select.
    """
    # Pre-extract arrays once
    KH2 = dimensionless_df['Keyholing_KH2'].values == 1
    KH3 = dimensionless_df['Keyholing_KH3'].values == 1
    LOF2_KH2 = dimensionless_df['LOF2_KH2'].values == 1
    LOF2_KH3 = dimensionless_df['LOF2_KH3'].values == 1

    w = dimensionless_df['width'].values
    l = dimensionless_df['length'].values
    depth_KH2 = dimensionless_df['depth_KH2_corrected'].values
    depth_KH3 = dimensionless_df['depth_KH3'].values
    powder_t = dimensionless_df['Powder_thick_m'].values

    # Balling criterion 2: (pi*w)/l < sqrt(2/3)
    ball2 = (np.pi * w / l) < np.sqrt(2 / 3)

    choices = ['Keyholing', 'Balling', 'LOF', 'Success']

    # --- Set 12: (LOF2, KH3, Ball2) ---
    conds = [KH3, ball2, LOF2_KH3]
    dimensionless_df['failure_mode_LOF2KH3Ball2'] = np.select(conds, choices[:3], default='Success')

    # --- Set 10: (LOF2, KH2, Ball2) ---
    conds = [KH2, ball2, LOF2_KH2]
    dimensionless_df['failure_mode_LOF2KH2Ball2'] = np.select(conds, choices[:3], default='Success')

    # --- Set 6: (LOF1, KH3, Ball2) ---
    LOF1_KH3 = depth_KH2 <= powder_t  # LOF1 uses depth <= powder thickness
    conds = [KH3, ball2, LOF1_KH3]
    dimensionless_df['failure_mode_LOF1KH3Ball2'] = np.select(conds, choices[:3], default='Success')

    # --- Set 4: (LOF1, KH2, Ball2) ---
    conds = [KH2, ball2, LOF1_KH3]
    dimensionless_df['failure_mode_LOF1KH2Ball2'] = np.select(conds, choices[:3], default='Success')

    for dim_ball in dim_ball_value:
        # Balling criterion 1: l/w >= dim_ball
        ball1 = l / w >= dim_ball

        # --- Set 11: (LOF2, KH3, Ball1) ---
        conds = [KH3, ball1, LOF2_KH3]
        dimensionless_df[f'failure_mode_LOF2KH3Ball1_{dim_ball}'] = np.select(conds, choices[:3], default='Success')

        # --- Set 9: (LOF2, KH2, Ball1) ---
        conds = [KH2, ball1, LOF2_KH2]
        dimensionless_df[f'failure_mode_LOF2KH2Ball1_{dim_ball}'] = np.select(conds, choices[:3], default='Success')

        # --- Set 5: (LOF1, KH3, Ball1) ---
        LOF1_KH3_depth = depth_KH3 <= powder_t
        conds = [KH3, ball1, LOF1_KH3_depth]
        dimensionless_df[f'failure_mode_LOF1KH3Ball1_{dim_ball}'] = np.select(conds, choices[:3], default='Success')

        # --- Set 3: (LOF1, KH2, Ball1) ---
        LOF1_KH2_depth = depth_KH2 <= powder_t
        conds = [KH2, ball1, LOF1_KH2_depth]
        dimensionless_df[f'failure_mode_LOF1KH2Ball1_{dim_ball}'] = np.select(conds, choices[:3], default='Success')

    for dim_key in dim_key_value:
        KH1 = dimensionless_df[f'Keyholing_{dim_key}'].values == 1
        LOF2_KH1 = dimensionless_df[f'LOF2_KH1_{dim_key}'].values == 1
        depth_KH1 = dimensionless_df[f'depth_KH1_corr_{dim_key}'].values

        # --- Set 8: (LOF2, KH1, Ball2) ---
        conds = [KH1, ball2, LOF2_KH1]
        dimensionless_df[f'failure_mode_LOF2KH1Ball2_{dim_key}'] = np.select(conds, choices[:3], default='Success')

        # --- Set 2: (LOF1, KH1, Ball2) ---
        LOF1_KH1 = depth_KH1 <= powder_t
        conds = [KH1, ball2, LOF1_KH1]
        dimensionless_df[f'failure_mode_LOF1KH1Ball2_{dim_key}'] = np.select(conds, choices[:3], default='Success')

        for dim_ball in dim_ball_value:
            ball1 = l / w >= dim_ball

            # --- Set 7: (LOF2, KH1, Ball1) ---
            conds = [KH1, ball1, LOF2_KH1]
            dimensionless_df[f'failure_mode_LOF2KH1Ball1_{dim_key}_{dim_ball}'] = np.select(conds, choices[:3], default='Success')

            # --- Set 1: (LOF1, KH1, Ball1) ---
            conds = [KH1, ball1, LOF1_KH1]
            dimensionless_df[f'failure_mode_LOF1KH1Ball1_{dim_key}_{dim_ball}'] = np.select(conds, choices[:3], default='Success')

    return dimensionless_df
