"""
Eagar-Tsai melt pool models: Scaled, Neural Network, and Analytical.
Vectorized where possible for speed.
"""
import numpy as np
import pandas as pd
import os
import sys
import scipy.version
import ctypes
from scipy.integrate import quad
from pickle import load

from tensorflow.keras.models import load_model
from sklearn.pipeline import Pipeline


# ============================================================
# Scaled ET model (fully vectorized)
# ============================================================
def scaled_ET(dimensionless_df):
    """Vectorized scaled ET model - no row-by-row loops."""
    B = dimensionless_df['B'].values
    p = dimensionless_df['p'].values
    a = dimensionless_df['Beam_radium_m'].values

    # Peak temperature (valid for p <= 1)
    mask = p <= 1
    T_max = np.full(len(dimensionless_df), np.nan)
    T_liq = dimensionless_df['T_liquidus'].values
    T_max[mask] = T_liq[mask] * (2.1 - 1.9*p[mask] + 0.67*p[mask]**2) * B[mask]
    dimensionless_df['T_max'] = T_max

    log_p = np.log(p)
    log_B = np.log(B)

    # Depth (s)
    s = (a / np.sqrt(p)) * (
        0.008 - 0.0048*B - 0.047*p - 0.099*B*p
        + (0.32 + 0.015*B) * p * log_p
        + log_B * (0.0056 - 0.89*p + 0.29*p*log_p)
    ) * -1
    dimensionless_df['depth'] = s

    # Length (l)
    l = (a / p**2) * (
        0.0053 - 0.21*p + 1.3*p**2
        + (-0.11 - 0.17*B) * p**2 * log_p
        + B * (-0.0062 + 0.23*p + 0.75*p**2)
    )
    dimensionless_df['length'] = l

    # Width (w)
    w = (a / (B * p**3)) * (
        0.0021 - 0.047*p + 0.34*p**2 - 1.9*p**3 - 0.33*p**4
        + B * (0.00066 - 0.0070*p - 0.00059*p**2 + 2.8*p**3 - 0.12*p**4)
        + B**2 * (-0.00070 + 0.015*p - 0.12*p**2 + 0.59*p**3 - 0.023*p**4)
        + B**3 * (0.00001 - 0.00022*p + 0.0020*p**2 - 0.0085*p**3 + 0.0014*p**4)
    )
    dimensionless_df['width'] = w

    # Derived quantities
    V_pool = (np.pi / 6) * s * l * w
    dimensionless_df['V_pool'] = V_pool

    density = dimensionless_df['Density_kg/m3'].values
    M_pool = density * V_pool
    dimensionless_df['M_pool'] = M_pool

    H_after_boiling = dimensionless_df['H_after_boiling'].values
    mass_kg = dimensionless_df['mass_kg'].values
    T_b = dimensionless_df['T_b'].values
    absorp = dimensionless_df['Absorptivity'].values
    v = dimensionless_df['Velocity_m/s'].values
    P = dimensionless_df['Power'].values

    Eff_Cp = (H_after_boiling * M_pool) / (mass_kg * (T_b - 298))
    dimensionless_df['Eff_Cp_melt_pool'] = Eff_Cp

    T_max_est = (absorp * P * (l / v)) / Eff_Cp + 298
    dimensionless_df['T_max_est'] = T_max_est

    Res_t = l / v
    dimensionless_df['Residence_time'] = Res_t

    Q = absorp * P * Res_t
    dimensionless_df['Q_dep_energy_J'] = Q

    H_at_boiling = dimensionless_df['H_at_boiling'].values
    Q_b = (H_at_boiling * M_pool) / mass_kg
    dimensionless_df['H_MP_at_boiling_J'] = Q_b

    Q_pb = (H_after_boiling * M_pool) / mass_kg
    dimensionless_df['H_MP_after_boiling_J'] = Q_pb

    return dimensionless_df


# ============================================================
# NN ET model (batched prediction)
# ============================================================
def ET_NN(dimensionless_df, nn_model_dir):
    """Neural network ET model with batched prediction.

    Args:
        dimensionless_df: DataFrame with material+process parameters
        nn_model_dir: absolute path to the ET_NN model directory
    """
    def load_pipeline_keras(model, folder_name="model"):
        build_model = lambda: load_model(os.path.join(nn_model_dir, folder_name, model), compile=False)
        reg = Pipeline([('lstm', None)])  # placeholder
        from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
        reg_k = KerasRegressor(build_fn=build_model, epochs=50, batch_size=5, verbose=0)
        reg_k.model = build_model()
        return Pipeline([('lstm', reg_k)])

    # Build input
    data = pd.DataFrame({
        'v': dimensionless_df['Velocity_m/s'].values,
        'P': dimensionless_df['Power'].values,
        'twoSigma': dimensionless_df['Beam_diameter_m'].values,
        'A': dimensionless_df['Absorptivity'].values,
        'tMelt': dimensionless_df['T_liquidus'].values,
        'rho': dimensionless_df['Density_kg/m3'].values,
        'k': dimensionless_df['thermal_cond_liq'].values,
        'cp': dimensionless_df['Cp_J/kg'].values,
    })

    elements = dimensionless_df['Elements'].values
    atomic = dimensionless_df['Atomic_frac'].values

    data = data.dropna(axis=0, how='any')

    # Load scaler and classifier
    classifier_path = os.path.join(nn_model_dir, 'classifier_4_classes_119.pkl')
    scaler_path = os.path.join(nn_model_dir, 'scaler_x_all_119.pkl')
    Classifier_ = load(open(classifier_path, 'rb'))
    scaler = load(open(scaler_path, 'rb'))
    X = scaler.transform(data)

    Classes = Classifier_.predict(X)
    Classes_in_model = set(Classes)

    data_df = pd.DataFrame(X)
    data_df['8'] = Classes

    # Tmin_Tmax prediction (batch)
    Tmin_Tmax = load_pipeline_keras('Regression_Tmin_Tmax.h5', folder_name="Tmin_Tmax")
    Tmin_Tmax_values = Tmin_Tmax.predict(X)

    # Split by class
    Class_0 = data_df[data_df['8'] == 0].drop(columns=['8'])
    Class_1 = data_df[data_df['8'] == 1].drop(columns=['8'])
    Class_2 = data_df[data_df['8'] == 2].drop(columns=['8'])
    Class_3 = data_df[data_df['8'] == 3].drop(columns=['8'])

    result_frames = []

    if 0 in Classes_in_model and len(Class_0) > 0:
        Class_0 = Class_0.copy()
        Class_0['8'] = 0.0
        Class_0['9'] = 0.0
        Class_0['10'] = 0.0
        result_frames.append(Class_0)

    if 1 in Classes_in_model and len(Class_1) > 0:
        Class_1 = Class_1.copy()
        reg = load_pipeline_keras('Class_1_regression.h5', folder_name="Class_1_regression")
        y_1 = reg.predict(Class_1.to_numpy())
        scaler_y1 = load(open(os.path.join(nn_model_dir, 'Y_1_tranformation_1110.pkl'), 'rb'))
        y_1 = scaler_y1.inverse_transform(y_1)
        try:
            Class_1['8'] = y_1[:, 0]
            Class_1['9'] = y_1[:, 1]
        except IndexError:
            Class_1['8'] = y_1[0]
            Class_1['9'] = y_1[1]
        Class_1['10'] = 0.0
        result_frames.append(Class_1)

    if 2 in Classes_in_model and len(Class_2) > 0:
        Class_2 = Class_2.copy()
        reg = load_pipeline_keras('Class_2_regression.h5', folder_name="Class_2_regression")
        y_2 = reg.predict(Class_2.to_numpy())
        scaler_y2 = load(open(os.path.join(nn_model_dir, 'Y_2_tranformation_1110.pkl'), 'rb'))
        y_2 = scaler_y2.inverse_transform(y_2)
        try:
            Class_2['8'] = y_2[:, 0]
            Class_2['9'] = y_2[:, 1]
            Class_2['10'] = y_2[:, 2]
        except IndexError:
            Class_2['8'] = y_2[0]
            Class_2['9'] = y_2[1]
            Class_2['10'] = y_2[2]
        result_frames.append(Class_2)

    if 3 in Classes_in_model and len(Class_3) > 0:
        Class_3 = Class_3.copy()
        reg = load_pipeline_keras('Class_3_regression.h5', folder_name="Class_3_regression")
        y_3 = reg.predict(Class_3.to_numpy())
        scaler_y3 = load(open(os.path.join(nn_model_dir, 'Y_3_tranformation_1110.pkl'), 'rb'))
        y_3 = scaler_y3.inverse_transform(y_3)
        try:
            Class_3['8'] = y_3[:, 0]
            Class_3['9'] = y_3[:, 1]
            Class_3['10'] = y_3[:, 2]
        except IndexError:
            Class_3['8'] = y_3[0]
            Class_3['9'] = y_3[1]
            Class_3['10'] = y_3[2]
        result_frames.append(Class_3)

    AllClasses = pd.concat(result_frames, axis=0)
    AllClasses = AllClasses.rename(columns={"8": "length", "9": "width", "10": "depth"})

    try:
        Tmin_Tmax_df = pd.DataFrame(Tmin_Tmax_values, columns=['Tmax', 'Tmin'])
    except Exception:
        Tmin_Tmax_df = pd.DataFrame([list(Tmin_Tmax_values)], columns=['Tmax', 'Tmin'])

    Results_ET_NN = pd.DataFrame(data)
    Results_ET_NN = Results_ET_NN.join(AllClasses.iloc[:, 8:], how='outer')
    Results_ET_NN = Results_ET_NN.join(Tmin_Tmax_df, how='outer')
    Results_ET_NN = Results_ET_NN.abs()

    return Results_ET_NN


# ============================================================
# Analytical ET model (kept mostly as-is since it's integration-heavy)
# ============================================================
class _SimParam:
    def __init__(self, domain, spatialRes):
        self.domain = domain / 1.e6
        self.spatialRes = spatialRes / 1.e6

class _Beam:
    def __init__(self, twoSigma, P, v, A):
        self.twoSigma = twoSigma
        self.sigma = np.sqrt(2.0) * (self.twoSigma / 2.0)
        self.P = P
        self.v = v
        self.A = A

class _Material:
    def __init__(self, tMelt, k, rho, cp):
        self.tMelt = tMelt
        self.k = k
        self.rho = rho
        self.cp = cp


def _sasha_int(t, x, y, z, p):
    intpre = 1.0 / ((4*p*t + 1) * np.sqrt(t))
    intexp = (-(z**2)/(4*t)) - (((y**2) + (x-t)**2) / (4*p*t + 1))
    return intpre * np.exp(intexp)


def analytical_ET(dimensionless_df):
    """Analytical Eagar-Tsai model."""
    data = pd.DataFrame({
        'v': dimensionless_df['Velocity_m/s'].values,
        'P': dimensionless_df['Power'].values,
        'twoSigma': dimensionless_df['Beam_diameter_m'].values,
        'A': dimensionless_df['Absorptivity'].values,
        'tMelt': dimensionless_df['T_liquidus'].values,
        'k': dimensionless_df['thermal_cond_liq'].values,
        'rho': dimensionless_df['Density_kg/m3'].values,
        'cp': dimensionless_df['Cp_J/kg'].values,
    })
    data = data.dropna(axis=0, how='any')

    domain = np.array([1200.0, 1200.0, 1000.0])
    spatialRes = 1.0

    beam1 = _Beam(data['twoSigma'], data['P'], data['v'], data['A'])
    mat1 = _Material(data['tMelt'], data['k'], data['rho'], data['cp'])

    runSize = len(data)
    melt_length = np.zeros(runSize)
    melt_width = np.zeros(runSize)
    melt_depth = np.zeros(runSize)
    peakT = np.zeros(runSize)
    minT = np.zeros(runSize)

    for i in range(runSize):
        sim1 = _SimParam(domain.copy(), spatialRes)
        melt_length[i], melt_width[i], melt_depth[i], peakT[i], minT[i] = _eagarTsaiParam(beam1, mat1, sim1, i)
        if (i + 1) % 100 == 0:
            print(f'Analytical ET: {i+1} / {runSize}')

    results = pd.DataFrame({
        'v': beam1.v, 'P': beam1.P, 'twoSigma': beam1.twoSigma,
        'A': beam1.A, 'tMelt': mat1.tMelt, 'k': mat1.k,
        'rho': mat1.rho, 'cp': mat1.cp,
        'length': melt_length, 'width': melt_width,
        'depth': melt_depth, 'Tmax': peakT, 'Tmin': minT,
    })
    return results


def _eagarTsaiParam(beam, material, simParam, i):
    tMelt = material.tMelt.iloc[i]
    k = material.k.iloc[i]
    rho = material.rho.iloc[i]
    cp = material.cp.iloc[i]
    alpha = k / (rho * cp)
    P = beam.P.iloc[i]
    A = beam.A.iloc[i]
    v = beam.v.iloc[i]
    sigma = beam.sigma.iloc[i]
    delta = simParam.spatialRes

    xMin = round(-1.0 * beam.twoSigma.iloc[i] * 1.5, 5)
    xMax = simParam.domain[0]
    yMin = 0.0
    yMax = simParam.domain[1]
    zMin = -1.0 * simParam.domain[2]
    zMax = 0.0

    nx = int(np.round(abs(xMax - xMin) / delta)) + 1
    ny = int(np.round(abs(yMax - yMin) / delta)) + 1
    nz = int(np.round(abs(zMax - zMin) / delta)) + 1

    nxrange = np.linspace(xMin, xMax, nx)
    nyrange = np.linspace(yMin, yMax, ny)
    nzrange = np.linspace(zMin, zMax, nz)

    t0 = 300.0
    Ts = (A * P) / (np.pi * (k / alpha) * np.sqrt(np.pi * alpha * v * (sigma**3)))
    p = alpha / (v * sigma)

    tplanexy = np.zeros((ny, nx))
    tplanexz = np.zeros((nz, nx))

    for i1 in range(nx):
        x = nxrange[i1] / sigma
        for i2 in range(ny):
            y = nyrange[i2] / sigma
            tmpTemp = quad(_sasha_int, 0., np.inf, args=(x, y, 0.0, p))
            tplanexy[i2, i1] = t0 + Ts * tmpTemp[0]
        for i3 in range(nz):
            z = nzrange[i3] / np.sqrt((alpha * sigma) / v)
            tmpTemp = quad(_sasha_int, 0., np.inf, args=(x, 0.0, z, p))
            tplanexz[i3, i1] = t0 + Ts * tmpTemp[0]

    peakT = np.amax(tplanexy)
    minT = np.amin(tplanexy)

    if peakT > tMelt:
        meltXInd = np.squeeze(np.where(tplanexy[0, :] > tMelt))
        melt_length = np.amax(nxrange[meltXInd]) - np.amin(nxrange[meltXInd])

        yLength = 0
        zLength = 0
        for i1 in np.arange(np.size(meltXInd, axis=0)):
            meltYInd = np.squeeze(np.where(tplanexy[:, meltXInd[i1]] > tMelt))
            tmpYLength = np.amax(nyrange[meltYInd]) - np.amin(nyrange[meltYInd])
            if tmpYLength > yLength:
                yLength = tmpYLength
            meltZInd = np.squeeze(np.where(tplanexz[:, meltXInd[i1]] > tMelt))
            tmpZLength = np.amax(nzrange[meltZInd]) - np.amin(nzrange[meltZInd])
            if tmpZLength > zLength:
                zLength = tmpZLength

        if np.isclose(np.amax(nxrange[meltXInd]), xMax):
            simParam.domain[0] += sigma
            return _eagarTsaiParam(_Beam(beam.twoSigma, beam.P, beam.v, beam.A), material, simParam, 0)
        elif np.isclose(yLength, abs(yMax - yMin)):
            simParam.domain[1] += sigma
            return _eagarTsaiParam(_Beam(beam.twoSigma, beam.P, beam.v, beam.A), material, simParam, 0)
        elif np.isclose(zLength, abs(zMax - zMin)):
            simParam.domain[2] += sigma
            return _eagarTsaiParam(_Beam(beam.twoSigma, beam.P, beam.v, beam.A), material, simParam, 0)
        else:
            melt_width = yLength * 2
            melt_depth = zLength
    else:
        melt_width = 0.0
        melt_depth = 0.0
        melt_length = 0.0

    del tplanexy, tplanexz
    return melt_length, melt_width, melt_depth, peakT, minT
