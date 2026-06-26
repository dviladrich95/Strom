from typing import Tuple
import numpy as np
import pandas as pd
import cvxpy as cp

class House:
    """A class representing thermal properties and constraints of a house.
    
    Attributes:
        C_air (float): Heat capacity of air [kWh/°C]
        C_wall (float): Heat capacity of walls [kWh/°C]
        R_interior (float): Thermal resistance between air and walls [°C/kW]
        R_exterior (float): Thermal resistance between walls and exterior [°C/kW]
        Q_heater (float): Maximum heating power [kW]
        Q_cooling (float): Maximum cooling power [kW]
        freq (str): Time frequency for calculations (e.g., '1h')
        T_min (float): Minimum allowed interior temperature [°C]
        T_max (float): Maximum allowed interior temperature [°C]
        T_interior_init (float): Initial interior temperature [°C]
        T_wall_init (float): Initial wall temperature [°C]
        P_base (float): Base electricity price [€/kWh]
    """
    
    def __init__(self, 
                 C_air: float = 0.26,
                 C_wall: float = 19.1,
                 R_interior: float = 0.42,
                 R_exterior: float = 8.86,
                 Q_heater: float = 2.0,
                 Q_cooling: float = 2.0,
                 T_min: float = 18.0,
                 T_max: float = 24.0,
                 T_interior_init: float = 18.5,
                 T_wall_init: float = 18.5,
                 P_base: float = 0.01,
                 freq: str = '1h') -> None:
        self.C_air = C_air
        self.C_wall = C_wall
        self.R_interior = R_interior
        self.R_exterior= R_exterior
        self.Q_heater=Q_heater
        self.Q_cooling=Q_cooling
        self.freq=freq
        self.T_min=T_min
        self.T_max=T_max
        self.T_interior_init = T_interior_init
        self.T_wall_init = T_wall_init
        self.P_base = P_base

def smooth_temperature(data: pd.Series,
                      window_hours: float,
                      dt: float) -> np.ndarray:
    """Smooth temperature data using a rolling mean.
    
    Args:
        data: Array of temperature values
        window_hours: Size of smoothing window in hours
        dt: Time step size in hours
    
    Returns:
        Smoothed temperature array
    """
    window_size = round(window_hours / dt)
    
    # Convert to pandas Series to use rolling mean
    data_series = pd.Series(data)
    
    # Apply rolling mean with center alignment to avoid lag
    smoothed = data_series.rolling(window=window_size, min_periods=1, center=True).mean()
    
    return smoothed.to_numpy()

def calculate_baseline_target(ext_temp_series: pd.Series,
                            T_min: float,
                            T_max: float,
                            resolution_hours: float) -> np.ndarray:
    """Calculate target temperature profile based on exterior temperature.
    
    Args:
        ext_temp_series: Array of exterior temperatures
        T_min: Minimum allowed temperature
        T_max: Maximum allowed temperature
        resolution_hours: Time resolution in hours
    
    Returns:
        Array of target temperatures clipped to [T_min, T_max]
    """
    # Smooth temperature over 24 hours
    smoothed_ext = smooth_temperature(ext_temp_series, 24, resolution_hours)
    
    # Clip values to stay within [T_min, T_max]
    target = np.clip(smoothed_ext, T_min, T_max)
    
    return target

def find_heating_output(temp_price_df: pd.DataFrame,
                       house: House,
                       heating_mode: str) -> pd.DataFrame:
    """Optimize heating/cooling output based on prices and exterior temperature.
    
    Args:
        temp_price_df: DataFrame with columns 'ExteriorTemperature' and 'Price'
        house: House object containing thermal parameters
        heating_mode: Either 'optimal' (minimize cost) or 'baseline' (follow target)
    
    Returns:
        DataFrame with optimal heating/cooling schedule and resulting temperatures
    
    Raises:
        ValueError: If no optimal solution is found
    """
    state_df = temp_price_df.copy()  # Make a copy of the dataframe
    state_df = state_df.resample(house.freq).interpolate(method='linear').bfill().ffill()
    
    state_df['Price'] = state_df['Price'] + house.P_base  # Add custom tolls and taxes

    freq_timedelta = pd.to_timedelta(house.freq)
    dt = freq_timedelta.total_seconds() / 3600.0  # Convert to hours

    time_steps = len(state_df)
    T_exterior = state_df["ExteriorTemperature"]
    # Initialize CVXPY variables
    heater_output = cp.Variable(time_steps)
    cooling_output = cp.Variable(time_steps)

    constraints = []

    constraints.append(heater_output >= 0.0)
    constraints.append(heater_output <= 1.0)

    constraints.append(cooling_output >= 0.0)
    constraints.append(cooling_output <= 1.0)

    # Define the state vector variable: T[0,:] = T_interior, T[1,:] = wall_temperature
    T = cp.Variable((2, time_steps))

    # Initial conditions
    constraints.append(T[0, 0] == house.T_interior_init)
    constraints.append(T[1, 0] == house.T_wall_init)

    # System matrix A (constant 2x2). Kept as a NumPy array so the whole dynamics
    # block canonicalizes as ONE matrix constraint instead of ~2*time_steps scalar
    # ones — the per-timestep Python loop was ~150x slower than the actual solve.
    A = np.array([
        [-1. / (house.R_interior * house.C_air),  1. / (house.R_interior * house.C_air)],
        [ 1. / (house.R_interior * house.C_wall), -((1. / house.R_interior) + (1. / house.R_exterior)) / house.C_wall],
    ])

    # Per-step forcing b (2 x time_steps): air row driven by the actuators, wall row
    # by the exterior temperature.
    b_air  = (house.Q_heater * heater_output - house.Q_cooling * cooling_output) / house.C_air
    b_wall = T_exterior.to_numpy() / (house.R_exterior * house.C_wall)
    b = cp.vstack([b_air, b_wall])

    # Vectorized forward-Euler dynamics, identical to the elementwise recursion
    # T[:, t+1] = T[:, t] + dt * (A @ T[:, t] + b[:, t]).
    constraints.append(
        T[:, 1:] == T[:, :-1] + dt * (A @ T[:, :-1] + b[:, :-1])
    )

    constraints.append(T[0, :] >= house.T_min)  # Interior temperature constraint
    constraints.append(T[0, :] <= house.T_max)  # Interior temperature constraint

    # Objective functions for different scenarios
    obj_cost = cp.sum(cp.multiply(state_df["Price"], dt * ((house.Q_heater +1e-4) * heater_output+(house.Q_cooling +1e-4) *cooling_output) ))

    if heating_mode == "optimal":
        obj = obj_cost
    elif heating_mode == "baseline":
        # Minimum-effort policy: hard constraints already enforce T ∈ [T_min, T_max];
        # this tiebreaker prefers u=0 whenever comfort is free (spring/autumn).
        obj = 1e-3 * cp.sum(heater_output + cooling_output)
    else:
        raise ValueError("Invalid heating mode. Choose 'optimal' or 'baseline'.")
    objective = cp.Minimize(obj)
    
    # Solve optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL, verbose=True)

    # Check if an optimal solution was found
    if problem.status == cp.OPTIMAL:
        # Add the output to the dataframe
        state_df['HeaterOutput'] = heater_output.value
        state_df['CoolingOutput'] = cooling_output.value
        state_df['InteriorTemperature'] = T[0, :].value
        state_df['WallTemperature'] = T[1, :].value
        state_df['Cost'] = state_df['Price'] * dt * (state_df['HeaterOutput'] * house.Q_heater + state_df['CoolingOutput'] * house.Q_cooling)
    else:
        print("No optimal solution found.")
        # Fill with NaN arrays
        state_df['HeaterOutput'] = np.full(time_steps, np.nan)
        state_df['CoolingOutput'] = np.full(time_steps, np.nan)
        state_df['InteriorTemperature'] = np.full(time_steps, np.nan)
        state_df['WallTemperature'] = np.full(time_steps, np.nan)
        state_df['Cost'] = np.full(time_steps, np.nan)
    
    return state_df

def find_heating_output_thermostat(
    temp_price_df: pd.DataFrame,
    house: House,
) -> pd.DataFrame:
    """Rule-based deadband thermostat — reactive, non-anticipative.

    Propagates the 2R2C ODE forward with forward Euler using the same
    dynamics as find_heating_output, but with a purely reactive rule:
    heat when T_air < T_min, cool when T_air > T_max, off otherwise.
    No future temperature or price information is used.
    """
    state_df = temp_price_df.copy()
    state_df = state_df.resample(house.freq).interpolate(method="linear").bfill().ffill()
    state_df["Price"] = state_df["Price"] + house.P_base

    dt = pd.to_timedelta(house.freq).total_seconds() / 3600.0
    time_steps = len(state_df)
    T_ext = state_df["ExteriorTemperature"].values

    # 2R2C matrix coefficients — identical to the LP formulation
    a00 = -1.0 / (house.R_interior * house.C_air)
    a01 =  1.0 / (house.R_interior * house.C_air)
    a10 =  1.0 / (house.R_interior * house.C_wall)
    a11 = -((1.0 / house.R_interior) + (1.0 / house.R_exterior)) / house.C_wall

    T_air  = house.T_interior_init
    T_wall = house.T_wall_init

    heater_out  = np.zeros(time_steps)
    cooling_out = np.zeros(time_steps)
    T_air_traj  = np.zeros(time_steps)
    T_wall_traj = np.zeros(time_steps)

    for t in range(time_steps):
        T_air_traj[t]  = T_air
        T_wall_traj[t] = T_wall

        if T_air < house.T_min:
            u_h, u_c = 1.0, 0.0
        elif T_air > house.T_max:
            u_h, u_c = 0.0, 1.0
        else:
            u_h, u_c = 0.0, 0.0

        heater_out[t]  = u_h
        cooling_out[t] = u_c

        b0 = (house.Q_heater * u_h - house.Q_cooling * u_c) / house.C_air
        b1 = T_ext[t] / (house.R_exterior * house.C_wall)

        T_air_new  = T_air  + dt * (a00 * T_air + a01 * T_wall + b0)
        T_wall_new = T_wall + dt * (a10 * T_air + a11 * T_wall + b1)
        T_air, T_wall = T_air_new, T_wall_new

    state_df["HeaterOutput"]         = heater_out
    state_df["CoolingOutput"]        = cooling_out
    state_df["InteriorTemperature"]  = T_air_traj
    state_df["WallTemperature"]      = T_wall_traj
    state_df["Cost"] = state_df["Price"] * dt * (
        state_df["HeaterOutput"]  * house.Q_heater +
        state_df["CoolingOutput"] * house.Q_cooling
    )
    return state_df


def compare_output_costs(
    temp_price_df: pd.DataFrame,
    house: House,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare optimal, LP-baseline, and thermostat heating strategies.

    Returns:
        Tuple of DataFrames (optimal_results, lp_baseline_results, thermostat_results)
    """
    optimal_state_df     = find_heating_output(temp_price_df, house, "optimal")
    baseline_state_df    = find_heating_output(temp_price_df, house, "baseline")
    thermostat_state_df  = find_heating_output_thermostat(temp_price_df, house)

    return optimal_state_df, baseline_state_df, thermostat_state_df