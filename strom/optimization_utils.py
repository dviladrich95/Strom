from typing import Union, Tuple
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
    T_target = calculate_baseline_target(state_df["ExteriorTemperature"], house.T_min, house.T_max, dt)
    T_differential = 0.2
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

    # Define the system matrix A
    A = cp.vstack([
        [-1./(house.R_interior * house.C_air), 1./(house.R_interior * house.C_air)],
        [1./(house.R_interior * house.C_wall), -((1./house.R_interior) + (1./house.R_exterior)) / house.C_wall]
    ])

    # Dynamics constraints: For each time step, T[t+1] = T[t] + dt * (A @ T[t] + b_t)
    for t in range(time_steps - 1):
        b_t = cp.vstack([
            (house.Q_heater * heater_output[t] - house.Q_cooling * cooling_output[t]) / house.C_air,
            T_exterior.iloc[t] / (house.R_exterior * house.C_wall)
        ])
        constraints.append( T[0, t + 1] == T[0, t] + dt * (A[0,0] * T[0, t] + A[0,1] * T[1, t] + b_t[0]) )
        constraints.append( T[1, t + 1] == T[1, t] + dt * (A[1,0] * T[0, t] + A[1,1] * T[1, t] + b_t[1]) )

    constraints.append(T[0, :] >= house.T_min)  # Interior temperature constraint
    constraints.append(T[0, :] <= house.T_max)  # Interior temperature constraint

    # Objective functions for different scenarios
    obj_cost = cp.sum(cp.multiply(state_df["Price"], dt * ((house.Q_heater +1e-4) * heater_output+(house.Q_cooling +1e-4) *cooling_output) ))
    # make a modified objective function of obj_temp that takes the maximum between the difference and 0
    obj_temp = cp.sum(cp.abs(T[0, :] - (T_target - T_differential)) + cp.abs(T[0, :] - (T_target + T_differential)))


    tau = 0.01
    if heating_mode == "optimal":
        obj = obj_cost
    elif heating_mode == "baseline":
        obj = (1-tau)*obj_temp + tau*obj_cost # add a small amount of cost sensitivity to avoid unrealistic heater + cooling scenarios
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

def compare_output_costs(temp_price_df: pd.DataFrame,
                        house: House) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare optimal and baseline heating strategies.
    
    Args:
        temp_price_df: DataFrame with exterior temperature and price data
        house: House object containing thermal parameters
    
    Returns:
        Tuple of DataFrames (optimal_results, baseline_results)
    """
    optimal_state_df  = find_heating_output(temp_price_df, house, "optimal")
    baseline_state_df = find_heating_output(temp_price_df, house, "baseline")

    return optimal_state_df, baseline_state_df