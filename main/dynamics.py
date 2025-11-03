# TODO: reorganize all the code so that all parameters are defined together either in __init__ or in a separate function
from sympy import *
import numpy as np
import pandas as pd
from pathlib import Path

class Dynamics:
    def __init__(
            self,
            t_estimated_apogee: float = 13.571, # Estimated time until apogee
            dt: float = 0.01, # Time step for simulation
            x0: np.ndarray = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), # Initial state
        ):
        """Initialize the Dynamics class. Rocket body axis is aligned with z-axis.

        Args:
            t_estimated_apogee (float): Estimated time until apogee in seconds.
            dt (float): Time step for simulation in seconds.
            x0 (np.ndarray): Initial state vector.
        """

        self.t_estimated_apogee = t_estimated_apogee
        self.csv_path : Path = (
            Path(__file__).resolve().parents[1] / "data" / "openrocket_data.csv"
        )
        self.f_preburnout : Matrix = None
        self.f_postburnout : Matrix = None
        self.vars : list = None
        self.params : list = None
        self.f_params : Matrix = None
        self.f_subs : Matrix = None
        self.dt : float = dt
        self.x0 : np.ndarray = np.array(x0, dtype=float) if x0 is not None else None
        self.t0 : float = 0.0
        self.t_sym : Symbol = None

        # Logging
        self.states : list = [self.x0]
        self.ts : list = [self.t0]

        ## Uninitialized parameters ##
        
        # Rocket parameters
        self.I_0 : float = None # Initial moment of inertia in kg·m²
        self.I_f : float = None # Final moment of inertia in kg·m²
        self.I_3 : float = None # Rotational moment of inertia about z-axis in kg·m²
        self.x_CG_0 : float = None # Initial center of gravity location in meters
        self.x_CG_f : float = None # Final center of gravity location in meters
        self.m_0 : float = None # Initial rocket mass in kg
        self.m_f : float = None # Final rocket mass in kg
        self.m_p : float = None # Propellant mass in kg
        self.d : float = None # Rocket body diameter in meters
        self.L_ne : float = None # Length from nose to nozzle in meters
        self.Cnalpha_rocket : float = None # Rocket normal force coefficient derivative
        self.t_motor_burnout : float = None # Time to motor burnout in seconds
        self.t_launch_rail_clearance : float = None # Time to launch rail clearance in seconds

        # Environmental parameters
        self.rho : float = 1.225 # Air density kg/m^3
        self.g : float = -9.81 # Gravitational acceleration m/s^2

        # Fin parameters
        self.N : float = None # Number of fins
        self.Cr : float = None # Root chord in meters
        self.Ct : float = None # Tip chord in meters
        self.s : float = None # Span in meters
        self.Cnalpha_fin : float = None # Normal force coefficient normalized by angle of attack for 1 fin
        self.delta : float = None # Fin cant angle in degrees

    def setRocketParams(self, I_0: float, I_f: float, I_3: float,
                        x_CG_0: float, x_CG_f: float,
                        m_0: float, m_f: float, m_p: float,
                        d: float, L_ne: float, Cnalpha_rocket: float,
                        t_motor_burnout: float, t_launch_rail_clearance: float, t_estimated_apogee: float = 13.571):
        """Set the rocket parameters.

        Args:
            I_0 (float): Initial moment of inertia in kg·m².
            I_f (float): Final moment of inertia in kg·m².
            I_3 (float): Moment of inertia about the z-axis in kg·m².
            x_CG_0 (float): Initial center of gravity location in meters.
            x_CG_f (float): Final center of gravity location in meters.
            m_0 (float): Initial mass in kg.
            m_f (float): Final mass in kg.
            m_p (float): Propellant mass in kg.
            d (float): Rocket diameter in meters.
            L_ne (float): Length from nose to engine exit in meters.
            Cnalpha_rocket (float): Rocket normal force coefficient derivative.
            t_motor_burnout (float): Time until motor burnout in seconds.
            t_estimated_apogee (float, optional): Estimated time until apogee in seconds. Defaults to 13.571.
            t_launch_rail_clearance (float): Time until launch rail clearance in seconds.
        """
        self.I_0 = I_0
        self.I_f = I_f
        self.I_3 = I_3
        self.x_CG_0 = x_CG_0
        self.x_CG_f = x_CG_f
        self.m_0 = m_0
        self.m_f = m_f
        self.m_p = m_p
        self.d = d
        self.L_ne = L_ne
        self.Cnalpha_rocket = Cnalpha_rocket
        self.t_motor_burnout = t_motor_burnout
        self.t_estimated_apogee = t_estimated_apogee
        self.t_launch_rail_clearance = t_launch_rail_clearance
    
    def setEnvParams(self, rho: float, g: float):
        """Set the environmental parameters.

        Args:
            rho (float): Air density in kg/m^3.
            g (float): Gravitational acceleration in m/s^2.
        """
        self.rho = rho
        self.g = g

    def setFinParams(self, N: int, Cr: float, Ct: float, s: float, Cnalpha_fin: float, delta: float):
        """Set the fin parameters.

        Args:
            N (int): Number of fins.
            Cr (float): Fin root chord in meters.
            Ct (float): Fin tip chord in meters.
            s (float): Fin span in meters.
            Cnalpha_fin (float): Fin normal force coefficient derivative.
            delta (float): Fin cant angle in degrees.
        """
        self.N = N
        self.Cr = Cr
        self.Ct = Ct
        self.s = s
        self.Cnalpha_fin = Cnalpha_fin
        self.delta = delta
        
        
    def checkParamsSet(self):
        """Check if all necessary parameters have been set.

        Raises:
            ValueError: If any parameter is not set.
        """
        required_params = [
            'I_0', 'I_f', 'I_3',
            'x_CG_0', 'x_CG_f',
            'm_0', 'm_f', 'm_p',
            'd', 'L_ne',
            't_motor_burnout', 't_estimated_apogee', 't_launch_rail_clearance',
            'rho', 'g',
            'N', 'Cr', 'Ct', 's', 'Cnalpha_fin'
        ]
        for param in required_params:
            if not hasattr(self, param):
                raise ValueError(f"Parameter '{param}' is not set. Please set all necessary parameters before proceeding.")

    # TODO: Potential to implement our own line of best fit function instead of using numpy's polyfit
    def getLineOfBestFitTime(self, var: str, n: int = 1):
        """Get the line of best fit for the given data with a polynomial of degree n.

        Args:
            var (str): The variable to fit the line to.
            n (int, optional): The degree of the polynomial to fit. Defaults to 1.

        Returns:
            tuple: A tuple containing the coefficients of the polynomial and its degree.
        """
        # Load the CSV data into a DataFrame
        data = pd.read_csv(self.csv_path)
        t = data["# Time (s)"]
        y = None
        if (var == "mass"):
            y = data["Mass (g)"] / 1000  # Convert to kg
        elif (var == "inertia"):
            y = data["Longitudinal moment of inertia (kg·m²)"]
        elif (var == "CG"):
            y = data["CG location (cm)"] / 100  # Convert to m
        else:
            raise ValueError(
                "Invalid variable. Choose from: " \
                "'mass'," \
                "'inertia'," \
                "'CG'. " \
                )

        # Filter data based on motor burnout
        mask = t <= self.t_motor_burnout
        t = t[mask]
        y = y[mask]
        coeffs = np.polyfit(t, y, n)
        return coeffs, n

    # TODO: Potential to implement our own interpolation function instead of using numpy's interp
    def getThrust(self, t: float):
        """Get the thrust for the rocket at time t.

        Args:
            t (float): The time in seconds.

        Returns:
            dict: A dictionary containing inertia, mass, CG, and thrust at time t.
        """

        constants = dict()
        ## Post burnout constants ##
        I = Matrix([0.287, 0.287, 0.0035]) # Post burnout inertia values from OpenRocket, kg*m^2
        m = 2.589  # Post burnout mass from OpenRocket, kg
        CG = 63.5/100  # Post burnout CG from OpenRocket, m
        T = Matrix([0., 0., 0.])  # N

        motor_burnout = t > self.t_motor_burnout

        # TODO: for added efficiency, only call getLineOfBestFitTime once per variable and store the results
        if not motor_burnout:   
            coeffs_mass, degree_mass = self.getLineOfBestFitTime("mass")
            m = sum(coeffs_mass[i] * t**(degree_mass - i) for i in range(degree_mass + 1))

            coeffs_inertia, degree_inertia = self.getLineOfBestFitTime("inertia")
            I_long = sum(coeffs_inertia[i] * t**(degree_inertia - i) for i in range(degree_inertia + 1))
            I[0] = I_long # Ixx
            I[1] = I_long # Iyy

            coeffs_CG, degree_CG = self.getLineOfBestFitTime("CG")
            CG = sum(coeffs_CG[i] * t**(degree_CG - i) for i in range(degree_CG + 1))

            times = pd.read_csv(self.csv_path)["# Time (s)"]
            thrust = pd.read_csv(self.csv_path)["Thrust (N)"]
            T[2] = np.interp(t, times, thrust) # Thrust acting in z direction

        constants["inertia"] = I
        constants["mass"] = m
        constants["CG"] = CG
        constants["thrust"] = T
        
        return constants
    

    def quat_to_euler_xyz(self, q: np.ndarray, degrees=False, eps=1e-9):
        """
        Convert quaternion [w, x, y, z] to Euler angles (theta, phi, psi)
        using the intrinsic XYZ convention:
            theta: rotation about x (pitch)
            phi:   rotation about y (yaw)
            psi:   rotation about z (roll)
        Such that: R = Rz(psi) @ Ry(phi) @ Rx(theta)

        Args:
            q (array-like): Quaternion [w, x, y, z].
            degrees (bool): If True, return angles in degrees. (default: radians)
            eps (float):    Small epsilon to handle numerical edge cases.

        Returns:
            (theta, phi, psi): tuple of floats
        """
        # normalize to be safe
        n = np.linalg.norm(q)
        if n < eps:
            raise ValueError("Zero-norm quaternion")
        w = q[0] / n
        x = q[1] / n
        y = q[2] / n
        z = q[3] / n

        # Rotation matrix from quaternion (world<-body)
        # R[i,j] = row i, column j
        xx, yy, zz = x*x, y*y, z*z
        wx, wy, wz = w*x, w*y, w*z
        xy, xz, yz = x*y, x*z, y*z

        R = np.array([
            [1 - 2*(yy + zz),   2*(xy - wz),       2*(xz + wy)],
            [2*(xy + wz),       1 - 2*(xx + zz),   2*(yz - wx)],
            [2*(xz - wy),       2*(yz + wx),       1 - 2*(xx + yy)]
        ])

        # Extract for intrinsic XYZ (q = qz(psi) ⊗ qy(phi) ⊗ qx(theta))
        # From R = Rz(psi) Ry(phi) Rx(theta):
        #   phi   = asin(-R[2,0])
        #   theta = atan2(R[2,1], R[2,2])
        #   psi   = atan2(R[1,0], R[0,0])
        #
        # Handle numerical drift by clamping asin argument.
        s = -R[2, 0]
        s = np.clip(s, -1.0, 1.0)
        phi   = np.arcsin(s)
        theta = np.arctan2(R[2, 1], R[2, 2])

        # If cos(phi) ~ 0 (gimbal lock), fall back to a stable computation for psi
        if abs(np.cos(phi)) < eps:
            # At gimbal lock, theta and psi are coupled; choose a consistent psi:
            # Use elements that remain well-defined:
            # when cos(phi) ~ 0, use psi from atan2(-R[0,1], R[1,1])
            psi = np.arctan2(-R[0, 1], R[1, 1])
        else:
            psi = np.arctan2(R[1, 0], R[0, 0])

        if degrees:
            return np.degrees(theta), np.degrees(phi), np.degrees(psi)
        return theta, phi, psi


    def R_BW_from_q(self, qw, qx, qy, qz):
        """Convert a quaternion to a rotation matrix. World to body frame.

        Args:
            qw (float): The scalar component of the quaternion.
            qx (float): The x component of the quaternion.
            qy (float): The y component of the quaternion.
            qz (float): The z component of the quaternion.

        Returns:
            Matrix: The rotation matrix from world to body frame.
        """
        s = (qw**2 + qx**2 + qy**2 + qz**2)**-Rational(1,2) # Normalizing factor
        qw, qx, qy, qz = qw*s, qx*s, qy*s, qz*s # Normalized quaternion components

        xx,yy,zz = qx*qx, qy*qy, qz*qz
        wx,wy,wz = qw*qx, qw*qy, qw*qz
        xy,xz,yz = qx*qy, qx*qz, qy*qz
        return Matrix([
            [1-2*(yy+zz),   2*(xy+wz),   2*(xz-wy)],
            [2*(xy-wz),     1-2*(xx+zz), 2*(yz+wx)],
            [2*(xz+wy),     2*(yz-wx),   1-2*(xx+yy)]
        ])


    def deriveEOM(self, post_burnout: bool):
        """Get the equations of motion for the rocket. Sets self.f_preburnout or self.f_postburnout.

        ## Assumptions:
        - Rocket body axis is aligned with z-axis
        - No centrifugal forces are considered to simplify AoA and beta calculations
        - Coefficient of lift is approximated as 2*pi*AoA (thin airfoil theory)
        - Thrust acts only in the z direction of the body frame
        - No wind or atmospheric disturbances are considered
        - Density of air is constant at 1.225 kg/m^3

        ## Notes:
        - The state vector is [w1, w2, w3, v1, v2, v3, qw, qx, qy, qz] where w is angular velocity, v is linear velocity, and q is the quaternion.
        - The input vector is [delta1] where delta1 is the aileron angle
        - Thrust, mass, and inertia are time-varying based on the motor burn state
        - Normal force coefficient Cn is modeled as a polynomial function of velocity, with different coefficients pre- and post-motor burnout
        - Drag force Fd is modeled as a quadratic function of velocity magnitude
        - Lift force Fl is modeled using thin airfoil theory, proportional to angle of attack (AoA)
        - Corrective moment coefficient C is modeled as a function of velocity magnitude, normal force coefficient Cn, stability margin SM, and rocket diameter
        - Normal force coefficient derivative Cnalpha is modeled as Cn * (AoA / (AoA^2 + aoa_eps^2)) to ensure smoothness at AoA = 0
        - Stability margin SM is modeled as a polynomial function of AoA
        - Small terms are added to avoid division by zero in velocity magnitude and AoA calculations (denoted as eps and aoa_eps)
        - All polynomial equations are determined from experimental OpenRocket data and curve fitting using Google Sheets
        - Piecewise functions are used to bound certain variables (e.g., AoA, Cnalpha, C) to ensure numerical stability and physical realism

        ## Usage:
        - To derive the full set of equations of motion, call setup_eom() which derives both pre- and post-burnout EOMs.

        Args:
            post_burnout (bool): Whether the rocket is in the post-burnout phase.
        """
        self.checkParamsSet()
        w1, w2, w3, v1, v2 = symbols('w_1 w_2 w_3 v_1 v_2', real = True) # Angular and linear velocities
        v3 = symbols('v_3', real = True, positive = True) # Longitudinal velocity, assumed positive during flight
        qw, qx, qy, qz = symbols('q_w q_x q_y q_z', real = True) # Quaternion components
        I1, I2, I3 = symbols('I_1 I_2 I_3', real = True) # Moments of inertia
        T1, T2, T3 = symbols('T_1 T_2 T_3', real = True) # Thrusts
        mass, rho, d, g, CG = symbols('m rho d g CG', real = True) # Mass, air density, diameter, gravity, center of gravity
        delta = symbols('delta', real = True) # Fin cant angle
        Cnalpha_fin, Cnalpha_rocket = symbols('Cnalpha_fin Cnalpha_rocket', real = True, positive = True) # Fin and rocket normal force coefficient derivatives
        Cr, Ct, s = symbols('Cr Ct s', real = True, positive = True) # Fin root chord, tip chord, span
        N = symbols('N', real = True, positive = True) # Number of fins
        t_sym = symbols('t', real = True) # Time symbol for Heaviside function
        
        self.t_sym = t_sym
        H = Heaviside(t_sym - Float(self.t_launch_rail_clearance), 0)  # 0 if t < t_launch_rail_clearance, 1 if t >= t_launch_rail_clearance

        epsAoA = Float(1e-3)  # Small term to avoid division by zero in AoA calculation
        AoA = atan2(sqrt(v1**2 + v2**2), v3 + epsAoA) # Angle of attack
        AoA_eff = Piecewise(
            (0,   Abs(AoA) <= epsAoA),                # inside deadband
            (Min(Abs(AoA), 15 * pi / 180) * (AoA/Abs(AoA)), True)  # ±15°
        )

        eps = Float(1e-3)  # Small term to avoid division by zero
        v = Matrix([v1, v2, v3]) # Velocity vector
        v_mag = sqrt(v1**2 + v2**2 + v3**2 + eps**2) # Magnitude of velocity with small term to avoid division by zero
        vhat = v / v_mag  # Unit vector in direction of velocity

        ## Rocket reference area ##
        A = pi * (d/2)**2 # m^2

        ## Thrust ##
        T : Matrix = Matrix([T1, T2, T3])  # Thrust vector, T1 and T2 are assumed 0

        ## Gravity ##
        Fg_world = Matrix([0.0, 0.0, mass * g])
        R_world_to_body = self.R_BW_from_q(qw, qx, qy, qz)  # Rotation matrix from world to body frame
        Fg : Matrix = R_world_to_body * Fg_world  # Transform gravitational force to body frame

        ## Drag Force ##
        Fd_mag = -(0.627 + -0.029*v_mag + 1.95e-3*v_mag**2) # Drag force approximation
        Fd : Matrix = Fd_mag * vhat # Drag force vector

        ## Lift Force ##
        eps_beta = Float(1e-3)
        nan_guard = sqrt(v1**2 + v2**2 + eps_beta**2)
        beta = 2 * atan2(v2, nan_guard + v1) # Equivalent to atan2(v2, v1) but avoids NaN at (0,0)
        L = H * 1/2 * rho * v_mag**2 * (2 * pi * AoA_eff) * A # Lift force approximation
        nL = Matrix([
            -cos(AoA_eff) * cos(beta),
            -cos(AoA_eff) * sin(beta),
            sin(AoA_eff)
        ]) # Lift direction unit vector
        Fl : Matrix = L * nL # Lift force vector

        ## Total Forces ##
        F = T + Fd + Fl + Fg # Thrust + Drag + Lift + Gravity

        ## Stability Margin ##
        AoA_deg = AoA_eff * 180 / pi # Convert AoA to degrees for polynomial fit
        SM = 0
        if not post_burnout:
            SM = 2.8 + -0.48*AoA_deg + 0.163*AoA_deg**2 + -0.0386*AoA_deg**3 + 5.46E-03*AoA_deg**4 + -4.61E-04*AoA_deg**5 + 2.28E-05*AoA_deg**6 + -6.1E-07*AoA_deg**7 + 6.79E-09*AoA_deg**8
        else:
            SM = -0.086*AoA_deg + 2.73

        ## Corrective moment coefficient ##
        # Multiplying by stability because CG is where rotation is about and CP is where force is applied
            # SM = (CP - CG) / d
        C_raw = H * v_mag**2 * A * Cnalpha_rocket * AoA_eff * (SM * d) * rho / 2
        Ccm = Matrix([C_raw * sin(beta), -C_raw * cos(beta), 0])  # Corrective moment vector

        ## Propulsive Damping Moment Coefficient (Cdp) ##
        mdot = self.m_p / self.t_motor_burnout # kg/s, average mass flow rate during motor burn
        Cdp = mdot * (self.L_ne - CG)**2 # kg*m^2/s

        ## Aerodynamic Damping Moment Coefficient (Cda) ##
        Cda = H * (rho * v_mag * A / 2) * (Cnalpha_rocket * AoA_eff * (SM * d)**2)

        ## Damping Moment Coefficient (Cdm) ##
        Cdm = Cdp + Cda

        ## Moment due to fin cant angle ##
        gamma = Ct/Cr
        r_t = d/2
        tau = (s + r_t) / r_t
        # Roll forcing moment
        Y_MA = (s/3) * (1 + 2*gamma)/(1+gamma) # Spanwise location of fin aerodynamic center
        K_f = (1/pi**2) * \
            ((pi**2/4)*((tau+1)**2/tau**2) \
            + (pi*(tau**2+1)**2/(tau**2*(tau-1)**2))*asin((tau**2-1)/(tau**2+1)) \
            - (2*pi*(tau+1))/(tau*(tau-1)) \
            + ((tau**2+1)**2/(tau**2*(tau-1)**2))*asin((tau**2-1)/(tau**2+1))**2 \
            - (4*(tau+1)/(tau*(tau-1)))*asin((tau**2-1)/(tau**2+1)) \
            + (8/(tau-1)**2)*log((tau**2+1)/(2*tau)))
        M_f = K_f * (1/2 * rho * v_mag**2) * (N * (Y_MA + r_t) * Cnalpha_fin * delta * A) # Forcing roll moment due to fin cant angle delta

        # Roll damping moment
        trap_integral = s/12 * ((Cr + 3*Ct)*s**2 + 4*(Cr+2*Ct)*s*r_t + 6*(Cr + Ct)*r_t**2)
        C_ldw = 2 * N * Cnalpha_fin / (A * d**2) * cos(delta) * trap_integral
        K_d = 1 + ((tau-gamma)/tau - (1-gamma)/(tau-1)*ln(tau))/((tau+1)*(tau-gamma)/2 - (1-gamma)*(tau**3-1)/(3*(tau-1))) # Correction factor for conical fins
        M_d = K_d * (1/2 * rho * v_mag**2) * A * d * C_ldw * (w3 * d / (2 * v_mag)) # Damping roll moment
        
        M_fin = Matrix([0, 0, M_f - M_d])
        
        M1 = M_fin[0] + Ccm[0] - Cdm * w1
        M2 = M_fin[1] + Ccm[1] - Cdm * w2
        M3 = M_fin[2]

        ## Quaternion kinematics ##
        S = Matrix([[0, -w3, w2],
                    [w3, 0, -w1],
                    [-w2, w1, 0]])
        q_vec = Matrix([qw, qx, qy, qz])
        Omega = Matrix([
            [0, -w1, -w2, -w3],
            [w1, 0, w3, -w2],
            [w2, -w3, 0, w1],
            [w3, w2, -w1, 0]
        ])
        
        # -------------------------------------------- #

        ## Equations of motion ##
        w1dot = ((I2 - I3) * w2 * w3 + M1) / I1
        w2dot = ((I3 - I1) * w3 * w1 + M2) / I2
        w3dot = ((I1 - I2) * w1 * w2 + M3) / I3
        vdot = F/mass - S * v
        qdot = (Omega * q_vec) * Float(1/2)

        f = Matrix([
            [w1dot],
            [w2dot],
            [w3dot],
            [vdot[0]],
            [vdot[1]],
            [vdot[2]],
            [qdot[0]],
            [qdot[1]],
            [qdot[2]],
            [qdot[3]]
        ])

        vars = [w1, w2, w3, v1, v2, v3, qw, qx, qy, qz]
        self.vars = vars
        params = [I1, I2, I3, T1, T2, T3, mass, rho, d, g, CG, delta, Cnalpha_fin, Cnalpha_rocket, Cr, Ct, s, N]
        self.params = params

        if (not post_burnout):
            self.f_preburnout = f
        else:
            self.f_postburnout = f


    def setup_eom(self):
        """Setup the equations of motion by deriving pre- and post-burnout EOMs.

        Returns:
            None
        """
        self.deriveEOM(post_burnout=False)
        self.deriveEOM(post_burnout=True)


    def get_f(self, t: float, xhat: np.array):
        """Compute the A and B matrices at time t.
        Args:
            t (float): The time in seconds.
            xhat (np.array): The state estimation vector as a numpy array.
            u (np.array): The input vector as a numpy array.
        ## Sets:
            self.f_params (Matrix): The parameterized equations of motion with time-variant constants.
            self.f_subs (Matrix): The substituted equations of motion at a state.
        Returns:
            None
        """
        self.checkParamsSet()
        if self.f_preburnout is None or self.f_postburnout is None or self.vars is None:
            print("Equations of motion have not been derived yet. Call deriveEOM() first on pre- and post-burnout.")
            return None, None, None, None

        w1, w2, w3, v1, v2, v3, qw, qx, qy, qz = self.vars
        I1, I2, I3, T1, T2, T3, mass, rho, d, g, CG, delta, Cnalpha_fin, Cnalpha_rocket, Cr, Ct, s, N = self.params

        ## Get time varying constants ##
        constants = self.getThrust(t)
        mass_rocket = constants["mass"]
        inertia = constants["inertia"]
        x_CG = constants["CG"]
        thrust = constants["thrust"]

        # mass_rocket = self.m_0 - self.m_p / self.t_motor_burnout * t if t <= self.t_motor_burnout else self.m_f
        # I_long = self.I_0 - (self.I_0 - self.I_f) / self.t_motor_burnout * t if t <= self.t_motor_burnout else self.I_f
        # inertia = [I_long, I_long, self.I_3]
        # x_CG = self.x_CG_0 - (self.x_CG_0 - self.x_CG_f) / self.t_motor_burnout * t if t <= self.t_motor_burnout else self.x_CG_f

        params = {
            I1: Float(inertia[0]), # Ixx
            I2: Float(inertia[1]), # Iyy
            I3: Float(inertia[2]), # Izz
            T1: thrust[0],
            T2: thrust[1],
            T3: thrust[2],
            mass: Float(mass_rocket),
            CG: Float(x_CG), # m center of gravity
            rho: self.rho, # kg/m^3
            g: self.g, # m/s^2
            N: self.N, # number of fins
            d: self.d, # m rocket diameter
            Cr: self.Cr, # m fin root chord
            Ct: self.Ct, # m fin tip chord
            s: self.s, # m fin span
            Cnalpha_fin: self.Cnalpha_fin, # fin normal force coefficient derivative
            Cnalpha_rocket: self.Cnalpha_rocket, # rocket normal force coefficient derivative
            delta: rad(self.delta), # rad fin cant angle
            self.t_sym: Float(t)
        }

        ## Select pre- or post-burnout equations ##
        preburnout = t <= self.t_motor_burnout
        postburnout = t > self.t_motor_burnout
        f_params = None
        if preburnout:
            f_params = self.f_preburnout.subs(params)
        elif postburnout:
            f_params = self.f_postburnout.subs(params)

        ## Replace sqrt(v1^2 + v2^2) with a non-zero term to avoid NaNs in A matrix ##
        eps = Float(1e-3)  # Small term to avoid division by zero
        vxy = sqrt(v1**2 + v2**2 + eps**2)
        repl = {
            sqrt(v1**2 + v2**2): vxy,
            (v1**2 + v2**2)**(Float(1)/2): vxy
        }
        f_params = f_params.xreplace(repl)

        ## Substitute state variables ##
        m_e = {
            w1: xhat[0],
            w2: xhat[1],
            w3: xhat[2],
            v1: xhat[3],
            v2: xhat[4],
            v3: xhat[5],
            qw: xhat[6],
            qx: xhat[7],
            qy: xhat[8],
            qz: xhat[9],
        }

        f_subs = f_params.subs(m_e)

        self.f_params = f_params
        self.f_subs = f_subs
    

    def _f(self, t, x):
        # assumes you already called computeAB/computeC to refresh self.f_subs
        f = np.asarray(self.f_subs, float).reshape(-1)
        return f

    def _rk4_step(self, t, x):
        dt = self.dt
        k1 = self._f(t,       x)
        k2 = self._f(t+dt/2., x + dt*k1/2.)
        k3 = self._f(t+dt/2., x + dt*k2/2.)
        k4 = self._f(t+dt,    x + dt*k3)
        return x + (dt/6.)*(k1 + 2*k2 + 2*k3 + k4)
    
    def run_rk4(self, xhat: np.array):
        """Runge-Kutta 4th order integration of the state estimator recursively until the estimated apogee time is reached.
        Args:
            t (float): The current time in seconds.
            xhat (np.array): The current state estimate as a numpy array.
            u (np.array): The current input as a numpy array.
        Returns:
            np.array: The updated state estimate as a numpy array.
        """
        # loop, not recursion
        t = self.t0
        while xhat[5] >= 0 or t < self.t_estimated_apogee:
            self.get_f(t, xhat)  # refresh self.f_subs
            xhat = self._rk4_step(t, xhat)

            # Normalize quaternion
            qn = np.linalg.norm(xhat[6:10])
            xhat[6:10] = np.array([1.,0.,0.,0.]) if qn < 1e-12 else xhat[6:10]/qn

            # log + advance time
            self.states.append(xhat.copy())
            t += self.dt
            self.ts.append(t)
            print(f"t: {t:.3f}")


    def forward_euler(self, xhat: np.array):
        """Test the equations of motion by computing f_subs at the given state and input.

        Args:
            t (float): The current time in seconds.
            xhat (np.array): The current state estimate as a numpy array.
            u (np.array): The current input as a numpy array.
        """
        t = self.t0
        while t < self.t_estimated_apogee:
            print(f"t: {t:.3f}, xhat: {xhat}")

            self.get_f(t, xhat)
            f_subs = np.array(self.f_subs, dtype=float).reshape(-1)
            xhat = xhat + f_subs * self.dt
            xhat[6:10] /= np.linalg.norm(xhat[6:10])

            self.states.append(xhat)
            t += self.dt
            self.ts.append(t)
            if f_subs[5] < 0:
                print("Warning: Longitudinal velocity v3 is negative at time t =", t)
                print(f"t: {t:.3f}, xhat: {xhat}")

      
# For testing
def main():
    ## Define initial conditions ##
    xhat0 = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]) # Initial state estimate
    sampling_rate = 20.0  # Hz
    dt = 1.0 / sampling_rate

    dynamics = Dynamics(dt=dt, x0=xhat0)
    dynamics.setup_eom()
    dynamics.run_rk4(t=0.0, xhat=xhat0)


if __name__ == "__main__":
    main()
