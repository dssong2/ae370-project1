from sympy import *
import numpy as np
import pandas as pd
from pathlib import Path

class Dynamics:
    def __init__(
            self,
            t_motor_burnout: float = 1.971,
            t_estimated_apogee: float = 13.571,
            t_launch_rail_clearance: float = 0.164,
            prop_mass: float = 0.355, # kg
            L_ne: float = 1.17, # m
            dt: float = 0.01,
            x0: np.ndarray = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), # Initial state
        ):
        """Initialize the Controls class. Rocket body axis is aligned with y-axis.

        Args:
            t_motor_burnout (float): Time until motor burnout in seconds. Defaults to 1.971.
            t_estimated_apogee (float): Estimated time until apogee in seconds. Defaults to 13.571.
            t_launch_rail_clearance (float): Time until launch rail clearance in seconds. Defaults to 0.164.
            prop_mass (float): Propellant mass in kg. Defaults to 0.355.
            L_ne (float): Distance the nozzle is from the tip of the nose cone in meters. Defaults to 1.17.
            dt (float): Time step for simulation in seconds. Defaults to 0.01.
        """

        self.t_motor_burnout = t_motor_burnout # seconds
        self.t_estimated_apogee = t_estimated_apogee # seconds
        self.t_launch_rail_clearance = t_launch_rail_clearance # seconds
        self.prop_mass = prop_mass # kg
        self.L_ne = L_ne # m
        self.csv_path = self.csv_path = (
            Path(__file__).resolve().parents[1] / "data" / "openrocket_data.csv"
        )
        self.f_preburnout : Matrix = None
        self.f_postburnout : Matrix = None
        self.vars : list = None
        self.f_params : Matrix = None
        self.f_subs : Matrix = None
        self.dt = dt
        self.x0 = np.array(x0, dtype=float) if x0 is not None else None
        self.t0 = 0.0
        self.t_sym : Symbol = None

        # Logging
        self.states = [self.x0]
        self.ts = [self.t0]


    def setRocketParams(self, t_motor_burnout: float, t_estimated_apogee: float, t_launch_rail_clearance: float, prop_mass: float):
        """Set the rocket parameters.

        Args:
            t_motor_burnout (float): Time until motor burnout in seconds.
            t_estimated_apogee (float): Estimated time until apogee in seconds.
            t_launch_rail_clearance (float): Time until launch rail clearance in seconds.
            prop_mass (float): Propellant mass in kg.
        """
        self.t_motor_burnout = t_motor_burnout
        self.t_estimated_apogee = t_estimated_apogee
        self.t_launch_rail_clearance = t_launch_rail_clearance
        self.prop_mass = prop_mass

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


    def getTimeConstants(self, t: float):
        """Get the constants for the rocket at time t.

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


    def euler_to_quat_xyz(self, theta, phi, psi, degrees=False):
        """
        Convert Euler angles to a quaternion using intrinsic XYZ:
            - theta: rotation about x (pitch)
            - phi:   rotation about y (yaw)
            - psi:   rotation about z (roll)
        Convention: R = Rz(psi) @ Ry(phi) @ Rx(theta)
        Quaternion is returned as [w, x, y, z].

        Args:
            theta, phi, psi : floats (radians by default; set degrees=True if in deg)
            degrees         : if True, inputs are in degrees

        Returns:
            np.ndarray shape (4,) -> [w, x, y, z]
        """
        if degrees:
            theta, phi, psi = np.radians([theta, phi, psi])

        # half-angles
        cth, sth = np.cos(theta/2.0), np.sin(theta/2.0)
        cph, sph = np.cos(phi/2.0),   np.sin(phi/2.0)
        cps, sps = np.cos(psi/2.0),   np.sin(psi/2.0)

        # intrinsic XYZ closed form (q = qz * qy * qx), scalar-first
        qw =  cph*cps*cth + sph*sps*sth
        qx = -sph*sps*cth + sth*cph*cps
        qy =  sph*cps*cth + sps*sth*cph
        qz = -sph*sth*cps + sps*cph*cth

        q = np.array([qw, qx, qy, qz], dtype=float)
        # normalize to guard against numerical drift
        q /= np.linalg.norm(q)
        return q
    

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
        """Get the equations of motion for the rocket, derive the A and B matrices at time t.

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
        - To derive the full set of equations of motion, call deriveEOM() twice: once with post_burnout=False and once with post_burnout=True

        Args:
            post_burnout (bool): Whether the rocket is in the post-burnout phase.
        Returns:
            tuple: A tuple containing the A and B Numpy arrays evaluated at the operating state xhat and input u.
        """
        w1, w2, w3, v1, v2 = symbols('w_1 w_2 w_3 v_1 v_2', real = True) # Angular and linear velocities
        v3 = symbols('v_3', real = True, positive = True) # Longitudinal velocity, assumed positive during flight
        qw, qx, qy, qz = symbols('q_w q_x q_y q_z', real = True) # Quaternion components
        I1, I2, I3 = symbols('I_1 I_2 I_3', real = True) # Moments of inertia
        M1, M2, M3 = symbols('M_1 M_2 M_3', real = True) # Moments
        T1, T2, T3 = symbols('T_1 T_2 T_3', real = True) # Thrusts
        mass, rho, A, g, CG = symbols('m rho A g CG', real = True) # Mass, air density, reference area, gravity, center of gravity
        delta1 = symbols('delta_1', real = True) # Aileron angle

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

        ## Thrust ##
        T : Matrix = Matrix([T1, T2, T3])  # Thrust vector, T1 and T2 are assumed 0

        ## Gravity ##
        Fg_world = Matrix([0.0, 0.0, -mass * g])
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

        ## Cnalpha ##
        # TODO: Show how this is calculated from OpenRocket data
        Cnalpha = 0.207  # Linear assumption of Cn vs AoA slope from OpenRocket data (fitted to quadratic, minimal x^2 coefficient)

        ## Stability Margin ##
        # TODO: Potential to implement our own polynomial fitting function instead of using hardcoded coefficients from Google Sheets
        AoA_deg = AoA_eff * 180 / pi # Convert AoA to degrees for polynomial fit
        SM = 0
        if not post_burnout:
            SM = 2.8 + -0.48*AoA_deg + 0.163*AoA_deg**2 + -0.0386*AoA_deg**3 + 5.46E-03*AoA_deg**4 + -4.61E-04*AoA_deg**5 + 2.28E-05*AoA_deg**6 + -6.1E-07*AoA_deg**7 + 6.79E-09*AoA_deg**8
        else:
            SM = -0.086*AoA_deg + 2.73

        ## Rocket diameter ##
        d = Float(7.87)/100 # m

        ## Corrective moment coefficient ##
        # Multiplying by stability because CG is where rotation is about and CP is where force is applied
            # SM = (CP - CG) / d
        # TODO: Show how this is calculated (Apogee Rocketry report reference)
        C_raw = H * v_mag**2 * A * Cnalpha * AoA_eff * (SM * d) * rho / 2 # See if it's Cnalpha or Cn, Cn = Cnalpha * AoA_eff
        Ccm = Matrix([C_raw * sin(beta), -C_raw * cos(beta), 0])  # Corrective moment vector

        ## Propulsive Damping Moment Coefficient (Cdp) ##
        # TODO: Show how this is calculated (Apogee Rocketry report reference)
        mdot = self.prop_mass / self.t_motor_burnout # kg/s, average mass flow rate during motor burn
        Cdp = mdot * (self.L_ne - CG)**2 # kg*m^2/s

        ## Aerodynamic Damping Moment Coefficient (Cda) ##
        # TODO: Show how this is calculated (Apogee Rocketry report reference)
        Cda = H * (rho * v_mag * A / 2) * (Cnalpha * AoA_eff * (SM * d)**2)

        ## Damping Moment Coefficient (Cdm) ##
        Cdm = Cdp + Cda

        ## Moment due to aileron deflection ##
        # Fin misalignment moment, remove for 0 roll (ideal rocket flight)
        M_fin = (Float(1)/2 * rho * v_mag**2) * Matrix([0, 0, 1e-8])
        M1 = M_fin[0] + Ccm[0] - Cdm * w1
        M2 = M_fin[1] + Ccm[1] - Cdm * w2
        M3 = M_fin[2] + Ccm[2]

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

        vars = [w1, w2, w3, v1, v2, v3, qw, qx, qy, qz, delta1, I1, I2, I3, T1, T2, T3, mass, rho, A, g, CG]
        self.vars = vars

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
        if self.f_preburnout is None or self.f_postburnout is None or self.vars is None:
            print("Equations of motion have not been derived yet. Call deriveEOM() first on pre- and post-burnout.")
            return None, None, None, None
        
        w1, w2, w3, v1, v2, v3, qw, qx, qy, qz, delta1, I1, I2, I3, T1, T2, T3, mass, rho, A, g, CG = self.vars

        ## Get time varying constants ##
        constants = self.getTimeConstants(t)
        mass_rocket = constants["mass"]
        inertia = constants["inertia"]
        CoG = constants["CG"]
        thrust = constants["thrust"]

        params = {
            I1: Float(inertia[0]), # Ixx
            I2: Float(inertia[1]), # Iyy
            I3: Float(inertia[2]), # Izz
            T1: thrust[0],
            T2: thrust[1],
            T3: thrust[2],
            mass: Float(mass_rocket),
            rho: Float(1.225), # kg/m^3 temp constant rho
            A: pi * Float((7.87/100/2)**2), # m^2 reference area
            g: Float(9.81), # m/s^2
            CG: Float(CoG), # m center of gravity
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

        ## NOTE: Not finding equilibrium states, using trajectory/operating-point linearization
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


    def get_thrust_accel(self, t: float):
        """Get the thrust acceleration at time t.

        Args:
            t (float): The time in seconds.

        Returns:
            np.array: The thrust acceleration vector as a numpy array.
        """
        thrust = self.getTimeConstants(t)["thrust"]
        m = self.getTimeConstants(t)["mass"]
        a_thrust = np.zeros(10)
        a_thrust[3] = thrust[0] / m
        a_thrust[4] = thrust[1] / m
        a_thrust[5] = thrust[2] / m
        return a_thrust


    def get_gravity_accel(self, xhat: np.array):
        """Get the gravity acceleration in body frame at time t.

        Args:
            xhat (np.array): The current state estimate as a numpy array.

        Returns:
            np.array: The gravity acceleration vector as a numpy array.
        """
        g = np.array([0.0, 0.0, -9.81])
        qw, qx, qy, qz = xhat[6], xhat[7], xhat[8], xhat[9]
        R_world_to_body = np.array(self.R_BW_from_q(qw, qx, qy, qz)).astype(np.float64)
        g_body = R_world_to_body @ g
        a_gravity = np.zeros(10)
        a_gravity[3:6] = g_body
        return a_gravity
    

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
        while t < self.t_estimated_apogee:
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


    def test_eom(self, xhat: np.array):
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
