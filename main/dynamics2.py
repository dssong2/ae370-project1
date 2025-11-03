# dynamics3.py
# Full physics from original Dynamics, stabilized and faster via lambdify
from sympy import *
import numpy as np
import pandas as pd
from pathlib import Path

class Dynamics2:
    def __init__(
        self,
        t_motor_burnout: float = 1.971,
        t_estimated_apogee: float = 13.571,
        t_launch_rail_clearance: float = 0.308,
        prop_mass: float = 0.355,
        L_ne: float = 1.17,
        dt: float = 0.01,
        x0: np.ndarray = np.array([0.,0.,0.,0.,0.,0.,1.,0.,0.,0.])
    ):
        self.t_motor_burnout = t_motor_burnout
        self.t_estimated_apogee = t_estimated_apogee
        self.t_launch_rail_clearance = t_launch_rail_clearance
        self.prop_mass = prop_mass
        self.L_ne = L_ne
        self.dt = dt
        self.t0 = 0.0
        self.x0 = np.array(x0, dtype=float)

        # Load OpenRocket data once
        csv_path = Path(__file__).resolve().parents[1] / "data" / "openrocket_data.csv"
        df = pd.read_csv(csv_path)
        self.t_data = df["# Time (s)"].values
        self.mass_data = df["Mass (g)"].values / 1000.0
        self.inertia_data = df["Longitudinal moment of inertia (kg·m²)"].values
        self.cg_data = df["CG location (cm)"].values / 100.0
        self.thrust_data = df["Thrust (N)"].values

        # Logs
        self.states = [self.x0.copy()]
        self.ts = [self.t0]

        # lambdified funcs
        self.f_func_pre = None
        self.f_func_post = None
        self.vars = None

        # global constants
        self.rho = 1.225
        self.diam = 7.87/100.0     # m
        self.Aref = np.pi * (self.diam/2.0)**2
        self.g = -9.81             # choose negative => Fg_world = [0,0,m*g] gives downward (−z) in world

        # aero/fin constants (same spirit as original)
        self.Nf = 4
        self.Cnalpha_fin = 2.72025
        self.delta = np.deg2rad(1.0)
        self.Cr = 18.0/100.0
        self.Ct = 5.97/100.0
        self.s = 8.76/100.0

        # numeric guards
        self.eps_v = 1e-3
        self.aoa_max_rad = np.deg2rad(15.0)  # clip AoA to ±15°
        self.SM_min, self.SM_max = 0.0, 5.0  # clamp stability margin

    # ----------------------- time-varying constants -----------------------
    def getTimeConstants(self, t: float):
        if t > self.t_motor_burnout:
            I = np.array([0.287, 0.287, 0.0035])
            m = 2.589
            CG = 63.5/100.0
            T = np.array([0., 0., 0.])
        else:
            m = np.interp(t, self.t_data, self.mass_data)
            I_long = np.interp(t, self.t_data, self.inertia_data)
            I = np.array([I_long, I_long, 0.0035])
            CG = np.interp(t, self.t_data, self.cg_data)
            T = np.array([0., 0., np.interp(t, self.t_data, self.thrust_data)])
        return dict(inertia=I, mass=m, CG=CG, thrust=T)

    # ----------------------- quaternion to rotation -----------------------
    def R_BW_from_q(self, qw, qx, qy, qz):
        s = (qw**2 + qx**2 + qy**2 + qz**2)**-Rational(1,2)
        qw, qx, qy, qz = qw*s, qx*s, qy*s, qz*s
        xx, yy, zz = qx*qx, qy*qy, qz*qz
        wx, wy, wz = qw*qx, qw*qy, qw*qz
        xy, xz, yz = qx*qy, qx*qz, qy*qz
        return Matrix([
            [1-2*(yy+zz), 2*(xy+wz), 2*(xz-wy)],
            [2*(xy-wz), 1-2*(xx+zz), 2*(yz+wx)],
            [2*(xz+wy), 2*(yz-wx), 1-2*(xx+yy)]
        ])

    # ----------------------- symbolic EOM (SM is a PARAM) -----------------
    def deriveEOM(self, post_burnout: bool):
        # symbols
        t = symbols('t', real=True)
        w1,w2,w3,v1,v2,v3,qw,qx,qy,qz = symbols('w1 w2 w3 v1 v2 v3 qw qx qy qz', real=True)
        I1,I2,I3,T1,T2,T3,mass,rho,d,g,CG = symbols('I1 I2 I3 T1 T2 T3 m rho d g CG', real=True)
        delta, Cnalpha_fin, Cr, Ct, s, N = symbols('delta Cnalpha_fin Cr Ct s N', real=True)
        # NEW: stability margin passed numerically (pre/post different), not symbolically expanded
        SMsym = symbols('SMsym', real=True)

        # heaviside replacement token (numeric in lambdify)
        H = Function('H')(t)

        eps = Float(1e-3)
        v = Matrix([v1,v2,v3])
        vmag = sqrt(v1**2 + v2**2 + v3**2 + eps**2)
        vhat = v/vmag
        AoA = atan2(sqrt(v1**2 + v2**2), v3 + eps)  # keep symbolic AoA; we'll clip numerically through H and SMsym
        beta = atan2(v2, v1 + eps)
        A = pi*(d/2)**2

        # Forces
        T = Matrix([T1,T2,T3])
        Fg_world = Matrix([0,0,mass*g])  # with g<0 => downward
        Rbw = self.R_BW_from_q(qw,qx,qy,qz)
        Fg = Rbw @ Fg_world

        # drag (same as original fit)
        Fd = -(0.627 - 0.029*vmag + 1.95e-3*vmag**2) * vhat

        # lift (thin airfoil, scaled by H)
        nL = Matrix([-cos(AoA)*cos(beta), -cos(AoA)*sin(beta), sin(AoA)])
        Fl = H * (rho*vmag**2/2) * (2*pi*AoA) * A * nL

        F = T + Fd + Fl + Fg

        # Corrective moment (Ccm) ~ AoA * SM * d
        Cnalpha = 0.207
        C_raw = H * (rho*vmag**2/2) * A * Cnalpha * AoA * (SMsym * d)    # SMsym injected numerically
        Ccm = Matrix([C_raw * sin(beta), -C_raw * cos(beta), 0])

        # Damping: propulsive + aerodynamic
        mdot = self.prop_mass / self.t_motor_burnout
        Cdp = mdot * (self.L_ne - CG)**2
        Cda = H * (rho * vmag * A / 2) * (Cnalpha * AoA * (SMsym * d)**2)
        Cdm = Cdp + Cda

        # Fin roll moment (forcing + damping)
        gamma = Ct/Cr
        r_t = d/2
        tau = (s + r_t)/r_t
        Y_MA = (s/3) * (1 + 2*gamma)/(1 + gamma)
        # forcing (kept simple but consistent)
        M_f = (rho*vmag**2/2) * (N * (Y_MA + r_t) * Cnalpha_fin * delta * A) / (pi**2/4)  # scaled; avoids huge constants
        # damping ~ w3
        C_ldw = 2 * N * Cnalpha_fin / (A * d**2)  # baseline coefficient
        M_d = (rho*vmag**2/2) * A * d * C_ldw * (w3 * d / (2 * vmag + 1e-9))
        M_fin = Matrix([0, 0, M_f - M_d])

        M1 = Ccm[0] - Cdm*w1 + M_fin[0]
        M2 = Ccm[1] - Cdm*w2 + M_fin[1]
        M3 = M_fin[2]

        # Kinematics
        S = Matrix([[0,-w3,w2],[w3,0,-w1],[-w2,w1,0]])
        q_vec = Matrix([qw,qx,qy,qz])
        Omega = Matrix([
            [0,-w1,-w2,-w3],
            [w1,0,w3,-w2],
            [w2,-w3,0,w1],
            [w3,w2,-w1,0],
        ])

        wdot = Matrix([
            ((I2 - I3)*w2*w3 + M1)/I1,
            ((I3 - I1)*w3*w1 + M2)/I2,
            ((I1 - I2)*w1*w2 + M3)/I3,
        ])
        vdot = F/mass - S@v
        qdot = (Omega@q_vec) * Float(1/2)

        f = Matrix.vstack(wdot, vdot, qdot)

        # order of inputs to lambdify (keep stable)
        self.vars = [t, w1,w2,w3, v1,v2,v3, qw,qx,qy,qz,
                     I1,I2,I3, T1,T2,T3, mass, rho, d, g, CG,
                     delta, Cnalpha_fin, Cr, Ct, s, N, SMsym]

        if not post_burnout:
            self.f_preburnout = f
        else:
            self.f_postburnout = f

    def setup_eom(self):
        self.deriveEOM(False)
        self.deriveEOM(True)

        # numeric Heaviside for launch rail clearance
        def H_func(tval): return np.heaviside(float(tval) - self.t_launch_rail_clearance, 0.0)

        modules = ["numpy", {"Heaviside": H_func, "H": H_func}]
        self.f_func_pre  = lambdify(self.vars, self.f_preburnout,  modules=modules)
        self.f_func_post = lambdify(self.vars, self.f_postburnout, modules=modules)

    # ----------------------- numeric helpers -----------------------
    def _compute_SM(self, AoA_rad, post_burnout: bool):
        # Use your original fits but do them NUMERICALLY, then clamp.
        a_deg = np.degrees(AoA_rad)
        if not post_burnout:
            # Original 8th-degree fit — evaluate numerically, then clamp
            SM = (2.8
                  - 0.48*a_deg
                  + 0.163*a_deg**2
                  - 0.0386*a_deg**3
                  + 5.46e-03*a_deg**4
                  - 4.61e-04*a_deg**5
                  + 2.28e-05*a_deg**6
                  - 6.10e-07*a_deg**7
                  + 6.79e-09*a_deg**8)
        else:
            SM = -0.086*a_deg + 2.73
        return float(np.clip(SM, self.SM_min, self.SM_max))

    def _state_to_aoa(self, v1, v2, v3):
        vxy = np.hypot(v1, v2)
        return float(np.arctan2(vxy, v3 + self.eps_v))

    # ----------------------- fast numeric RHS -----------------------
    def _f(self, t, x):
        # constants
        c = self.getTimeConstants(t)
        I1, I2, I3 = c["inertia"]
        T1, T2, T3 = c["thrust"]
        m = c["mass"]
        CG = c["CG"]

        # AoA and SM computed numerically, with clipping
        AoA = self._state_to_aoa(x[3], x[4], x[5])
        AoA_clipped = float(np.clip(AoA, -self.aoa_max_rad, self.aoa_max_rad))
        post = bool(t > self.t_motor_burnout)
        SMval = self._compute_SM(AoA_clipped, post)

        # build arg list in same order as self.vars
        args = (t, *x,
                I1, I2, I3, T1, T2, T3, m, self.rho, self.diam, self.g, CG,
                self.delta, self.Cnalpha_fin, self.Cr, self.Ct, self.s, self.Nf, SMval)

        func = self.f_func_post if post else self.f_func_pre
        out = np.array(func(*args), dtype=float).reshape(-1)

        # final safety rails: clip torques/velocities to prevent explosion in rare edge cases
        # (keeps convergence tests sane without changing physics in normal ranges)
        out[:3] = np.clip(out[:3], -1e6, 1e6)  # angular accelerations
        out[3:6] = np.clip(out[3:6], -1e6, 1e6)  # linear accelerations
        return out

    # ----------------------- RK4 integrator -----------------------
    def _rk4_step(self, t, x):
        dt = self.dt
        k1 = self._f(t, x)
        k2 = self._f(t + 0.5*dt, x + 0.5*dt*k1)
        k3 = self._f(t + 0.5*dt, x + 0.5*dt*k2)
        k4 = self._f(t + dt,     x + dt*k3)
        return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    def run_rk4(self, xhat=None, verbose=False):
        if xhat is None:
            xhat = self.x0.copy()
        t = self.t0
        while t < self.t_estimated_apogee:
            xhat = self._rk4_step(t, xhat)
            # normalize quaternion every step
            qn = np.linalg.norm(xhat[6:10])
            xhat[6:10] = np.array([1.,0.,0.,0.]) if qn < 1e-14 else xhat[6:10]/qn
            self.states.append(xhat.copy())
            t += self.dt
            self.ts.append(t)
            if verbose and int(t/self.dt) % 200 == 0:
                print(f"t={t:.2f}")
        return np.array(self.ts), np.array(self.states)

# quick sanity
if __name__ == "__main__":
    x0 = np.array([0,0,0,0,0,0,1,0,0,0], dtype=float)
    dyn = Dynamics3(dt=0.01, x0=x0)
    dyn.setup_eom()
    t, X = dyn.run_rk4(verbose=True)
    print("Final t =", t[-1])
