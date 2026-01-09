
import jax
import jax.numpy as jnp
import argparse
from dataclasses import dataclass

jax.config.update("jax_enable_x64", True)

# ---------- Safe helpers ----------
def safe_exp(x, low=-700.0, high=700.0):
    return jnp.exp(jnp.clip(x, low, high))

def safe_div(n, d, eps=1e-300):
    return n / (d + eps)

# ---------- Parameters ----------
@dataclass
class Params:
    k: float = 1.0
    L: float = 1.0
    rc: float = 12.0
    vUV: float = 0.1
    vIR: float = 0.01
    Ny: int = 3001
    stretch: float = 0.35
    alpha: float = 4.0
    rtol: float = 1e-12
    atol: float = 1e-14
    max_substeps: int = 800
    clip: float = 1e12
    kappa5_sq: float = 1.0

    # Quantum corrections
    eps_JT: float = 0.0
    eps_Sch: float = 0.0
    ir_window_center_frac: float = 0.95
    ir_window_width_frac: float = 0.02
    sch_sat: float = 1.0

    # UV counter-terms
    delta_m2_UV: float = 0.0
    delta_lambda_UV: float = 0.0

    # Physical scales
    mH_bare: float = 125.0
    M5: float = 1.0e18

# ---------- Grid ----------
def make_stretched_grid(p: Params):
    Ymax = jnp.pi * jnp.array(p.rc, dtype=jnp.float64)
    s = jnp.array(p.stretch, dtype=jnp.float64)
    xi = jnp.linspace(0.0, 1.0, p.Ny, dtype=jnp.float64)
    f = ((1.0 - s) * xi + s * (xi ** p.alpha)) / ((1.0 - s) + s)
    y = Ymax * f
    return y, Ymax

def ir_window(y, Ymax, p: Params):
    y0 = p.ir_window_center_frac * Ymax
    w  = p.ir_window_width_frac * Ymax
    return 0.5 * (1.0 + jnp.tanh((y - y0) / w))

# ---------- Superpotential ----------
def W0(p: Params):
    return jnp.array(3.0 * p.k / p.kappa5_sq, dtype=jnp.float64)

def rhs_system(p: Params, y, U, Ymax, c2):
    phi, A = U
    dphi = 2.0 * c2 * phi
    dA = (p.kappa5_sq / 3.0) * (W0(p) + c2 * (phi**2))
    return jnp.array([dphi, dA])

def rhs_system_corrected(p: Params, y, U, Ymax, c2, vUV_eff):
    phi, A = U
    dphi = 2.0 * c2 * phi
    Aprime_base = (p.kappa5_sq / 3.0) * (W0(p) + c2 * (phi**2))
    wIR = ir_window(y, Ymax, p)
    dA_JT = p.eps_JT * wIR
    sat = p.sch_sat
    dA_S = p.eps_Sch * wIR * (Aprime_base**2) / (1.0 + (Aprime_base**2) / (sat**2))
    dA = Aprime_base + dA_JT + dA_S
    return jnp.array([dphi, dA])

# ---------- Integrator (RK45) ----------
def integrate_first_order_with_fun(fun, p: Params, y, U0, Ymax, c2):
    Ny = y.shape[0]
    rtol = jnp.array(p.rtol, dtype=jnp.float64)
    atol = jnp.array(p.atol, dtype=jnp.float64)
    max_substeps = p.max_substeps
    clip_val = jnp.array(p.clip, dtype=jnp.float64)

    def one_interval(U, i):
        yi = y[i]; yi1 = y[i+1]
        h_total = yi1 - yi
        state0 = (U, h_total, h_total, jnp.array(0, dtype=jnp.int32))

        def cond_fn(state):
            Uc, h_try, h_rem, sub = state
            return (h_rem > jnp.array(1e-18)) & (sub < jnp.array(max_substeps))

        def body_fn(state):
            Uc, h_try, h_rem, sub = state
            h_use = jnp.minimum(h_try, h_rem)
            y_local = yi + (h_total - h_rem)
            k1 = fun(p, y_local, Uc, Ymax, c2)
            k2 = fun(p, y_local + 0.25*h_use, Uc + h_use*(0.25*k1), Ymax, c2)
            k3 = fun(p, y_local + 3.0/8.0*h_use, Uc + h_use*(3.0/32.0*k1 + 9.0/32.0*k2), Ymax, c2)
            k4 = fun(p, y_local + 12.0/13.0*h_use, Uc + h_use*(1932.0/2197.0*k1 - 7200.0/2197.0*k2 + 7296.0/2197.0*k3), Ymax, c2)
            k5 = fun(p, y_local + h_use, Uc + h_use*(439.0/216.0*k1 - 8.0*k2 + 3680.0/513.0*k3 - 845.0/4104.0*k4), Ymax, c2)
            k6 = fun(p, y_local + 0.5*h_use, Uc + h_use*(-8.0/27.0*k1 + 2.0*k2 - 3544.0/2565.0*k3 + 1859.0/4104.0*k4 - 11.0/40.0*k5), Ymax, c2)
            U5 = Uc + h_use*(16.0/135.0*k1 + 6656.0/12825.0*k3 + 28561.0/56430.0*k4 - 9.0/50.0*k5 + 2.0/55.0*k6)
            U4 = Uc + h_use*(25.0/216.0*k1 + 1408.0/2565.0*k3 + 2197.0/4104.0*k4 - 1.0/5.0*k5)
            err = jnp.linalg.norm(U5 - U4)
            scale = atol + rtol * jnp.maximum(jnp.linalg.norm(Uc), jnp.linalg.norm(U5))
            accept = err <= scale
            Uc_new = jnp.where(accept, U5, Uc)
            Uc_new = jnp.clip(Uc_new, -clip_val, clip_val)
            h_rem_new = jnp.where(accept, h_rem - h_use, h_rem)
            h_try_new = jnp.where(accept, h_use*1.3, h_use*0.5)
            return (Uc_new, h_try_new, h_rem_new, sub+1)

        U_final, _, _, sub_final = jax.lax.while_loop(cond_fn, body_fn, state0)
        return U_final, (U_final, sub_final)

    carry, ys = jax.lax.scan(one_interval, U0, jnp.arange(Ny-1))
    states, sub_counts = ys
    Y = jnp.vstack([U0, states])
    return Y, sub_counts

# ---------- Analytics & audits ----------
def hierarchy_from_A(A: jnp.ndarray):
    logV = -4.0 * A
    V_eff = safe_exp(logV)
    R_eff = jnp.power(V_eff, 0.25)
    A_eff = jnp.power(R_eff, 3.0)
    dA = jnp.diff(A)
    eps_local = safe_exp(-4.0 * dA)
    eps_mean = jnp.mean(eps_local) if eps_local.size > 0 else jnp.nan
    return {"logV": logV, "V_eff": V_eff, "R_eff": R_eff, "A_eff": A_eff,
            "eps_local": eps_local, "eps_mean": float(eps_mean)}

def audit_volume_ratio_pointwise(V_eff: jnp.ndarray, A: jnp.ndarray, tol: float = 1e-6):
    if V_eff.size < 2 or A.size < 2:
        return {"pass": False, "max_error": float('inf'), "mean_error": float('inf')}
    obs = V_eff[1:] / safe_div(V_eff[:-1], 1.0)
    dA  = jnp.diff(A)
    exp_ratio = safe_exp(-4.0 * dA)
    err_vec = jnp.abs(obs - exp_ratio)
    return {"max_error": float(jnp.max(err_vec)),
            "mean_error": float(jnp.mean(err_vec)),
            "pass": bool(jnp.max(err_vec) < tol)}

def audit_local_consistency(eps_local: jnp.ndarray, tol_frac: float = 0.05):
    if eps_local.size == 0:
        return {"pass": True, "rel_std": 0.0}
    mu = jnp.mean(eps_local); sigma = jnp.std(eps_local)
    rel = safe_div(sigma, jnp.abs(mu))
    return {"pass": bool(rel < tol_frac), "rel_std": float(rel)}

def audit_monotone_A(A: jnp.ndarray):
    V_eff = safe_exp(-4.0 * A)
    diffs = jnp.diff(V_eff)
    nonincreasing = jnp.all(diffs <= 0.0) if diffs.size else True
    return {"nonincreasing": bool(nonincreasing)}

# ---------- Analytic checks ----------
def analytic_phi(y, vUV, c2):
    return vUV * jnp.exp(2.0 * c2 * y)

def analytic_A(y, p: Params, vUV, c2):
    term = (vUV**2 / 12.0) * (jnp.exp(4.0 * c2 * y) - 1.0)
    return p.k * y + term

def error_metrics(phi_num, A_num, y, p: Params, vUV, c2):
    phi_ref = analytic_phi(y, vUV, c2)
    A_ref   = analytic_A(y, p, vUV, c2)
    phi_err = jnp.max(jnp.abs(phi_num - phi_ref))
    A_err   = jnp.max(jnp.abs(A_num - A_ref))
    return {"phi_max_error": float(phi_err), "A_max_error": float(A_err)}

# ---------- Redshift dan Planck mass ----------
def redshift_outputs(A, p: Params):
    A_IR = A[-1]
    redshift = jnp.exp(-A_IR)
    M_UV = jnp.array(1.0e19)  # GeV
    M_IR = M_UV * redshift
    return {"A_IR": float(A_IR), "redshift": float(redshift), "M_IR": float(M_IR)}

def planck_mass_integral(A, y):
    integrand = jnp.exp(2.0 * A)
    dx = jnp.diff(y)
    avg = 0.5 * (integrand[:-1] + integrand[1:])
    val = jnp.sum(avg * dx)
    return float(val)

# ---------- Baseline solver (RS) ----------
def solve_superpotential_and_hierarchy(p: Params):
    y, Ymax = make_stretched_grid(p)
    c2 = jnp.log(jnp.array(p.vIR / p.vUV, dtype=jnp.float64)) / (2.0 * Ymax)
    U0 = jnp.array([p.vUV, 0.0], dtype=jnp.float64)
    Y, sub_counts = integrate_first_order_with_fun(rhs_system, p, y, U0, Ymax, c2)
    phi = Y[:, 0]; A = Y[:, 1]
    hdict = hierarchy_from_A(A)
    audits = {
        "volume_ratio_pointwise": audit_volume_ratio_pointwise(hdict["V_eff"], A),
        "local_consistency": audit_local_consistency(hdict["eps_local"]),
        "monotone_A": audit_monotone_A(A),
    }
    errors   = error_metrics(phi, A, y, p, p.vUV, c2)
    redshift = redshift_outputs(A, p)
    Mpl_eff  = planck_mass_integral(A, y)
    return {
        "y": y, "phi": phi, "A": A,
        "c2": float(c2),
        "hierarchy": hdict,
        "audits": audits,
        "errors": errors,
        "redshift": redshift,
        "Mpl_eff": Mpl_eff,
        "substeps_last": int(sub_counts[-1])
    }

# ---------- UV renormalized params ----------
def uv_renormalized_params(p: Params, Ymax):
    c2_base = jnp.log(jnp.array(p.vIR / p.vUV, dtype=jnp.float64)) / (2.0 * Ymax)
    a_m = jnp.array(0.5); a_l = jnp.array(0.1)
    b_m = jnp.array(0.25); b_l = jnp.array(0.05)
    delta_c2 = a_m * p.delta_m2_UV + a_l * p.delta_lambda_UV
    delta_vUV = b_m * p.delta_m2_UV + b_l * p.delta_lambda_UV
    vUV_eff = p.vUV * (1.0 + delta_vUV)
    c2_eff = c2_base + delta_c2
    return float(vUV_eff), float(c2_eff)

# ---------- Quantum-corrected solver ----------
def solve_superpotential_with_qcorr(p: Params):
    y, Ymax = make_stretched_grid(p)
    vUV_eff, c2_eff = uv_renormalized_params(p, Ymax)
    U0 = jnp.array([vUV_eff, 0.0], dtype=jnp.float64)
    fun = lambda p_, y_, U_, Ymax_, c2_: rhs_system_corrected(p_, y_, U_, Ymax_, c2_, vUV_eff)
    Y, sub_counts = integrate_first_order_with_fun(fun, p, y, U0, Ymax, c2_eff)
    phi = Y[:, 0]; A = Y[:, 1]
    hdict = hierarchy_from_A(A)
    audits = {
        "volume_ratio_pointwise": audit_volume_ratio_pointwise(hdict["V_eff"], A),
        "local_consistency": audit_local_consistency(hdict["eps_local"]),
        "monotone_A": audit_monotone_A(A),
    }
    errors = error_metrics(phi, A, y, p, vUV_eff, c2_eff)
    redshift = redshift_outputs(A, p)
    Mpl_eff = planck_mass_integral(A, y)

    mH_eff = p.mH_bare * redshift["redshift"]
    G_IR = safe_div(1.0, Mpl_eff) * jnp.exp(2.0 * A[-1])
    obs = {
        "mH_eff": float(mH_eff),
        "G_IR_eff": float(G_IR),
        "A_IR": redshift["A_IR"],
        "redshift": redshift["redshift"],
        "Mpl_eff": Mpl_eff,
        "vUV_eff": vUV_eff,
        "c2_eff": c2_eff
    }
    return {
        "y": y, "phi": phi, "A": A,
        "params_eff": {"vUV_eff": vUV_eff, "c2_eff": c2_eff},
        "hierarchy": hdict,
        "audits": audits,
        "errors": errors,
        "redshift": redshift,
        "Mpl_eff": Mpl_eff,
        "observables": obs,
        "substeps_last": int(sub_counts[-1])
    }

# ---------- RS vs quantum-corrected comparison ----------
def solve_RS_baseline(p: Params):
    p0 = Params(**{**p.__dict__, "eps_JT": 0.0, "eps_Sch": 0.0,
                   "delta_m2_UV": 0.0, "delta_lambda_UV": 0.0})
    return solve_superpotential_and_hierarchy(p0)

def compare_qcorr_vs_RS(p: Params):
    base = solve_RS_baseline(p)
    corr = solve_superpotential_with_qcorr(p)

    def pct(a, b):
        denom = b if jnp.abs(b) > 1e-30 else 1.0
        return float(100.0 * (a - b) / denom)

    Omega_IR_corr = float(jnp.exp(-4.0 * corr["A"][-1]))
    Omega_IR_base = float(jnp.exp(-4.0 * base["A"][-1]))

    report = {
        "A_IR_dev": corr["redshift"]["A_IR"] - base["redshift"]["A_IR"],
        "redshift_pct_dev": pct(corr["redshift"]["redshift"], base["redshift"]["redshift"]),
        "c2_eff_dev": corr["observables"]["c2_eff"] - base["c2"],
        "ΔmH_eff_GeV": corr["observables"]["mH_eff"] - (p.mH_bare * base["redshift"]["redshift"]),
        "G_IR_eff_pct_dev": pct(corr["observables"]["G_IR_eff"],
                                (1.0 / base["Mpl_eff"]) * float(jnp.exp(2.0 * base["A"][-1]))),
        "Ω_IR_corr": Omega_IR_corr,
        "Ω_IR_base": Omega_IR_base
    }

    print("\n=== Quantum corrections vs RS baseline ===")
    print(f"A_IR (corr - base): {report['A_IR_dev']:.6f}")
    print(f"Redshift deviation (%): {report['redshift_pct_dev']:.3f}")
    print(f"c2_eff - c2_base: {report['c2_eff_dev']:.6e}")
    print(f"Δ m_H^eff (GeV): {report['ΔmH_eff_GeV']:.6f}")
    print(f"G_IR_eff deviation (%): {report['G_IR_eff_pct_dev']:.3f}")
    print(f"Ω_IR (proxy) corr/base: {report['Ω_IR_corr']:.3e} / {report['Ω_IR_base']:.3e}")

    return {"baseline": base, "corrected": corr, "report": report}

# ---------- Convergence & sensitivity ----------
def run_study(p: Params):
    base = solve_superpotential_and_hierarchy(p)
    print("\n=== Convergence study ===")
    for Ny in [3001, 6001, 12001]:
        p2 = Params(**{**p.__dict__, "Ny": Ny})
        out2 = solve_superpotential_and_hierarchy(p2)
        n = min(base["phi"].size, out2["phi"].size)
        dphi = float(jnp.max(jnp.abs(out2["phi"][:n] - base["phi"][:n])))
        dA   = float(jnp.max(jnp.abs(out2["A"][:n]   - base["A"][:n])))
        print(f"Ny={Ny}: ΔΦ={dphi:.2e}, ΔA={dA:.2e}")

    print("\n=== Sensitivity study (UV counter-terms) ===")
    for dv in [0.99, 1.01]:
        p2 = Params(**{**p.__dict__, "vUV": p.vUV * dv})
        out2 = solve_superpotential_and_hierarchy(p2)
        print(f"vUV×{dv}: A_IR={out2['redshift']['A_IR']:.2f}, redshift={out2['redshift']['redshift']:.2e}")

    for rc in [8, 10, 12, 14]:
        p2 = Params(**{**p.__dict__, "rc": rc})
        out2 = solve_superpotential_and_hierarchy(p2)
        print(f"rc={rc}: A_IR={out2['redshift']['A_IR']:.2f}, redshift={out2['redshift']['redshift']:.2e}")

    print("\n=== Planck mass integral ===")
    print("Effective M_Pl^2 integral:", base["Mpl_eff"])
    return base

# ---------- Main with CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Ny", type=int, default=3001)
    parser.add_argument("--rc", type=float, default=12.0)
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--atol", type=float, default=1e-14)
    parser.add_argument("--study", action="store_true")
    parser.add_argument("--compare", action="store_true")

    # Quantum correction args
    parser.add_argument("--epsJT", type=float, default=0.0)
    parser.add_argument("--epsSch", type=float, default=0.0)
    parser.add_argument("--dm2UV", type=float, default=0.0)
    parser.add_argument("--dlUV", type=float, default=0.0)

    args, _ = parser.parse_known_args()

    p = Params(
        Ny=args.Ny,
        rc=args.rc,
        rtol=args.rtol,
        atol=args.atol,
        eps_JT=args.epsJT,
        eps_Sch=args.epsSch,
        delta_m2_UV=args.dm2UV,
        delta_lambda_UV=args.dlUV
    )

    if args.compare:
        out = compare_qcorr_vs_RS(p)
    elif args.study:
        out = run_study(p)
    else:
        use_qcorr = (p.eps_JT != 0.0) or (p.eps_Sch != 0.0) or (p.delta_m2_UV != 0.0) or (p.delta_lambda_UV != 0.0)
        out = solve_superpotential_with_qcorr(p) if use_qcorr else solve_superpotential_and_hierarchy(p)

    print("\n=== Results (summary) ===")
    if isinstance(out, dict) and "observables" in out:
        print("Quantum-corrected run:")
        print("A_IR:", out["observables"]["A_IR"])
        print("redshift:", out["observables"]["redshift"])
        print("mH_eff (GeV):", out["observables"]["mH_eff"])
        print("G_IR_eff:", out["observables"]["G_IR_eff"])
    elif isinstance(out, dict) and "report" in out:
        print("Comparison report printed above.")
    else:
        print("Baseline RS run printed above.")
        print("A_IR:", out["redshift"]["A_IR"])
        print("redshift:", out["redshift"]["redshift"])
        print("Mpl_eff:", out["Mpl_eff"])
        print("c2:", out["c2"])
        print("phi_err_max:", out["errors"]["phi_max_error"])
        print("A_err_max:", out["errors"]["A_max_error"])
