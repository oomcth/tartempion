import numpy as np
import proxsuite
import pickle
import scipy.optimize as opt

np.set_printoptions(linewidth=np.inf)


def analyze_qp_convergence(Q, p, bounds=None, x0=None, verbose=True):
    """
    Analyse complète d'un problème QP pour diagnostiquer les problèmes de convergence

    Problème: min (1/2) x^T Q x + p^T x  s.t. -1 <= x <= 1

    Args:
        Q: Matrice hessienne (n x n)
        p: Vecteur gradient linéaire (n,)
        bounds: Bornes (par défaut [-1, 1] pour chaque variable)
        x0: Point de départ (optionnel)
        verbose: Affichage détaillé

    Returns:
        dict: Dictionnaire complet d'analyse
    """

    Q = np.array(Q)
    p = np.array(p)
    n = len(p)

    if bounds is None:
        bounds = [(-1, 1)] * n

    if x0 is None:
        x0 = np.zeros(n)

    results = {
        "problem_size": n,
        "hessian_analysis": {},
        "constraints_analysis": {},
        "numerical_analysis": {},
        "convergence_diagnosis": {},
        "recommendations": [],
    }

    # ========================================
    # 1. ANALYSE DE LA MATRICE HESSIENNE
    # ========================================
    if verbose:
        print("=" * 60)
        print("ANALYSE DE LA MATRICE HESSIENNE Q")
        print("=" * 60)

    # Propriétés de base
    is_symmetric = np.allclose(Q, Q.T)
    frobenius_norm = np.linalg.norm(Q, "fro")

    results["hessian_analysis"]["is_symmetric"] = is_symmetric
    results["hessian_analysis"]["frobenius_norm"] = frobenius_norm

    if verbose:
        print(f"Dimension: {n}x{n}")
        print(f"Symétrique: {is_symmetric}")
        print(f"Norme de Frobenius: {frobenius_norm:.6e}")

    if not is_symmetric:
        results["recommendations"].append(
            "CRITIQUE: Q n'est pas symétrique. Utilisez Q_sym = (Q + Q^T)/2"
        )
        Q = (Q + Q.T) / 2  # Correction pour l'analyse

    # Analyse spectrale
    try:
        eigenvals = np.linalg.eigvals(Q)
        lambda_min = np.min(eigenvals)
        lambda_max = np.max(eigenvals)

        results["hessian_analysis"]["eigenvalues"] = eigenvals
        results["hessian_analysis"]["lambda_min"] = lambda_min
        results["hessian_analysis"]["lambda_max"] = lambda_max

        if verbose:
            print(f"λ_min: {lambda_min:.6e}")
            print(f"λ_max: {lambda_max:.6e}")

        # Conditionnement
        if abs(lambda_min) > 1e-12:
            condition_number = lambda_max / abs(lambda_min)
            results["hessian_analysis"]["condition_number"] = condition_number
            if verbose:
                print(f"Nombre de condition: {condition_number:.6e}")
        else:
            results["hessian_analysis"]["condition_number"] = np.inf
            if verbose:
                print("Nombre de condition: ∞ (matrice singulière)")

        # Classification
        tol = 1e-12
        if lambda_min > tol:
            definiteness = "définie positive"
            convex = True
        elif lambda_min > -tol:
            definiteness = "semi-définie positive"
            convex = True
        elif lambda_max < -tol:
            definiteness = "définie négative"
            convex = False
        elif lambda_max < tol:
            definiteness = "semi-définie négative"
            convex = False
        else:
            definiteness = "indéfinie"
            convex = False

        results["hessian_analysis"]["definiteness"] = definiteness
        results["hessian_analysis"]["is_convex"] = convex

        if verbose:
            print(f"Classification: {definiteness}")
            print(f"Problème convexe: {convex}")

    except Exception as e:
        if verbose:
            print(f"Erreur dans l'analyse spectrale: {e}")
        results["hessian_analysis"]["spectral_error"] = str(e)

    # ========================================
    # 2. ANALYSE DES CONTRAINTES
    # ========================================
    if verbose:
        print("\n" + "=" * 60)
        print("ANALYSE DES CONTRAINTES")
        print("=" * 60)

    # Vérification de faisabilité
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])

    feasible = np.all(lower_bounds <= upper_bounds)
    volume = np.prod(upper_bounds - lower_bounds) if feasible else 0

    results["constraints_analysis"]["feasible"] = feasible
    results["constraints_analysis"]["domain_volume"] = volume
    results["constraints_analysis"]["lower_bounds"] = lower_bounds
    results["constraints_analysis"]["upper_bounds"] = upper_bounds

    if verbose:
        print(f"Domaine faisable: {feasible}")
        print(f"Volume du domaine: {volume:.6e}")
        print(f"Bornes inférieures: {lower_bounds}")
        print(f"Bornes supérieures: {upper_bounds}")

    # Point initial
    x0_feasible = np.all((x0 >= lower_bounds) & (x0 <= upper_bounds))
    results["constraints_analysis"]["x0_feasible"] = x0_feasible

    if verbose:
        print(f"Point initial faisable: {x0_feasible}")

    # ========================================
    # 3. RÉSOLUTION ET ANALYSE NUMÉRIQUE
    # ========================================
    if verbose:
        print("\n" + "=" * 60)
        print("RÉSOLUTION ET ANALYSE NUMÉRIQUE")
        print("=" * 60)

    def objective(x):
        return 0.5 * x.T @ Q @ x + p.T @ x

    def gradient(x):
        return Q @ x + p

    def hessian(x):
        return Q

    # Tentative de résolution avec différentes méthodes
    methods = ["trust-constr", "SLSQP", "L-BFGS-B", "proxsuite"]
    solutions = {}

    for method in methods:
        try:
            if verbose:
                print(f"\nTentative avec {method}...")

            start_time = __import__("time").time()

            if method == "trust-constr":
                res = opt.minimize(
                    objective,
                    x0,
                    method=method,
                    jac=gradient,
                    hess=hessian,
                    bounds=bounds,
                    options={"disp": False, "maxiter": 1000},
                )
                print(res.x)
            elif method == "proxsuite":
                qp.init(
                    H=Q,
                    g=p,
                    A=None,
                    b=None,
                    C=np.identity(6),
                    l=-np.ones(6),
                    u=np.ones(6),
                )
                qp.solve()
                print("proxsuite", qp.results.x)
            else:
                res = opt.minimize(
                    objective,
                    x0,
                    method=method,
                    jac=gradient,
                    bounds=bounds,
                    options={"disp": False, "maxiter": 1000},
                )
                print(res.x)

            solve_time = __import__("time").time() - start_time
            solutions[method] = {
                "success": res.success,
                "x": res.x,
                "fun": res.fun,
                "nit": res.nit,
                "solve_time": solve_time,
                "message": res.message,
            }

            if verbose:
                print(f"  Succès: {res.success}")
                print(f"  Itérations: {res.nit}")
                print(f"  Temps: {solve_time:.4f}s")
                print(f"  Valeur finale: {res.fun:.6e}")
                print(f"  Message: {res.message}")

        except Exception as e:
            solutions[method] = {"error": str(e)}
            if verbose:
                print(f"  Erreur: {e}")

    results["numerical_analysis"]["solutions"] = solutions

    # ========================================
    # 4. DIAGNOSTIC DE CONVERGENCE
    # ========================================
    if verbose:
        print("\n" + "=" * 60)
        print("DIAGNOSTIC DE CONVERGENCE")
        print("=" * 60)

    issues = []
    severity = []

    # Analyse des problèmes potentiels
    if "lambda_min" in results["hessian_analysis"]:
        lambda_min = results["hessian_analysis"]["lambda_min"]

        if lambda_min < -1e-10:
            issues.append("Problème non-convexe (λ_min < 0)")
            severity.append("CRITIQUE")

        elif abs(lambda_min) < 1e-10:
            issues.append("Hessienne singulière ou presque singulière")
            severity.append("MAJEUR")

        if "condition_number" in results["hessian_analysis"]:
            cond = results["hessian_analysis"]["condition_number"]
            if cond > 1e12:
                issues.append(f"Très mal conditionné (κ = {cond:.2e})")
                severity.append("MAJEUR")
            elif cond > 1e8:
                issues.append(f"Mal conditionné (κ = {cond:.2e})")
                severity.append("MODÉRÉ")

    # Analyse des performances de résolution
    successful_methods = [m for m, s in solutions.items() if s.get("success", False)]
    if len(successful_methods) == 0:
        issues.append("Aucune méthode n'a convergé")
        severity.append("CRITIQUE")
    elif len(successful_methods) < len(methods):
        issues.append("Convergence partielle selon les méthodes")
        severity.append("MODÉRÉ")

    # Analyse de cohérence des solutions
    if len(successful_methods) > 1:
        solutions_x = [solutions[m]["x"] for m in successful_methods]
        solutions_fun = [solutions[m]["fun"] for m in successful_methods]

        max_x_diff = max(
            np.linalg.norm(solutions_x[i] - solutions_x[j])
            for i in range(len(solutions_x))
            for j in range(i + 1, len(solutions_x))
        )
        max_fun_diff = max(
            abs(solutions_fun[i] - solutions_fun[j])
            for i in range(len(solutions_fun))
            for j in range(i + 1, len(solutions_fun))
        )

        if max_x_diff > 1e-6:
            issues.append(
                f"Solutions incohérentes entre méthodes (diff_x = {max_x_diff:.2e})"
            )
            severity.append("MAJEUR")
        if max_fun_diff > 1e-8:
            issues.append(
                f"Valeurs objectives incohérentes (diff_f = {max_fun_diff:.2e})"
            )
            severity.append("MODÉRÉ")

    results["convergence_diagnosis"]["issues"] = issues
    results["convergence_diagnosis"]["severity"] = severity

    if verbose:
        if issues:
            print("PROBLÈMES DÉTECTÉS:")
            for issue, sev in zip(issues, severity):
                print(f"  [{sev}] {issue}")
        else:
            print("Aucun problème majeur détecté.")

    # ========================================
    # 5. RECOMMANDATIONS
    # ========================================
    recommendations = []

    if not results["hessian_analysis"].get("is_symmetric", True):
        recommendations.append("Symétriser Q: Q_new = (Q + Q^T)/2")

    if results["hessian_analysis"].get("lambda_min", 1) < 1e-10:
        recommendations.append("Régulariser Q: Q_new = Q + εI avec ε > 0")

    if results["hessian_analysis"].get("condition_number", 1) > 1e8:
        recommendations.append("Préconditionner le système ou changer d'échelle")

    if not results["constraints_analysis"].get("x0_feasible", True):
        recommendations.append("Choisir un point initial faisable")

    if len(successful_methods) == 0:
        recommendations.append("Essayer d'autres solveurs (OSQP, CVXPY)")
        recommendations.append("Vérifier la faisabilité du problème")

    results["recommendations"].extend(recommendations)

    if verbose:
        print("\n" + "=" * 60)
        print("RECOMMANDATIONS")
        print("=" * 60)
        for i, rec in enumerate(results["recommendations"], 1):
            print(f"{i}. {rec}")

    return results


# H = np.array(
#     [
#         [1.57829, -0.056257, -0.061231, -0.000919349, 0.436799, -0.404142],
#         [-0.056257, 1.63934, 1.33457, 1.07629, -0.0590133, 0.897061],
#         [-0.061231, 1.33457, 1.21062, 1.03347, -0.0311883, 0.897061],
#         [-0.000919349, 1.07629, 1.03347, 1.01038, -0.00698783, 0.897061],
#         [0.436799, -0.0590133, -0.0311883, -0.00698783, 1.00687, 0.0],
#         [-0.404142, 0.897061, 0.897061, 0.897061, 0.0, 1.0001],
#     ]
# )

# H += 1 * np.identity(6)

p = np.array([0.0516045, 7.67063, 6.29417, 5.1057, 2.49459, 4.43859])
G1 = np.array(
    [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ],
    dtype=float,
)

lb = 1 * np.array([-1, -1, -1, -1, -1, -1], dtype=float)
ub = 1 * np.array([1, 1, 1, 1, 1, 1], dtype=float)

n = 6
m = G1.shape[0]
qp = proxsuite.proxqp.dense.QP(n, 0, m)


with open("matrices.pkl", "rb") as f:  # "rb" = read binary
    H, p, _ = pickle.load(f)

# print(H)
# print(p)
analyze_qp_convergence(H, p)
# input()

# Check strict convexity
assert np.all(np.linalg.eigvals(H) > 1e-4), "H must be strictly positive definite"


def solve_qp(H, g, G, lb, ub, eps_rel=1e-10, eps_abs=1e-10):
    qp.settings.eps_abs = eps_rel
    qp.settings.eps_rel = eps_abs
    qp.init(H=H, g=g, A=None, b=None, C=G, l=lb, u=ub)
    qp.solve()
    return qp.results.x.copy(), qp.results.z


def finite_diff_grad(H, g, G, lb, ub, eps=1e-7):
    n = H.shape[0]

    dldH = np.zeros((n, n, n))
    dldg = np.zeros((n, n))

    # Gradient w.r.t. H
    for j in range(n):
        for k in range(n):
            H_perturb = H.copy()
            H_perturb[j, k] += eps
            x_plus, _ = solve_qp(H_perturb, g, G, lb, ub)

            H_perturb[j, k] -= 2 * eps
            x_minus, _ = solve_qp(H_perturb, g, G, lb, ub)

            dldH[:, j, k] = (x_plus - x_minus) / (2 * eps)

    # Gradient w.r.t. g
    for k in range(n):
        g_perturb = g.copy()
        g_perturb[k] += eps
        x_plus, _ = solve_qp(H, g_perturb, G, lb, ub)

        g_perturb[k] -= 2 * eps
        x_minus, _ = solve_qp(H, g_perturb, G, lb, ub)

        dldg[:, k] = (x_plus - x_minus) / (2 * eps)

    return (dldH + dldH.swapaxes(1, 2)) / 2, dldg


dldH_fd, dldg_fd = finite_diff_grad(H, p, G1, lb, ub, eps=1e-8)

dl_dQ = np.zeros((n, n, n))
dl_dp = np.zeros((n, n))
for i in range(n):
    dloss = np.zeros(2 * n)
    dloss[i] = 1
    solve_qp(H, p, G1, lb, ub)
    ana_grad = proxsuite.proxqp.dense.compute_backward(qp, dloss, 1e-10, 1e-10, 1e-10)
    dl_dQ_ = qp.model.backward_data.dL_dH
    dl_dp_ = qp.model.backward_data.dL_dg
    dl_dQ[i] = dl_dQ_
    dl_dp[i] = dl_dp_

print("QP output:")
print(solve_qp(H, p, G1, lb, ub)[0])

print("=== Gradient comparison w.r.t. H (no saturation) ===")
# print(dl_dQ)
# print(dldH_fd)
err_abs = np.max(np.abs(dl_dQ - dldH_fd))
err_rel = err_abs / (np.max(np.abs(dl_dQ)) + 1e-6)
print("Max error:", err_abs)
print("\033[91mRelative error:\033[0m", err_rel)

print("\n=== Gradient comparison w.r.t. g (no saturation) ===")
err_abs = np.max(np.abs(dl_dp - dldg_fd))
err_rel = err_abs / (np.max(np.abs(dl_dp)) + 1e-6)
print("Max error:", err_abs)
print("\033[91mRelative error:\033[0m", err_rel)

lb = 1 * np.array([-1, -1, -1, -1, -1, -1], dtype=float)
ub = 1 * np.array([1, 1, 1, 1, 1, 1], dtype=float)

dldH_fd, dldg_fd = finite_diff_grad(H, p, G1, lb, ub, eps=1e-8)

dl_dQ = np.zeros((n, n, n))
dl_dp = np.zeros((n, n))
for i in range(n):
    dloss = np.zeros(2 * n)
    dloss[i] = 1
    solve_qp(H, p, G1, lb, ub)
    ana_grad = proxsuite.proxqp.dense.compute_backward(qp, dloss, 1e-7, 1e-7, 1e-7)
    dl_dQ_ = qp.model.backward_data.dL_dH
    dl_dp_ = qp.model.backward_data.dL_dg
    dl_dQ[i] = dl_dQ_
    dl_dp[i] = dl_dp_

print("=== Gradient comparison w.r.t. H (saturation) ===")
print("qp output", solve_qp(H, p, G1, lb, ub)[0])
err_abs = np.max(np.abs(dl_dQ - dldH_fd))
err_rel = err_abs / (np.max(np.abs(dl_dQ)) + 1e-6)
print("Max error:", err_abs)
print("\033[91mRelative error:\033[0m", err_rel)

print("\n=== Gradient comparison w.r.t. g (saturation) ===")
err_abs = np.max(np.abs(dl_dp - dldg_fd))
err_rel = err_abs / (np.max(np.abs(dl_dp)) + 1e-6)
print("Max error:", err_abs)
print("\033[91mRelative error:\033[0m", err_rel)


histQ = []
histp = []
epss = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]
for ep in epss:
    dldH_fd, dldg_fd = finite_diff_grad(H, p, G1, lb, ub, eps=ep)

    dl_dQ = np.zeros((n, n, n))
    dl_dp = np.zeros((n, n))
    for i in range(n):
        dloss = np.zeros(2 * n)
        dloss[i] = 1
        solve_qp(H, p, G1, lb, ub)
        ana_grad = proxsuite.proxqp.dense.compute_backward(qp, dloss, 1e-7, 1e-7, 1e-7)
        dl_dQ_ = qp.model.backward_data.dL_dH
        dl_dp_ = qp.model.backward_data.dL_dg
        dl_dQ[i] = dl_dQ_
        dl_dp[i] = dl_dp_

    # print("=== Gradient comparison w.r.t. H (saturation) ===")
    # print("qp output", solve_qp(H, p, G1, lb, ub)[0])
    err_abs = np.max(np.abs(dl_dQ - dldH_fd))
    err_rel = err_abs / (np.max(np.abs(dl_dQ)) + 1e-6)
    histQ.append(err_rel)
    # print("Max error:", err_abs)
    # print("\033[91mRelative error:\033[0m", err_rel)

    # print("\n=== Gradient comparison w.r.t. g (saturation) ===")
    err_abs = np.max(np.abs(dl_dp - dldg_fd))
    err_rel = err_abs / (np.max(np.abs(dl_dp)) + 1e-6)
    histp.append(err_rel)
    # print("Max error:", err_abs)
    # print("\033[91mRelative error:\033[0m", err_rel)


import matplotlib.pyplot as plt

plt.plot(epss, histQ)
plt.xscale("log")
plt.yscale("log")
plt.show()
plt.plot(epss, histp)
plt.xscale("log")
plt.yscale("log")
plt.show()
