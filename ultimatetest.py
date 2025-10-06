import meshcat.geometry as g
import numpy as np
import pinocchio  # type: ignore
from math import sqrt  # type: ignore  # noqa: F401
import torch  # type: ignore
import copy  # type: ignore  # noqa: F401
import sys  # type: ignore
import cvxpy as cp  # type: ignore
from tqdm import tqdm  # type: ignore
from autogradQP import QPkkt
import matplotlib.pyplot as plt
import pinocchio as pin
from colorama import Fore, Style, init
import torch.nn as nn
from scipy.optimize import minimize
import time
import example_robot_data as erd
from viewer import Viewer
import proxsuite
from qpsolvers import solve_qp
from Quartz import CGEventSourceKeyState
import os
import matplotlib.pyplot as plt
import pickle
import scipy.optimize as opt
from tqdm import tqdm
import seaborn as sns
from colorama import Fore, Style, init
import viewer
from casadi import *
import casadi as ca
import pinocchio.casadi as cpin


sys.path.insert(0, "/Users/mscheffl/Desktop/pinocchio-minimal-main/build/python")
import tartempion  # type: ignore


SPACE_KEY_CODE = 49
F12_KEY_CODE = 53


def is_F12_pressed():
    return CGEventSourceKeyState(0, F12_KEY_CODE)


def is_space_pressed():
    return CGEventSourceKeyState(0, SPACE_KEY_CODE)


init(autoreset=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q_reg = 1e-4
bound = -1000
robot = erd.load("ur5")
rmodel, gmodel, vmodel = robot.model, robot.collision_model, robot.visual_model
rmodel.data = rmodel.createData()
tool_id = 21
dt = 1e-2
pin.seed(21)
np.random.seed(21)

logTarget = pin.Motion.Random()
q_ = pin.randomConfiguration(rmodel)
lambda_ = -1
T_star = pin.SE3.Random()

qp = proxsuite.proxqp.dense.QP(rmodel.nq, 0, rmodel.nq)
qp.settings.eps_abs = 1e-8
qp.settings.eps_rel = 1e-8


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
                    C=np.identity(rmodel.nq),
                    l=-np.ones(rmodel.nq),
                    u=np.ones(rmodel.nq),
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


# ========================================
# FONCTIONS UTILITAIRES SUPPLÉMENTAIRES
# ========================================


def generate_test_qp(n=10, condition_number=1e3, problem_type="well_conditioned"):
    """
    Génère des problèmes QP de test pour validation
    """

    if problem_type == "well_conditioned":
        # Problème bien conditionné
        eigenvals = np.logspace(0, np.log10(condition_number), n)
        U, _ = np.linalg.qr(np.random.randn(n, n))
        Q = U @ np.diag(eigenvals) @ U.T

    elif problem_type == "ill_conditioned":
        # Problème mal conditionné
        eigenvals = np.array([1e-8] + [1.0] * (n - 1))
        U, _ = np.linalg.qr(np.random.randn(n, n))
        Q = U @ np.diag(eigenvals) @ U.T

    elif problem_type == "non_convex":
        # Problème non-convexe
        eigenvals = np.random.randn(n)
        U, _ = np.linalg.qr(np.random.randn(n, n))
        Q = U @ np.diag(eigenvals) @ U.T

    else:
        Q = np.random.randn(n, n)
        Q = Q @ Q.T  # Définie positive

    p = np.random.randn(n)

    return Q, p


def plot_convergence_analysis(results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    if "eigenvalues" in results["hessian_analysis"]:
        eigenvals = results["hessian_analysis"]["eigenvalues"]
        axes[0, 0].semilogy(sorted(eigenvals), "bo-")
        axes[0, 0].set_title("Spectre de Q")
        axes[0, 0].set_xlabel("Index")
        axes[0, 0].set_ylabel("Valeur propre")
        axes[0, 0].grid(True)

    solutions = results["numerical_analysis"]["solutions"]
    methods = list(solutions.keys())
    times = [s.get("solve_time", 0) for s in solutions.values()]
    success = [s.get("success", False) for s in solutions.values()]

    colors = ["green" if s else "red" for s in success]
    axes[0, 1].bar(methods, times, color=colors)
    axes[0, 1].set_title("Temps de résolution")
    axes[0, 1].set_ylabel("Temps (s)")
    axes[0, 1].tick_params(axis="x", rotation=45)

    if "domain_volume" in results["constraints_analysis"]:
        vol = results["constraints_analysis"]["domain_volume"]
        axes[1, 0].text(
            0.5,
            0.5,
            f"Volume domaine:\n{vol:.2e}",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
        )
        axes[1, 0].set_title("Info contraintes")
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)

    issues = results["convergence_diagnosis"]["issues"]
    severity = results["convergence_diagnosis"]["severity"]

    if issues:
        sev_colors = {"CRITIQUE": "red", "MAJEUR": "orange", "MODÉRÉ": "yellow"}
        colors = [sev_colors.get(s, "blue") for s in severity]
        axes[1, 1].barh(range(len(issues)), [1] * len(issues), color=colors)
        axes[1, 1].set_yticks(range(len(issues)))
        axes[1, 1].set_yticklabels(
            [f"{s}: {i[:20]}..." for s, i in zip(severity, issues)]
        )
        axes[1, 1].set_title("Problèmes détectés")

    plt.tight_layout()
    plt.show()


def forward(q, logTarget, T_star, prin=False):

    pin.framesForwardKinematics(rmodel, rmodel.data, q)
    pin.updateFramePlacement(rmodel, rmodel.data, tool_id)
    J_cine = pin.computeFrameJacobian(rmodel, rmodel.data, q, 21, pin.LOCAL)
    Q = J_cine.T @ J_cine + q_reg * np.identity(rmodel.nq)
    err = pin.log6(rmodel.data.oMf[tool_id].actInv(pin.exp6(logTarget)))
    # print("err", err)
    p = lambda_ * J_cine.T @ err
    lb = -1 * np.ones(rmodel.nq)
    ub = -lb
    identity = np.identity(rmodel.nq)
    qp.init(H=Q, g=p, A=None, b=None, C=identity, l=lb * 1000, u=ub * 1000)
    qp.solve()
    q_next = pin.integrate(rmodel, q, dt * qp.results.x.copy())
    # print("q_next", q_next)
    # print("q_next inte", pin.integrate(rmodel, q, dt * qp.results.x.copy()))
    # input()
    pin.framesForwardKinematics(rmodel, rmodel.data, q_next)
    pin.updateFramePlacement(rmodel, rmodel.data, 21)
    print("cond", np.linalg.cond(J_cine))
    print(
        "cond qnext",
        np.linalg.cond(
            pin.computeFrameJacobian(rmodel, rmodel.data, q_next, 21, pin.LOCAL)
        ),
    )
    print(
        np.linalg.eigh(
            pin.computeFrameJacobian(rmodel, rmodel.data, q_next, 21, pin.LOCAL)
        )
    )
    print("j", J_cine)
    print(
        "jnext",
        pin.computeFrameJacobian(rmodel, rmodel.data, q_next, 21, pin.LOCAL),
    )
    print(q)
    print(q_next)
    print(rmodel.data.oMf[21])
    print(qp.results.x.copy())
    return ((pin.log6(rmodel.data.oMf[21].inverse() * T_star).vector) ** 2).sum()


def casadi_forward(rmodel, tool_id, q_reg, lambda_, dt, T_star):
    nq = rmodel.nq
    q = ca.SX.sym("q", nq)
    logTarget = ca.SX.sym("logTarget", 6)
    rmodel = cpin.Model(rmodel)
    data = rmodel.createData()
    T_star_se3 = cpin.SE3(T_star)
    cpin.framesForwardKinematics(rmodel, data, q)
    cpin.updateFramePlacement(rmodel, data, tool_id)
    J_cine = cpin.computeFrameJacobian(rmodel, data, q, tool_id, pin.LOCAL)
    Q = J_cine.T @ J_cine + q_reg * ca.SX.eye(nq)
    err = cpin.log6(data.oMf[tool_id].actInv(cpin.exp6(logTarget))).vector
    # print("err casa", err)
    p = lambda_ * J_cine.T @ err
    dq = -ca.solve(Q, p)
    q_next = q + dt * dq
    cpin.framesForwardKinematics(rmodel, data, q_next)
    cpin.updateFramePlacement(rmodel, data, tool_id)
    cost = ca.sumsqr(cpin.log6(data.oMf[tool_id].inverse() * T_star_se3).vector)
    grad_q = ca.gradient(cost, q)
    grad_logTarget = ca.gradient(cost, logTarget)
    # grad_Q = ca.gradient(cost, Q)
    # grad_J_cine = ca.gradient(cost, J_cine)

    f = ca.Function(
        "f",
        [q, logTarget],
        [cost, grad_q, grad_logTarget],
    )

    def f_numpy(q_val, logTarget_val):
        out = f(ca.DM(q_val), ca.DM(logTarget_val))
        return (
            np.array(out[0]).flatten(),
            np.array(out[1]).flatten(),
            np.array(out[2]).flatten(),
        )

    return f_numpy


def forward_Q(q1, q, logTarget, T_star):
    J_cine2 = pin.computeFrameJacobian(rmodel, rmodel.data, q1, 21, pin.LOCAL)
    Q = J_cine2.T @ J_cine2 + q_reg * np.identity(rmodel.nq)
    pin.framesForwardKinematics(rmodel, rmodel.data, q)
    pin.updateFramePlacement(rmodel, rmodel.data, tool_id)
    J_cine = pin.computeFrameJacobian(rmodel, rmodel.data, q, 21, pin.LOCAL)
    err = pin.log6(rmodel.data.oMf[tool_id].actInv(pin.exp6(logTarget)))
    p = lambda_ * J_cine.T @ err
    lb = -np.ones(rmodel.nq)
    ub = -lb
    identity = np.identity(rmodel.nq)
    qp.init(H=Q, g=p, A=None, b=None, C=identity, l=lb * 10000, u=ub * 10000)
    qp.solve()
    q_next = pin.integrate(rmodel, q, dt * qp.results.x.copy())
    pin.framesForwardKinematics(rmodel, rmodel.data, q_next)
    pin.updateFramePlacement(rmodel, rmodel.data, 21)
    return ((pin.log6(rmodel.data.oMf[21].inverse() * T_star).vector) ** 2).sum()


def forward_Q2(Q, q, logTarget, T_star):
    Q = Q.T @ Q + q_reg * np.identity(rmodel.nq)
    pin.framesForwardKinematics(rmodel, rmodel.data, q)
    pin.updateFramePlacement(rmodel, rmodel.data, tool_id)
    J_cine = pin.computeFrameJacobian(rmodel, rmodel.data, q, 21, pin.LOCAL)
    err = pin.log6(rmodel.data.oMf[tool_id].actInv(pin.exp6(logTarget)))
    p = lambda_ * J_cine.T @ err
    lb = -np.ones(rmodel.nq)
    ub = -lb
    identity = np.identity(rmodel.nq)
    qp.init(H=Q, g=p, A=None, b=None, C=identity, l=lb * 10000, u=ub * 10000)
    qp.solve()
    q_next = pin.integrate(rmodel, q, dt * qp.results.x.copy())
    pin.framesForwardKinematics(rmodel, rmodel.data, q_next)
    pin.updateFramePlacement(rmodel, rmodel.data, 21)
    return ((pin.log6(rmodel.data.oMf[21].inverse() * T_star).vector) ** 2).sum()


def forward_Tq(qT, q, logTarget, T_star):
    J_cine = pin.computeFrameJacobian(rmodel, rmodel.data, q, 21, pin.LOCAL)
    pin.framesForwardKinematics(rmodel, rmodel.data, qT)
    pin.updateFramePlacement(rmodel, rmodel.data, tool_id)
    Q = J_cine.T @ J_cine + q_reg * np.identity(rmodel.nq)
    err = pin.log6(rmodel.data.oMf[tool_id].actInv(pin.exp6(logTarget)))
    p = lambda_ * J_cine.T @ err
    lb = -np.ones(rmodel.nq)
    ub = -lb
    identity = np.identity(rmodel.nq)
    qp.init(H=Q, g=p, A=None, b=None, C=identity, l=lb * 100000, u=ub * 100000)
    qp.solve()
    q_next = pin.integrate(rmodel, q, dt * qp.results.x.copy())
    pin.framesForwardKinematics(rmodel, rmodel.data, q_next)
    pin.updateFramePlacement(rmodel, rmodel.data, 21)
    return ((pin.log6(rmodel.data.oMf[21].inverse() * T_star).vector) ** 2).sum()


def forward_Jp(qJ, q, logTarget, T_star):
    J = pin.computeFrameJacobian(rmodel, rmodel.data, qJ, 21, pin.LOCAL)
    pin.framesForwardKinematics(rmodel, rmodel.data, q)
    pin.updateFramePlacement(rmodel, rmodel.data, tool_id)
    J_cine = pin.computeFrameJacobian(rmodel, rmodel.data, q, 21, pin.LOCAL)
    Q = J_cine.T @ J_cine + q_reg * np.identity(rmodel.nq)
    err = pin.log6(rmodel.data.oMf[tool_id].actInv(pin.exp6(logTarget)))
    p = lambda_ * J.T @ err
    lb = -np.ones(rmodel.nq)
    ub = -lb
    identity = np.identity(rmodel.nq)
    qp.init(H=Q, g=p, A=None, b=None, C=identity, l=lb * 100000, u=ub * 10000)
    qp.solve()
    q_next = pin.integrate(rmodel, q, dt * qp.results.x.copy())
    pin.framesForwardKinematics(rmodel, rmodel.data, q_next)
    pin.updateFramePlacement(rmodel, rmodel.data, 21)
    return ((pin.log6(rmodel.data.oMf[21].inverse() * T_star).vector) ** 2).sum()


def forward_q_next(q, T_star: pin.SE3):
    pin.framesForwardKinematics(rmodel, rmodel.data, q)
    pin.updateFramePlacement(rmodel, rmodel.data, 21)
    return ((pin.log6(rmodel.data.oMf[21].inverse() * T_star).vector) ** 2).sum()


def casadi_forward_q_next(rmodel, T_star: pin.SE3):
    T_star = cpin.SE3(T_star)
    q = ca.SX.sym("q", rmodel.nq)
    rmodel = cpin.Model(rmodel)
    data = rmodel.createData()
    cpin.framesForwardKinematics(rmodel, data, q)
    cpin.updateFramePlacement(rmodel, data, 21)
    cost = ca.sumsqr(cpin.log6(data.oMf[21].inverse() * T_star).vector)
    grad_q = ca.gradient(cost, q)

    f = ca.Function(
        "f",
        [q],
        [cost, grad_q],
    )

    def f_numpy(q_val):
        out = f(ca.DM(q_val))
        return (
            np.array(out[0]).flatten(),
            np.array(out[1]).flatten(),
        )

    return f_numpy


def casadi_forward_Q2(rmodel, q, logTarget, T_star):
    T_star = cpin.SE3(T_star)
    pin.framesForwardKinematics(rmodel, rmodel.data, q)
    pin.updateFramePlacement(rmodel, rmodel.data, tool_id)
    J_cine = pin.computeFrameJacobian(rmodel, rmodel.data, q, 21, pin.LOCAL)
    err = pin.log6(rmodel.data.oMf[tool_id].actInv(pin.exp6(logTarget)))
    p = lambda_ * J_cine.T @ err
    lb = -np.ones(rmodel.nq)
    ub = -lb
    identity = np.identity(rmodel.nq)
    rmodel = cpin.Model(rmodel)
    data = rmodel.createData()
    Q = ca.SX.sym("Q", 6, 6)
    QQP = Q.T @ Q + q_reg * np.identity(rmodel.nq)
    dq = -ca.solve(QQP, p)

    q_next = cpin.integrate(rmodel, ca.SX(q), dq * dt)
    cpin.framesForwardKinematics(rmodel, data, q_next)
    cpin.updateFramePlacement(rmodel, data, 21)
    cost = ca.sumsqr(cpin.log6(data.oMf[21].inverse() * T_star).vector)
    grad = ca.jacobian(cost, Q)

    f = ca.Function(
        "f",
        [Q],
        [cost, grad, q_next, dq],
    )

    def f_numpy(q_val):
        out = f(ca.DM(q_val))
        return (
            np.array(out[0]).flatten(),
            np.array(out[1]).flatten(),
            np.array(out[2]).flatten(),
            np.array(out[3]).flatten(),
        )

    return f_numpy


def casadiforward_Jp(rmodel, q, logTarget, T_star):
    qJ = ca.SX.sym("qJ", rmodel.nq)
    pin.framesForwardKinematics(rmodel, rmodel.data, q)
    pin.updateFramePlacement(rmodel, rmodel.data, tool_id)
    J_cine = pin.computeFrameJacobian(rmodel, rmodel.data, q, 21, pin.LOCAL)
    Q = J_cine.T @ J_cine + q_reg * np.identity(rmodel.nq)
    err = pin.log6(rmodel.data.oMf[tool_id].actInv(pin.exp6(logTarget)))
    rmodel = cpin.Model(rmodel)
    data = rmodel.createData()
    J = cpin.computeFrameJacobian(rmodel, data, qJ, 21, pin.LOCAL)
    p = lambda_ * J.T @ err.vector
    lb = -np.ones(rmodel.nq)
    ub = -lb
    identity = np.identity(rmodel.nq)
    dq = -ca.solve(Q, p)

    q_next = q + dt * dq
    cpin.framesForwardKinematics(rmodel, data, q_next)
    cpin.updateFramePlacement(rmodel, data, 21)
    cost = ca.sumsqr(cpin.log6(data.oMf[21].inverse() * cpin.SE3(T_star)).vector)
    grad = ca.gradient(cost, qJ)

    f = ca.Function(
        "f",
        [qJ],
        [cost, grad],
    )

    def f_numpy(q_val):
        out = f(ca.DM(q_val))
        return (
            np.array(out[0]).flatten(),
            np.array(out[1]).flatten(),
        )

    return f_numpy


def numeric_grad_forward_q_next(forward_q_next, q, T_star, eps=1e-6):
    nq = q.size
    grad = np.zeros(nq, dtype=np.float64)

    qp = q.copy()
    qm = q.copy()

    for i in range(nq):
        qp[i] = q[i] + eps
        qm[i] = q[i] - eps
        f_plus = forward_q_next(qp, T_star)
        f_minus = forward_q_next(qm, T_star)
        grad[i] = (f_plus - f_minus) / (2.0 * eps)
        qp[i] = q[i]
        qm[i] = q[i]

    return grad


def numeric_gradient_forward(forward, q, logTarget, T_star, eps=1e-6, out=False):
    nq = q.size

    lt_vec = logTarget.vector.copy()
    nlt = lt_vec.size

    grad_q = np.zeros(nq, dtype=np.float64)
    grad_logTarget = np.zeros(nlt, dtype=np.float64)
    # vi = Viewer(rmodel, gmodel, vmodel, True)
    # vi1 = Viewer(rmodel, gmodel, vmodel, True)
    # vi2 = Viewer(rmodel, gmodel, vmodel, True)
    qp = q.copy()
    qm = q.copy()
    for i in range(nq):
        qp[i] = q[i] + eps
        qm[i] = q[i] - eps
        f_plus = forward(qp, logTarget, T_star)
        f_minus = forward(qm, logTarget, T_star)
        grad_q[i] = (f_plus - f_minus) / (2.0 * eps)
        qp[i] = q[i]
        qm[i] = q[i]
        print(grad_q[i])
        print(f_minus)
        print(f_plus)
        input()
        # print(forward(q, logTarget, T_star))
        # vi.display(q)
        # vi1.display(qp)
        # vi2.display(qm)
        # input()

    lt_p = lt_vec.copy()
    lt_m = lt_vec.copy()
    for j in range(nlt):
        lt_p[j] = lt_vec[j] + eps
        lt_m[j] = lt_vec[j] - eps
        motion_p = pin.Motion(lt_p)
        motion_m = pin.Motion(lt_m)
        f_plus = forward(q, motion_p, T_star)
        f_minus = forward(q, motion_m, T_star)
        grad_logTarget[j] = (f_plus - f_minus) / (2.0 * eps)
        lt_p[j] = lt_vec[j]
        lt_m[j] = lt_vec[j]

    if out:
        return forward(q, logTarget, T_star), grad_q, grad_logTarget
    return grad_q, grad_logTarget


def numeric_grad_Q(forward_Q_use, Q, q, logTarget, T_star, eps=1e-6):
    nq = Q.shape[0]
    grad_Q = np.zeros_like(Q)
    for i in range(nq):
        Qp = Q.copy()
        Qm = Q.copy()
        Qp[i] += eps
        Qm[i] -= eps
        f_plus = forward_Q_use(Qp, q, logTarget, T_star)
        f_minus = forward_Q_use(Qm, q, logTarget, T_star)
        grad_Q[i] = (f_plus - f_minus) / (2.0 * eps)

    return grad_Q


def numeric_grad_Q2(forward_Q_use, Q, q, logTarget, T_star, eps=1e-6):
    nq = Q.shape[0]
    grad_Q = np.zeros_like(Q)

    for i in range(nq):
        for j in range(nq):
            Qp = Q.copy()
            Qm = Q.copy()
            Qp[i, j] += eps
            Qm[i, j] -= eps
            f_plus = forward_Q_use(Qp, q, logTarget, T_star)
            f_minus = forward_Q_use(Qm, q, logTarget, T_star)
            grad_Q[i, j] = (f_plus - f_minus) / (2.0 * eps)

    return grad_Q


def compute_frame_hessian(q):
    v = np.zeros(rmodel.nq)
    a = np.zeros(rmodel.nq)
    Hessian = np.zeros((6, rmodel.nq, rmodel.nq))
    for k in range(rmodel.nq):
        v[:] = 0.0
        v[k] = 1.0
        pin.computeForwardKinematicsDerivatives(rmodel, rmodel.data, q, v, v)
        (v_partial_dq, v_partial_dv) = pin.getFrameVelocityDerivatives(
            rmodel, rmodel.data, tool_id, pin.LOCAL
        )
        Hessian[:, :, k] = v_partial_dq
    return Hessian


def fd_hess(q, eps=1e-7):
    hess = np.zeros((6, 6, 6))
    for i in range(len(q)):
        qplus = q.copy()
        qminus = q.copy()
        qplus[i] += eps
        qminus[i] -= eps
        jplus = pin.computeFrameJacobian(rmodel, rmodel.data, qplus, 21, pin.LOCAL)
        jminus = pin.computeFrameJacobian(rmodel, rmodel.data, qminus, 21, pin.LOCAL)
        hess[:, :, i] = (jplus - jminus) / (2 * eps)
    return hess


def numerical_jacobian(f, x0, eps=1e-8):
    n = 6
    m = 6
    J = np.zeros((m, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        J[:, i] = (f(x0, pin.Motion(dx)) - f(x0, pin.Motion(-dx))) / (2 * eps)
    return J


def fexp(delta, xi):
    M = pin.exp6(delta)
    M_plus = pin.exp6(xi + delta)
    return pin.log6(M.inverse() * M_plus)


def analytical(q, logTarget, T_star, debug=False, out=False):
    pin.framesForwardKinematics(rmodel, rmodel.data, q)
    pin.updateFramePlacement(rmodel, rmodel.data, tool_id)
    J_cine = pin.computeFrameJacobian(rmodel, rmodel.data, q, 21, pin.LOCAL)
    Q = J_cine.T @ J_cine + q_reg * np.identity(rmodel.nq)
    err = pin.log6(rmodel.data.oMf[tool_id].actInv(pin.exp6(logTarget)))
    Adj = (rmodel.data.oMf[tool_id].actInv(pin.exp6(logTarget))).toActionMatrixInverse()
    Jlog = pin.Jlog6((rmodel.data.oMf[tool_id].actInv(pin.exp6(logTarget))))
    p = lambda_ * J_cine.T @ err
    lb = -100 * np.ones(rmodel.nq)
    ub = -lb
    identity = np.identity(rmodel.nq)
    if debug:
        print("cost", Q)
        print("target", p)
        print("ub", ub)
        print("lb", lb)
    qp.init(H=Q, g=p, A=None, b=None, C=identity, l=lb * 1000000, u=ub * 100000)
    qp.solve()
    q_next = pin.integrate(rmodel, q, dt * qp.results.x.copy())
    if debug:
        print("qp out :", qp.results.x.copy())
        print("q", q)
        print("q_next", q_next)
        print("J", J_cine)
        print("J cond", np.linalg.cond(J_cine))
        print("err", err)
        print("J.Terr", J_cine.T @ err)
        print("p", p)
        print("T_star", T_star)
    pin.framesForwardKinematics(rmodel, rmodel.data, q_next)
    pin.updateFramePlacement(rmodel, rmodel.data, 21)

    # backward
    pin.framesForwardKinematics(rmodel, rmodel.data, q_next)
    pin.updateFramePlacement(rmodel, rmodel.data, 21)
    A = rmodel.data.oMf[21].copy()

    C: pin.SE3 = A.copy().actInv(T_star.copy())
    adj = np.array(C.toActionMatrixInverse()).copy()

    jlog = pin.Jlog6(C)
    Jexp = pin.Jexp6(logTarget)
    log = pin.log6(C).vector
    jac = pin.computeFrameJacobian(rmodel, rmodel.data, q_next, 21, pin.LOCAL)
    dl_dq_next = -jac.T @ adj.T @ jlog.T @ (2 * log)
    # casa_f = casadi_forward_q_next(rmodel, T_star.copy())
    # out, casa_dldq = casa_f(q_next)
    outpython = ((pin.log6(rmodel.data.oMf[21].inverse() * T_star).vector) ** 2).sum()
    # out_old = out
    # assert np.allclose(out, outpython)
    # fd = numeric_grad_forward_q_next(forward_q_next, q_next, T_star, 1e-5)
    # print("casa dldq", casa_dldq)
    # print("python dldq", dl_dq_next)
    # print("fd", fd)
    # assert np.allclose(casa_dldq, dl_dq_next)
    # assert np.allclose(fd, dl_dq_next, atol=1e-5)
    # assert np.allclose(fd, dl_dq_next, atol=1e-5)

    # try:
    #     if debug:
    #         print(Fore.RED + "-" * 50)
    #         np.testing.assert_allclose(
    #             dl_dq_next,
    #             numeric_grad_forward_q_next(forward_q_next, q_next, T_star, 1e-5),
    #             rtol=1e-3,
    #             atol=3e-6,
    #         )
    #         print("dldq ok")
    # except Exception as e:
    #     print("dldq nok")
    #     print(str(e))
    # with open("matrices.pkl", "wb") as f:  # "wb" = write binary
    #     pickle.dump((Q, p, np.hstack([dl_dq_next, 0 * dl_dq_next]) * dt), f)

    qp.init(H=Q, g=p, A=None, b=None, C=identity, l=100000 * lb, u=100000 * ub)
    qp.solve()
    Hess = compute_frame_hessian(q).swapaxes(1, 2)
    J = pin.computeFrameJacobian(rmodel, rmodel.data, q, 21, pin.LOCAL)
    J1 = pin.computeFrameJacobian(rmodel, rmodel.data, q_next, 21, pin.LOCAL)
    # np.testing.assert_allclose(
    #     Hess @ qp.results.x.copy(), (J1 - J) / dt, rtol=1e3, atol=3e-6
    # )
    # try:
    #     if debug:
    #         print(Fore.RED + "-" * 50)
    #         np.testing.assert_allclose(
    #             compute_frame_hessian(q).swapaxes(1, 2),
    #             fd_hess(q, 1e-6),
    #             rtol=1e-3,
    #             atol=3e-6,
    #         )
    #         print("Hessian ok")
    # except Exception as e:
    #     print("Hessian nok")
    #     print(str(e))

    proxsuite.proxqp.dense.compute_backward(
        qp, np.hstack([dl_dq_next, 0 * dl_dq_next]) * dt, 1e-8, 1e-8, 1e-8
    )
    dl_dQ_ = qp.model.backward_data.dL_dH
    dl_dp_ = qp.model.backward_data.dL_dg
    grad_J_kine = np.einsum("kl,kjl->j", 2 * J_cine @ dl_dQ_, compute_frame_hessian(q))
    # f_casa_grad_J_kine = casadi_forward_Q2(rmodel, q, logTarget, T_star)
    # fd = numeric_grad_Q2(forward_Q2, J_cine, q, logTarget, T_star, 1e-5)
    # out, grad, qqnext, jgac = f_casa_grad_J_kine(J_cine)
    # gradJcost = 2 * J_cine @ dl_dQ_
    # print("now")
    # print(q_next)
    # print(qqnext)
    # print(jgac)
    # print(qp.results.x)
    # assert np.allclose(
    #     J_cine.flatten(), (J_cine.T @ J_cine + q_reg * np.identity(6)).flatten()
    # )
    # print(out)
    # print(out_old)
    # print(outpython)
    # assert np.allclose(out, outpython)
    # print("casa dLdJ", grad)
    # print("ana dLdJ", gradJcost.T.flatten())
    # print("fd dLdJ", fd)
    # assert np.allclose(grad.flatten(), gradJcost.T.flatten(), atol=1e-5)
    # assert np.allclose(fd.flatten(), gradJcost.T.flatten(), atol=1e-5)
    # assert np.allclose(grad.flatten(), fd.flatten(), atol=1e-5)

    # try:
    #     if debug:
    #         np.testing.assert_allclose(
    #             gradJcost.flatten(),
    #             fd.flatten(),
    #             rtol=1e-3,
    #             atol=1e-5,
    #         )
    #         print("dldJ1 ok")
    # except Exception as e:
    #     print("dldJ1 nok")
    #     print(str(e))
    # input()
    # try:
    #     if debug:
    #         print(Fore.RED + "-" * 50)
    #         np.testing.assert_allclose(
    #             grad_J_kine,
    #             numeric_grad_Q(forward_Q, q, q, logTarget, T_star, 1e-6),
    #             rtol=1e-3,
    #             atol=3e-6,
    #         )
    #         print("dldqJ1 ok")
    # except Exception as e:
    #     print("dldqJ1 nok")
    #     print(str(e))

    grad_J_kine_p = -lambda_ * np.einsum(
        "l,kjl,k->j", -dl_dp_, compute_frame_hessian(q), err
    )
    fd = numeric_grad_Q(forward_Jp, q, q, logTarget, T_star, 1e-8)
    # f_casa = casadiforward_Jp(rmodel, q, logTarget, T_star)
    # out, grad = f_casa(q)
    # assert np.allclose(out, outpython)
    # print("casa dLdJp", grad.flatten())
    # print("ana dLdJp", grad_J_kine_p.flatten())
    # print("fd dLdJp", fd.flatten())
    # assert np.allclose(grad.flatten(), grad_J_kine_p.flatten(), atol=1e-5)
    # assert np.allclose(fd.flatten(), grad_J_kine_p.flatten(), atol=1e-5)
    # assert np.allclose(grad.flatten(), fd.flatten(), atol=1e-5)

    # try:
    #     if debug:
    #         print(Fore.RED + "-" * 50)
    #         np.testing.assert_allclose(
    #             grad_J_kine_p,
    #             numeric_grad_Q(forward_Jp, q, q, logTarget, T_star, 1e-6),
    #             atol=3e-6,
    #             rtol=1e-3,
    #         )
    #         print("dldqJ2 ok")
    # except Exception as e:
    #     print("dldqJ2 nok")
    #     print(str(e))

    pin.framesForwardKinematics(rmodel, rmodel.data, q)
    C: pin.SE3 = rmodel.data.oMf[21].actInv(pin.exp6(logTarget))
    grad_q_p_ = -J_cine.T @ Adj.T @ Jlog.T @ J_cine @ dl_dp_ * lambda_

    # try:
    #     if debug:
    #         print(Fore.RED + "-" * 50)
    #         np.testing.assert_allclose(
    #             grad_q_p_,
    #             numeric_grad_Q(forward_Tq, q, q, logTarget, T_star, 1e-6),
    #             atol=3e-6,
    #             rtol=1e-3,
    #         )
    #         print("dldqT ok")
    # except Exception as e:
    #     print("dldqT nok")
    #     print(str(e))

    # f_casadi = casadi_forward(rmodel, tool_id, q_reg, lambda_, dt, T_star.copy())
    # cost, grad_q, grad_logTarget = f_casadi(q.copy(), logTarget.copy().vector)
    temp = Jexp.T @ Jlog.T @ J_cine @ dl_dp_ * lambda_
    # print("python temp", temp)
    # print("casa temp", grad_logTarget)
    # assert np.allclose(grad_logTarget, temp, atol=1e-5)
    pin.framesForwardKinematics(rmodel, rmodel.data, q)
    pin.updateFramePlacement(rmodel, rmodel.data, tool_id)
    Jlog = pin.Jlog6((rmodel.data.oMf[tool_id].actInv(pin.exp6(logTarget))))
    Jexp = pin.Jexp6(logTarget)
    # print("Jexp")
    # print(Jexp)
    # print(
    #     numerical_jacobian(
    #         fexp,
    #         logTarget,
    #     )
    # )

    # try:
    #     if debug:
    #         print(Fore.RED + "-" * 50)
    #         np.testing.assert_allclose(
    #             Jexp,
    #             numerical_jacobian(
    #                 fexp,
    #                 logTarget,
    #             ),
    #             atol=3e-6,
    #             rtol=1e-3,
    #         )
    #         print("Jexp ok")
    # except Exception as e:
    #     print("Jexp nok")
    #     print(str(e))

    # return (
    #     numeric_grad_Q(forward_Tq, q, q, logTarget, T_star, 1e-6)
    #     + numeric_grad_Q(forward_Jp, q, q, logTarget, T_star, 1e-6)
    #     + numeric_grad_Q(forward_Q, q, q, logTarget, T_star, 1e-6)
    #     + dl_dq_next,
    #     temp,
    # )
    if out:
        return (
            outpython,
            grad_q_p_ + grad_J_kine + grad_J_kine_p + dl_dq_next,
            temp,
        )
    return (
        grad_q_p_ + grad_J_kine + grad_J_kine_p + dl_dq_next,
        temp,
    )


def analytical_dldq(q, T_star):
    pin.framesForwardKinematics(rmodel, rmodel.data, q)
    pin.updateFramePlacement(rmodel, rmodel.data, tool_id)
    A = rmodel.data.oMf[tool_id].copy()

    C: pin.SE3 = A.copy().actInv(T_star.copy())
    adj = np.array(C.toActionMatrixInverse()).copy()

    jlog = pin.Jlog6(C)
    log = pin.log6(C).vector
    jac = pin.computeFrameJacobian(rmodel, rmodel.data, q, tool_id, pin.LOCAL)
    return -2 * jac.T @ adj.T @ jlog.T @ log


def test_q_to_QP_cost():
    def finite_difference_gradient(rmodel, q, frame_id, eps=1e-6):
        data = rmodel.createData()
        nq = rmodel.nq
        grad_fd = np.zeros(nq)
        J0 = pin.computeFrameJacobian(rmodel, data, q, frame_id, pin.LOCAL)
        Q0 = J0.T @ J0 + 1e-4 * np.eye(nq)
        J_cost0 = 0.5 * np.trace(Q0)

        for i in range(nq):
            q_pert = q.copy()
            q_pert[i] += eps
            pin.computeFrameJacobian(rmodel, data, q_pert, frame_id, pin.LOCAL)
            J_pert = pin.computeFrameJacobian(rmodel, data, q_pert, frame_id, pin.LOCAL)
            Q_pert = J_pert.T @ J_pert + 1e-4 * np.eye(nq)
            J_cost_pert = 0.5 * np.trace(Q_pert)
            grad_fd[i] = (J_cost_pert - J_cost0) / eps
        return grad_fd

    def analytical_gradient(rmodel, q, frame_id):
        data = rmodel.createData()
        J_cine = pin.computeFrameJacobian(rmodel, data, q, frame_id, pin.LOCAL)
        hessian = compute_frame_hessian(q)
        grad = np.einsum("kl,kjl->j", J_cine, hessian)

        return grad

    grad_fd = finite_difference_gradient(rmodel, q_, tool_id)
    grad_ana = analytical_gradient(rmodel, q_, tool_id)
    np.testing.assert_allclose(grad_ana, grad_fd, rtol=1e3, atol=3e-6)


def test_hessian():
    dt = 1e-6
    q = pin.randomConfiguration(rmodel)
    q1 = pin.randomConfiguration(rmodel)
    Hess = compute_frame_hessian(q).swapaxes(1, 2)
    J = pin.computeFrameJacobian(rmodel, rmodel.data, q, 21, pin.LOCAL)
    J1 = pin.computeFrameJacobian(rmodel, rmodel.data, q + dt * q1, 21, pin.LOCAL)
    np.testing.assert_allclose(Hess @ q1, (J1 - J) / dt, rtol=1e3, atol=3e-6)


def logpos(q, rmodel):
    pin.framesForwardKinematics(rmodel, rmodel.data, q)
    return pin.log6(rmodel.data.oMf[tool_id])


def fd_log(q, rmodel, eps=1e-2):
    for i in range(6):
        qplus = q.copy()
        qminus = q.copy()
        qplus[i] += eps
        qminus[i] -= eps
        print(logpos(qplus, rmodel))
        print(logpos(qminus, rmodel))


def test_all():
    with open("worst_case_inputs.pkl", "rb") as f:
        worst_case = pickle.load(f)

    # q_rel_q = worst_case["worst_q_rel"]["q"]
    # q_rel_T_star = worst_case["worst_q_rel"]["T_star"]
    # q_rel_logTarget = worst_case["worst_q_rel"]["logTarget"]
    # q_rel_q = worst_case["worst_log_rel"]["q"]
    # q_rel_T_star = worst_case["worst_log_rel"]["T_star"]
    # q_rel_logTarget = worst_case["worst_log_rel"]["logTarget"]
    # err = worst_case["worst_log_rel"]["error"]
    # print(err)
    # input()

    # q_rel_q = np.random.randn(*q_rel_q.shape)
    # q_rel_T_star = pin.SE3.Random()
    # q_rel_logTarget = pin.Motion.Random()
    q_rel_q, q_rel_logTarget, q_rel_T_star = worst_case
    # q_rel_q[2] = 0
    # q_rel_logTarget.vector *= 100

    f_casadi = casadi_forward(rmodel, tool_id, q_reg, lambda_, dt, q_rel_T_star.copy())

    # q0 = np.random.randn(rmodel.nq)
    # logTarget0 = np.zeros(6)

    viz = Viewer(rmodel, gmodel, vmodel, True)
    # os.system("clear")
    pin.framesForwardKinematics(rmodel, rmodel.data, q_rel_q)
    print("placement", rmodel.data.oMf[tool_id])
    print(q_rel_T_star)
    print(q_rel_logTarget)
    print(q_rel_q)
    viz.viz.viewer["start2"].set_object(  # type: ignore
        g.Sphere(0.1),
        g.MeshLambertMaterial(
            color=0x00FFFF, transparent=True, opacity=0.5
        ),  # vert transparent
    )
    viz.viz.viewer["start2"].set_transform(pin.exp6(q_rel_logTarget).homogeneous)

    viz.display(q_rel_q)
    # print("casa err", f_casadi(q_rel_q, q_rel_logTarget.vector))
    cost, grad_q, grad_logTarget = (
        f_casadi(  # grad_q_next, grad_Q, grad_J_cine, grad_qJ
            q_rel_q.copy(), q_rel_logTarget.copy().vector
        )
    )
    # print(grad_q, grad_logTarget, grad_q_next, grad_Q, grad_J_cine, grad_qJ)
    # print("grad_q_next casa", grad_q_next)
    # print("grad_Q casa", grad_Q)
    # print("grad_J_cine casa", grad_J_cine)
    # print("grad_qJ casa", grad_qJ)
    fd1, fd2 = numeric_gradient_forward(
        forward, q_rel_q.copy(), q_rel_logTarget.copy(), q_rel_T_star.copy(), 1e-6
    )
    out = forward(q_rel_q.copy(), q_rel_logTarget.copy(), q_rel_T_star.copy())
    print("python", out)
    print("casa", cost)
    assert np.allclose(out, cost)
    ana1, ana2 = analytical(
        q_rel_q.copy(), q_rel_logTarget.copy(), q_rel_T_star.copy(), debug=True
    )
    print(Fore.RED + "-" * 50)
    print("grad, fd method on q", fd1)
    print("grad, analitycal method on q", ana1)
    print("grad, casady method on q", grad_q)
    print(Fore.RED + "-" * 50)
    print("grad, fd method on log_motion", fd2)
    print("grad, analitycal method on log_motion", ana2)
    print("grad, casady method on log motion", grad_logTarget)
    print(Fore.RED + "-" * 50)

    input()
    np.testing.assert_allclose(fd1, ana1, rtol=1e-10, atol=3e-60)
    np.testing.assert_allclose(fd2, ana2, atol=3e-60)


def robust_relative_error(analytical, numerical, abs_tol=1e-5):
    # print(analytical)
    # print(numerical)
    # """Erreur relative robuste qui évite la division par des valeurs trop petites"""
    abs_err = np.linalg.norm(numerical - analytical)
    scale = np.maximum(np.linalg.norm(analytical), abs_tol)
    # print(abs_err / scale)
    # input()
    return abs_err / scale


def test_gradient_consistency_plot(n=50, magnitude_threshold=1e-4):
    errors_rel_q = []
    errors_abs_q = []
    errors_rel_log = []
    errors_abs_log = []

    # Nouvelles métriques filtrées
    errors_rel_q_filtered = []
    errors_rel_log_filtered = []

    # Nouvelles métriques robustes
    errors_robust_q = []
    errors_robust_log = []

    # Track worst-case inputs
    worst_rel_q = -np.inf
    worst_rel_log = -np.inf
    worst_q_rel_inputs = None
    worst_log_rel_inputs = None

    for i in tqdm(range(n)):
        logTarget = pin.Motion.Random()
        q_ = pin.randomConfiguration(rmodel)
        T_star = pin.SE3.Random()
        fd1, fd2 = numeric_gradient_forward(
            forward, q_.copy(), logTarget.copy(), T_star.copy(), 1e-5
        )
        f_casadi = casadi_forward(rmodel, tool_id, q_reg, lambda_, dt, T_star.copy())
        cost, grad_q, grad_logTarget = f_casadi(q_.copy(), logTarget.copy().vector)
        ana1, ana2 = analytical(q_.copy(), logTarget.copy(), T_star.copy())

        err_abs_q = np.max(np.abs(fd1 - ana1))
        if err_abs_q > 1e-3:
            print("q")
            print("ana1", ana1)
            print("fd1", fd1)
            print("casa1", grad_q)
            print("motion")
            print("ana2", ana2)
            print("fd2", fd2)
            print("casa2", grad_logTarget)
            with open("worst_case_inputs.pkl", "wb") as f:
                pickle.dump((q_, logTarget, T_star), f)

        err_rel_q = err_abs_q / (np.abs(ana1) + 1e-12)

        err_abs_log = np.max(np.abs(fd2 - ana2))
        err_rel_log = err_abs_log / (np.abs(ana2) + 1e-12)

        # Calcul des erreurs relatives robustes
        err_robust_q = robust_relative_error(ana1, fd1)
        err_robust_log = robust_relative_error(ana2, fd2)

        # Calcul des erreurs relatives filtrées
        # Pour q : ne garder que les coordonnées où |ana1| > threshold
        mask_q = np.abs(ana1) > magnitude_threshold
        if np.any(mask_q):
            err_rel_q_filt = err_rel_q[mask_q]
        else:
            err_rel_q_filt = np.array([])  # Aucune coordonnée significative

        # Pour log : ne garder que les coordonnées où |ana2| > threshold
        mask_log = np.abs(ana2) > magnitude_threshold
        if np.any(mask_log):
            err_rel_log_filt = err_rel_log[mask_log]
        else:
            err_rel_log_filt = np.array([])  # Aucune coordonnée significative

        # Check for worst-case relative errors (utilise robust relative error)
        max_rel_q = np.max(err_robust_q)
        if max_rel_q > worst_rel_q:
            worst_rel_q = max_rel_q
            worst_q_rel_inputs = (q_.copy(), T_star.copy(), logTarget.copy())

        max_rel_log = np.max(err_robust_log)
        if max_rel_log > worst_rel_log:
            worst_rel_log = max_rel_log
            worst_log_rel_inputs = (q_.copy(), T_star.copy(), logTarget.copy())

        errors_abs_q.append(err_abs_q)
        errors_rel_q.append(err_rel_q)
        errors_abs_log.append(err_abs_log)
        errors_rel_log.append(err_rel_log)

        errors_rel_q_filtered.append(err_rel_q_filt)
        errors_rel_log_filtered.append(err_rel_log_filt)

        errors_robust_q.append(err_robust_q)
        errors_robust_log.append(err_robust_log)

    errors_abs_q = np.array(errors_abs_q)
    errors_rel_q = np.array(errors_rel_q)
    errors_abs_log = np.array(errors_abs_log)
    errors_rel_log = np.array(errors_rel_log)
    errors_robust_q = np.array(errors_robust_q)
    errors_robust_log = np.array(errors_robust_log)

    # Concaténer toutes les erreurs filtrées (en excluant les arrays vides)
    all_rel_q_filtered = np.concatenate(
        [arr for arr in errors_rel_q_filtered if len(arr) > 0]
    )
    all_rel_log_filtered = np.concatenate(
        [arr for arr in errors_rel_log_filtered if len(arr) > 0]
    )

    stats = {
        "q_abs": {
            "mean": errors_abs_q.mean(),
            "max": errors_abs_q.max(),
            "min": errors_abs_q.min(),
        },
        "q_rel": {
            "mean": errors_rel_q.mean(),
            "max": errors_rel_q.max(),
            "min": errors_rel_q.min(),
        },
        "log_abs": {
            "mean": errors_abs_log.mean(),
            "max": errors_abs_log.max(),
            "min": errors_abs_log.min(),
        },
        "log_rel": {
            "mean": errors_rel_log.mean(),
            "max": errors_rel_log.max(),
            "min": errors_rel_log.min(),
        },
        "q_rel_filtered": {
            "mean": (
                all_rel_q_filtered.mean() if len(all_rel_q_filtered) > 0 else np.nan
            ),
            "max": all_rel_q_filtered.max() if len(all_rel_q_filtered) > 0 else np.nan,
            "min": all_rel_q_filtered.min() if len(all_rel_q_filtered) > 0 else np.nan,
            "count": len(all_rel_q_filtered),
        },
        "log_rel_filtered": {
            "mean": (
                all_rel_log_filtered.mean() if len(all_rel_log_filtered) > 0 else np.nan
            ),
            "max": (
                all_rel_log_filtered.max() if len(all_rel_log_filtered) > 0 else np.nan
            ),
            "min": (
                all_rel_log_filtered.min() if len(all_rel_log_filtered) > 0 else np.nan
            ),
            "count": len(all_rel_log_filtered),
        },
        "q_robust": {
            "mean": errors_robust_q.mean(),
            "max": errors_robust_q.max(),
            "min": errors_robust_q.min(),
        },
        "log_robust": {
            "mean": errors_robust_log.mean(),
            "max": errors_robust_log.max(),
            "min": errors_robust_log.min(),
        },
    }

    # Plots avec 4 sous-graphiques
    # fig, axes = plt.subplots(1, 2, figsize=(15, 10))

    # # Erreurs relatives (toutes coordonnées)
    # axes[0, 0].boxplot(
    #     [errors_rel_q.flatten(), errors_rel_log.flatten()], labels=["q", "log_motion"]
    # )
    # axes[0, 0].set_title("Erreurs relatives (toutes coordonnées)")
    # axes[0, 0].set_yscale("log")
    # axes[0, 0].set_ylabel("Relative error")

    # # Erreurs relatives filtrées (coordonnées significatives seulement)
    # if len(all_rel_q_filtered) > 0 and len(all_rel_log_filtered) > 0:
    #     axes[0, 1].boxplot(
    #         [all_rel_q_filtered, all_rel_log_filtered], labels=["q", "log_motion"]
    #     )
    #     axes[0, 1].set_title(
    #         f"Erreurs relatives filtrées (|grad| > {magnitude_threshold})"
    #     )
    #     axes[0, 1].set_yscale("log")
    #     axes[0, 1].set_ylabel("Relative error (filtered)")
    # else:
    #     axes[0, 1].text(
    #         0.5,
    #         0.5,
    #         "Pas assez de données\npour les erreurs filtrées",
    #         transform=axes[0, 1].transAxes,
    #         ha="center",
    #         va="center",
    #     )
    #     axes[0, 1].set_title(
    #         f"Erreurs relatives filtrées (|grad| > {magnitude_threshold})"
    #     )

    # Erreurs relatives robustes
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [3, 1]}
    )
    ax1.set_title(
        "Distribution of absolute error between finite difference grad and analytic grad"
    )
    # ax1.title("tests")
    # KDE plots
    sns.kdeplot(
        errors_abs_q.flatten(),
        ax=ax1,
        color="#FF6B6B",
        label="q",
        log_scale=(True, True),
    )
    sns.kdeplot(
        errors_abs_log.flatten(),
        ax=ax1,
        color="#4ECDC4",
        label="log_motion",
        log_scale=(True, True),
    )
    ax1.set_xlabel("Absolute error (norm inf)")
    ax1.set_ylabel("Density (log)")
    ax1.legend()

    # Boxplots
    sns.boxplot(
        data=[errors_abs_q.flatten(), errors_abs_log.flatten()],
        ax=ax2,
        palette=["#FF6B6B", "#4ECDC4"],
    )
    ax2.set_xticks([0, 1], labels=["gradient\n on q", "gradient\n on log_motion"])
    ax2.set_yscale("log")
    ax2.set_ylabel("Absolute error (log scale)")
    ax2.set_title("Distribution (log-log)")

    plt.tight_layout()
    plt.show()
    # Print quelques statistiques utiles
    print(f"\nStatistiques avec seuil de magnitude = {magnitude_threshold}:")
    print(
        f"Coordonnées q significatives: {len(all_rel_q_filtered)} / {errors_rel_q.size}"
    )
    print(
        f"Coordonnées log significatives: {len(all_rel_log_filtered)} / {errors_rel_log.size}"
    )

    if len(all_rel_q_filtered) > 0:
        print(f"Erreur relative max filtrée (q): {all_rel_q_filtered.max():.2e}")
    if len(all_rel_log_filtered) > 0:
        print(f"Erreur relative max filtrée (log): {all_rel_log_filtered.max():.2e}")

    print(f"Erreur relative robuste max (q): {errors_robust_q.max():.2e}")
    print(f"Erreur relative robuste max (log): {errors_robust_log.max():.2e}")

    # Save worst-case inputs to pickle
    worst_case = {
        "worst_q_rel": {
            "error": worst_rel_q,
            "q": worst_q_rel_inputs[0],
            "T_star": worst_q_rel_inputs[1],
            "logTarget": worst_q_rel_inputs[2],
        },
        "worst_log_rel": {
            "error": worst_rel_log,
            "q": worst_log_rel_inputs[0],
            "T_star": worst_log_rel_inputs[1],
            "logTarget": worst_log_rel_inputs[2],
        },
        "threshold_used": magnitude_threshold,
        "filtered_stats": {
            "q_rel_filtered": stats["q_rel_filtered"],
            "log_rel_filtered": stats["log_rel_filtered"],
        },
    }
    with open("worst_case_inputs.pkl", "wb") as f:
        pickle.dump(worst_case, f)
    print(worst_case)

    return stats


def casadi_test_gradient_consistency_plot(n=50, magnitude_threshold=1e-4):
    errors_rel_q = []
    errors_abs_q = []
    errors_rel_log = []
    errors_abs_log = []

    # Nouvelles métriques filtrées
    errors_rel_q_filtered = []
    errors_rel_log_filtered = []

    # Nouvelles métriques robustes
    errors_robust_q = []
    errors_robust_log = []

    # Track worst-case inputs
    worst_rel_q = -np.inf
    worst_rel_log = -np.inf
    worst_q_rel_inputs = None
    worst_log_rel_inputs = None

    for i in tqdm(range(n)):
        logTarget = pin.Motion.Random()
        q_ = pin.randomConfiguration(rmodel)
        T_star = pin.SE3.Random()
        fd1, fd2 = numeric_gradient_forward(
            forward, q_.copy(), logTarget.copy(), T_star.copy(), 1e-5
        )
        f_casadi = casadi_forward(rmodel, tool_id, q_reg, lambda_, dt, T_star.copy())
        cost, grad_q, grad_logTarget = f_casadi(q_.copy(), logTarget.copy().vector)
        ana1, ana2 = analytical(q_.copy(), logTarget.copy(), T_star.copy())

        err_abs_q = np.max(np.abs(fd1 - grad_q))
        if err_abs_q > 1e-3:
            print("q")
            print("ana1", ana1)
            print("fd1", fd1)
            print("casa1", grad_q)
            print("motion")
            print("ana2", ana2)
            print("fd2", fd2)
            print("casa2", grad_logTarget)
            with open("worst_case_inputs.pkl", "wb") as f:
                pickle.dump((q_, logTarget, T_star), f)

        err_rel_q = err_abs_q / (np.abs(ana1) + 1e-12)

        err_abs_log = np.max(np.abs(fd2 - grad_logTarget))
        err_rel_log = err_abs_log / (np.abs(ana2) + 1e-12)

        # Calcul des erreurs relatives robustes
        err_robust_q = robust_relative_error(ana1, fd1)
        err_robust_log = robust_relative_error(ana2, fd2)

        # Calcul des erreurs relatives filtrées
        # Pour q : ne garder que les coordonnées où |ana1| > threshold
        mask_q = np.abs(ana1) > magnitude_threshold
        if np.any(mask_q):
            err_rel_q_filt = err_rel_q[mask_q]
        else:
            err_rel_q_filt = np.array([])  # Aucune coordonnée significative

        # Pour log : ne garder que les coordonnées où |ana2| > threshold
        mask_log = np.abs(ana2) > magnitude_threshold
        if np.any(mask_log):
            err_rel_log_filt = err_rel_log[mask_log]
        else:
            err_rel_log_filt = np.array([])  # Aucune coordonnée significative

        # Check for worst-case relative errors (utilise robust relative error)
        max_rel_q = np.max(err_robust_q)
        if max_rel_q > worst_rel_q:
            worst_rel_q = max_rel_q
            worst_q_rel_inputs = (q_.copy(), T_star.copy(), logTarget.copy())

        max_rel_log = np.max(err_robust_log)
        if max_rel_log > worst_rel_log:
            worst_rel_log = max_rel_log
            worst_log_rel_inputs = (q_.copy(), T_star.copy(), logTarget.copy())

        errors_abs_q.append(err_abs_q)
        errors_rel_q.append(err_rel_q)
        errors_abs_log.append(err_abs_log)
        errors_rel_log.append(err_rel_log)

        errors_rel_q_filtered.append(err_rel_q_filt)
        errors_rel_log_filtered.append(err_rel_log_filt)

        errors_robust_q.append(err_robust_q)
        errors_robust_log.append(err_robust_log)

    errors_abs_q = np.array(errors_abs_q)
    errors_rel_q = np.array(errors_rel_q)
    errors_abs_log = np.array(errors_abs_log)
    errors_rel_log = np.array(errors_rel_log)
    errors_robust_q = np.array(errors_robust_q)
    errors_robust_log = np.array(errors_robust_log)

    # Concaténer toutes les erreurs filtrées (en excluant les arrays vides)
    all_rel_q_filtered = np.concatenate(
        [arr for arr in errors_rel_q_filtered if len(arr) > 0]
    )
    all_rel_log_filtered = np.concatenate(
        [arr for arr in errors_rel_log_filtered if len(arr) > 0]
    )

    stats = {
        "q_abs": {
            "mean": errors_abs_q.mean(),
            "max": errors_abs_q.max(),
            "min": errors_abs_q.min(),
        },
        "q_rel": {
            "mean": errors_rel_q.mean(),
            "max": errors_rel_q.max(),
            "min": errors_rel_q.min(),
        },
        "log_abs": {
            "mean": errors_abs_log.mean(),
            "max": errors_abs_log.max(),
            "min": errors_abs_log.min(),
        },
        "log_rel": {
            "mean": errors_rel_log.mean(),
            "max": errors_rel_log.max(),
            "min": errors_rel_log.min(),
        },
        "q_rel_filtered": {
            "mean": (
                all_rel_q_filtered.mean() if len(all_rel_q_filtered) > 0 else np.nan
            ),
            "max": all_rel_q_filtered.max() if len(all_rel_q_filtered) > 0 else np.nan,
            "min": all_rel_q_filtered.min() if len(all_rel_q_filtered) > 0 else np.nan,
            "count": len(all_rel_q_filtered),
        },
        "log_rel_filtered": {
            "mean": (
                all_rel_log_filtered.mean() if len(all_rel_log_filtered) > 0 else np.nan
            ),
            "max": (
                all_rel_log_filtered.max() if len(all_rel_log_filtered) > 0 else np.nan
            ),
            "min": (
                all_rel_log_filtered.min() if len(all_rel_log_filtered) > 0 else np.nan
            ),
            "count": len(all_rel_log_filtered),
        },
        "q_robust": {
            "mean": errors_robust_q.mean(),
            "max": errors_robust_q.max(),
            "min": errors_robust_q.min(),
        },
        "log_robust": {
            "mean": errors_robust_log.mean(),
            "max": errors_robust_log.max(),
            "min": errors_robust_log.min(),
        },
    }

    # Plots avec 4 sous-graphiques
    # fig, axes = plt.subplots(1, 2, figsize=(15, 10))

    # # Erreurs relatives (toutes coordonnées)
    # axes[0, 0].boxplot(
    #     [errors_rel_q.flatten(), errors_rel_log.flatten()], labels=["q", "log_motion"]
    # )
    # axes[0, 0].set_title("Erreurs relatives (toutes coordonnées)")
    # axes[0, 0].set_yscale("log")
    # axes[0, 0].set_ylabel("Relative error")

    # # Erreurs relatives filtrées (coordonnées significatives seulement)
    # if len(all_rel_q_filtered) > 0 and len(all_rel_log_filtered) > 0:
    #     axes[0, 1].boxplot(
    #         [all_rel_q_filtered, all_rel_log_filtered], labels=["q", "log_motion"]
    #     )
    #     axes[0, 1].set_title(
    #         f"Erreurs relatives filtrées (|grad| > {magnitude_threshold})"
    #     )
    #     axes[0, 1].set_yscale("log")
    #     axes[0, 1].set_ylabel("Relative error (filtered)")
    # else:
    #     axes[0, 1].text(
    #         0.5,
    #         0.5,
    #         "Pas assez de données\npour les erreurs filtrées",
    #         transform=axes[0, 1].transAxes,
    #         ha="center",
    #         va="center",
    #     )
    #     axes[0, 1].set_title(
    #         f"Erreurs relatives filtrées (|grad| > {magnitude_threshold})"
    #     )

    # Erreurs relatives robustes
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [3, 1]}
    )
    ax1.set_title(
        "Distribution of absolute error between finite difference grad and casadi grad"
    )
    # ax1.title("tests")
    # KDE plots
    sns.kdeplot(
        errors_abs_q.flatten(),
        ax=ax1,
        color="#FF6B6B",
        label="q",
        log_scale=(True, True),
    )
    sns.kdeplot(
        errors_abs_log.flatten(),
        ax=ax1,
        color="#4ECDC4",
        label="log_motion",
        log_scale=(True, True),
    )
    ax1.set_xlabel("Absolute error (norm inf)")
    ax1.set_ylabel("Density (log)")
    ax1.legend()

    # Boxplots
    sns.boxplot(
        data=[errors_abs_q.flatten(), errors_abs_log.flatten()],
        ax=ax2,
        palette=["#FF6B6B", "#4ECDC4"],
    )
    ax2.set_xticks([0, 1], labels=["gradient\n on q", "gradient\n on log_motion"])
    ax2.set_yscale("log")
    ax2.set_ylabel("Absolute error (log scale)")
    ax2.set_title("Distribution (log-log)")

    plt.tight_layout()
    plt.show()
    # Print quelques statistiques utiles
    print(f"\nStatistiques avec seuil de magnitude = {magnitude_threshold}:")
    print(
        f"Coordonnées q significatives: {len(all_rel_q_filtered)} / {errors_rel_q.size}"
    )
    print(
        f"Coordonnées log significatives: {len(all_rel_log_filtered)} / {errors_rel_log.size}"
    )

    if len(all_rel_q_filtered) > 0:
        print(f"Erreur relative max filtrée (q): {all_rel_q_filtered.max():.2e}")
    if len(all_rel_log_filtered) > 0:
        print(f"Erreur relative max filtrée (log): {all_rel_log_filtered.max():.2e}")

    print(f"Erreur relative robuste max (q): {errors_robust_q.max():.2e}")
    print(f"Erreur relative robuste max (log): {errors_robust_log.max():.2e}")

    # Save worst-case inputs to pickle
    worst_case = {
        "worst_q_rel": {
            "error": worst_rel_q,
            "q": worst_q_rel_inputs[0],
            "T_star": worst_q_rel_inputs[1],
            "logTarget": worst_q_rel_inputs[2],
        },
        "worst_log_rel": {
            "error": worst_rel_log,
            "q": worst_log_rel_inputs[0],
            "T_star": worst_log_rel_inputs[1],
            "logTarget": worst_log_rel_inputs[2],
        },
        "threshold_used": magnitude_threshold,
        "filtered_stats": {
            "q_rel_filtered": stats["q_rel_filtered"],
            "log_rel_filtered": stats["log_rel_filtered"],
        },
    }
    with open("worst_case_inputs.pkl", "wb") as f:
        pickle.dump(worst_case, f)
    print(worst_case)

    return stats


def robust_relative_error(analytical, numerical, abs_tol=1e-5):
    """Erreur relative robuste qui évite la division par des valeurs trop petites"""
    abs_err = np.linalg.norm(numerical - analytical)
    scale = np.maximum(np.linalg.norm(analytical), abs_tol)
    return abs_err / scale


def test_gradient_consistency_plot3(n=50, magnitude_threshold=1e-4):
    # Erreurs pour les trois comparaisons
    errors_fd_vs_ana_q = []
    errors_fd_vs_ana_log = []

    errors_ana_vs_casa_q = []
    errors_ana_vs_casa_log = []

    errors_casa_vs_fd_q = []
    errors_casa_vs_fd_log = []

    # Track worst-case inputs
    worst_fd_ana_q = -np.inf
    worst_fd_ana_log = -np.inf
    worst_fd_ana_inputs = None

    for i in tqdm(range(n)):
        logTarget = pin.Motion.Random()
        q_ = pin.randomConfiguration(rmodel)
        T_star = pin.SE3.Random()

        # Calcul des trois gradients
        outfd, fd1, fd2 = numeric_gradient_forward(
            forward, q_.copy(), logTarget.copy(), T_star.copy(), 1e-5, out=True
        )
        f_casadi = casadi_forward(rmodel, tool_id, q_reg, lambda_, dt, T_star.copy())
        out_casadi, grad_q, grad_logTarget = f_casadi(
            q_.copy(), logTarget.copy().vector
        )
        out_ana, ana1, ana2 = analytical(
            q_.copy(), logTarget.copy(), T_star.copy(), out=True
        )

        assert np.allclose(out_ana, out_casadi)
        assert np.allclose(outfd, out_casadi)
        assert np.allclose(out_ana, outfd)

        # Comparaison 1: FD vs Analytical
        err_fd_ana_q = np.max(np.abs(fd1 - ana1))
        err_fd_ana_log = np.max(np.abs(fd2 - ana2))

        # Comparaison 2: Analytical vs CasADi
        err_ana_casa_q = np.max(np.abs(ana1 - grad_q))
        err_ana_casa_log = np.max(np.abs(ana2 - grad_logTarget))

        # Comparaison 3: CasADi vs FD
        err_casa_fd_q = np.max(np.abs(grad_q - fd1))
        err_casa_fd_log = np.max(np.abs(grad_logTarget - fd2))

        # Debug si erreur importante
        if err_fd_ana_q > 1e-0:
            print("q - Erreur importante détectée")
            print("ana1", ana1)
            print("fd1", fd1)
            print("casa1", grad_q)
            print("motion")
            print("ana2", ana2)
            print("fd2", fd2)
            print("casa2", grad_logTarget)
            with open("worst_case_inputs.pkl", "wb") as f:
                pickle.dump((q_, logTarget, T_star), f)

        # Track worst case
        if err_fd_ana_q > worst_fd_ana_q:
            worst_fd_ana_q = err_fd_ana_q
            worst_fd_ana_inputs = (q_.copy(), T_star.copy(), logTarget.copy())

        # Stockage des erreurs
        errors_fd_vs_ana_q.append(err_fd_ana_q)
        errors_fd_vs_ana_log.append(err_fd_ana_log)

        errors_ana_vs_casa_q.append(err_ana_casa_q)
        errors_ana_vs_casa_log.append(err_ana_casa_log)

        errors_casa_vs_fd_q.append(err_casa_fd_q)
        errors_casa_vs_fd_log.append(err_casa_fd_log)

    # Conversion en arrays
    errors_fd_vs_ana_q = np.array(errors_fd_vs_ana_q)
    errors_fd_vs_ana_log = np.array(errors_fd_vs_ana_log)
    errors_ana_vs_casa_q = np.array(errors_ana_vs_casa_q)
    errors_ana_vs_casa_log = np.array(errors_ana_vs_casa_log)
    errors_casa_vs_fd_q = np.array(errors_casa_vs_fd_q)
    errors_casa_vs_fd_log = np.array(errors_casa_vs_fd_log)

    # Création des statistiques
    stats = {
        "fd_vs_ana": {
            "q": {
                "mean": errors_fd_vs_ana_q.mean(),
                "max": errors_fd_vs_ana_q.max(),
                "min": errors_fd_vs_ana_q.min(),
            },
            "log": {
                "mean": errors_fd_vs_ana_log.mean(),
                "max": errors_fd_vs_ana_log.max(),
                "min": errors_fd_vs_ana_log.min(),
            },
        },
        "ana_vs_casa": {
            "q": {
                "mean": errors_ana_vs_casa_q.mean(),
                "max": errors_ana_vs_casa_q.max(),
                "min": errors_ana_vs_casa_q.min(),
            },
            "log": {
                "mean": errors_ana_vs_casa_log.mean(),
                "max": errors_ana_vs_casa_log.max(),
                "min": errors_ana_vs_casa_log.min(),
            },
        },
        "casa_vs_fd": {
            "q": {
                "mean": errors_casa_vs_fd_q.mean(),
                "max": errors_casa_vs_fd_q.max(),
                "min": errors_casa_vs_fd_q.min(),
            },
            "log": {
                "mean": errors_casa_vs_fd_log.mean(),
                "max": errors_casa_vs_fd_log.max(),
                "min": errors_casa_vs_fd_log.min(),
            },
        },
    }

    # Création des 3 boxplots
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))

    # Seuil acceptable: sqrt(1e-5)
    threshold = np.sqrt(1e-5)

    # Titre global
    fig.suptitle(
        "Comparison of gradient computation methods through a QP block\n"
        "Centered finite differences with increment 1e-5. Acceptable threshold: √(1e-5) ≈ 3.16e-3",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Boxplot 1: FD vs Analytical
    axes[0].boxplot(
        [errors_fd_vs_ana_q, errors_fd_vs_ana_log],
        labels=["gradient on q", "gradient on log_motion"],
        patch_artist=True,
        boxprops=dict(facecolor="#FF6B6B", alpha=0.7),
    )
    axes[0].axhline(
        y=threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold: √(1e-5) ≈ {threshold:.2e}",
    )
    axes[0].set_title(
        "Finite Differences vs Analytical", fontsize=12, fontweight="bold"
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Absolute error (log scale)", fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)

    # Boxplot 2: Analytical vs CasADi
    axes[1].boxplot(
        [errors_ana_vs_casa_q, errors_ana_vs_casa_log],
        labels=["gradient on q", "gradient on log_motion"],
        patch_artist=True,
        boxprops=dict(facecolor="#4ECDC4", alpha=0.7),
    )
    # axes[1].axhline(
    #     y=threshold,
    #     color="red",
    #     linestyle="--",
    #     linewidth=2,
    #     label=f"Threshold: √(1e-5) ≈ {threshold:.2e}",
    # )
    axes[1].set_title("Analytical vs CasADi", fontsize=12, fontweight="bold")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Absolute error (log scale)", fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8)

    # Boxplot 3: CasADi vs FD
    axes[2].boxplot(
        [errors_casa_vs_fd_q, errors_casa_vs_fd_log],
        labels=["gradient on q", "gradient on log_motion"],
        patch_artist=True,
        boxprops=dict(facecolor="#95E1D3", alpha=0.7),
    )
    axes[2].axhline(
        y=threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold: √(1e-5) ≈ {threshold:.2e}",
    )
    axes[2].set_title("CasADi vs Finite Differences", fontsize=12, fontweight="bold")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Absolute error (log scale)", fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()

    # Affichage des statistiques
    print("\n" + "=" * 80)
    print("STATISTIQUES DE COMPARAISON DES GRADIENTS")
    print("=" * 80)

    print("\n1. Finite Differences vs Analytical:")
    print(
        f"   gradient sur q   - Max: {errors_fd_vs_ana_q.max():.2e}, Mean: {errors_fd_vs_ana_q.mean():.2e}"
    )
    print(
        f"   gradient sur log - Max: {errors_fd_vs_ana_log.max():.2e}, Mean: {errors_fd_vs_ana_log.mean():.2e}"
    )

    print("\n2. Analytical vs CasADi:")
    print(
        f"   gradient sur q   - Max: {errors_ana_vs_casa_q.max():.2e}, Mean: {errors_ana_vs_casa_q.mean():.2e}"
    )
    print(
        f"   gradient sur log - Max: {errors_ana_vs_casa_log.max():.2e}, Mean: {errors_ana_vs_casa_log.mean():.2e}"
    )

    print("\n3. CasADi vs Finite Differences:")
    print(
        f"   gradient sur q   - Max: {errors_casa_vs_fd_q.max():.2e}, Mean: {errors_casa_vs_fd_q.mean():.2e}"
    )
    print(
        f"   gradient sur log - Max: {errors_casa_vs_fd_log.max():.2e}, Mean: {errors_casa_vs_fd_log.mean():.2e}"
    )
    print("=" * 80)

    # Sauvegarde du pire cas
    worst_case = {
        "worst_fd_ana": {
            "error_q": worst_fd_ana_q,
            "q": worst_fd_ana_inputs[0],
            "T_star": worst_fd_ana_inputs[1],
            "logTarget": worst_fd_ana_inputs[2],
        },
        "stats": stats,
    }
    with open("worst_case_inputs.pkl", "wb") as f:
        pickle.dump(worst_case, f)

    return stats


# test_all()
# stats = test_gradient_consistency_plot3(n=50_000)
# stats = casadi_test_gradient_consistency_plot(n=5000)
# exit()
# print(stats)
# exit()


q_reg = 1e-2
bound = -1000
kinematic_workspace = tartempion.KinematicsWorkspace()
workspace = tartempion.QPworkspace()
workspace.set_q_reg(q_reg)
workspace.set_bound(bound)
workspace.set_lambda(-1)
workspace.set_L1(0.00)
workspace.set_rot_w(1.0)
robot = erd.load("ur5")
rmodel, gmodel, vmodel = robot.model, robot.collision_model, robot.visual_model
rmodel.data = rmodel.createData()
tool_id = 21
workspace.set_tool_id(tool_id)
seq_len = 1000
dt = 1e-2
batch_size = 40
eq_dim = 1
n_threads = 20
os.environ["OMP_PROC_BIND"] = "spread"

workspace.set_tool_id(tool_id)
rmodel.data = rmodel.createData()
np.random.seed(22)
eq_dim = 1
p_np = np.zeros((batch_size, 6)).astype(np.float64)
p_np[:, -1] = 1
A_np = np.zeros((batch_size * seq_len, eq_dim, 6)).astype(np.float64)
b_np = np.zeros((batch_size, seq_len, 1)).astype(np.float64)
states_init = np.random.randn(*(batch_size, rmodel.nq)).astype(np.float64)
target = [pin.SE3.Random() for _ in range(batch_size)]


def forward_kine(p):
    return tartempion.forward_pass(
        workspace,
        np.tile(p[:, np.newaxis, :], (1, seq_len, 1)),
        A_np * 0,
        b_np * 0,
        states_init,
        rmodel,
        n_threads,
        target,
        dt,
    )


def forward_kine3(p, states_init, target):
    return tartempion.forward_pass(
        workspace,
        p,
        A_np * 0,
        b_np * 0,
        states_init,
        rmodel,
        n_threads,
        target,
        dt,
    )


def forward_kine2(p):
    return tartempion.forward_pass(
        workspace,
        np.tile(p[:, :, :], (1, 1, 1)),
        A_np * 0,
        b_np * 0,
        states_init,
        rmodel,
        n_threads,
        target,
        dt,
    )


def fd_dLdq(func, q_initial, epsilon=1e-5):
    nq = q_initial.shape[0]
    grad = np.zeros(nq)

    for i in range(nq):
        q_plus = q_initial.copy()
        q_minus = q_initial.copy()
        q_plus[i] += epsilon
        q_minus[i] -= epsilon
        q_plus = np.tile(q_plus[np.newaxis, :], (1, 1))
        q_minus = np.tile(q_minus[np.newaxis, :], (1, 1))
        f_plus = func(q_plus)
        f_minus = func(q_minus)
        grad[i] = (f_plus - f_minus) / (2 * epsilon)
    return grad


def fd_dLdq2(func, q_initial, epsilon=1e-5):
    m, n = q_initial.shape
    f0 = func(q_initial[np.newaxis, :, :])
    f0 = np.asarray(f0)

    grad_shape = f0.shape + (m, n)
    grad = np.zeros(grad_shape)

    for i in tqdm(range(m)):
        if i % 50 == 0:
            for j in tqdm(range(n)):
                q_plus = q_initial.copy()
                q_minus = q_initial.copy()
                q_plus[i, j] += epsilon
                q_minus[i, j] -= epsilon

                f_plus = np.asarray(func(q_plus[np.newaxis, :, :]))
                f_minus = np.asarray(func(q_minus[np.newaxis, :, :]))

                grad[:, i, j] = (f_plus - f_minus) / (2 * epsilon)

    return grad


p_np = torch.load("p.pth", weights_only=False).cpu().detach().numpy()
states_init = torch.load("q.pth", weights_only=False)
target = torch.load("target.pth", weights_only=False)

# print(target)
# input()
asks = [pin.exp6(p_np[i, 0]) for i in range(len(p_np))]
# print(p_np)
# print(asks)
# input()
# print(states_init)

for i in range(len(p_np)):
    if i == 31:
        print("ask", asks[i])
        print("q0", states_init[i])
        print("T*", target[i])
# input()
# p_np = p_np[:2]
# states_init = states_init[:2]
# target = target[:2]
# batch_size = 2

p_np = p_np[31][np.newaxis, :, :]
target = [target[31]]
states_init = states_init[31][np.newaxis, :]


print(p_np)
print(target)
print(states_init)

batch_size = 1

p_np = p_np * 0.8
# p_np[1] = p_np[0]
n = 0
histerra = []
histerrr = []
errs = []
for i in tqdm(range(n)):
    # states_init = np.random.randn(*(batch_size, rmodel.nq)).astype(np.float64)
    # target = [pin.SE3.Random() for _ in range(batch_size)]
    # for i in range(p_np.shape[0]):
    #     target[i] = pin.exp6(pin.Motion(p_np[i, 0]))
    # pred = []
    # for i in range(len(target)):
    #     xi = 1e-3 * np.random.randn(6)
    #     dT = pin.exp6(xi)
    #     pred.append(pin.log6(target[i] * dT))
    # p_np = np.tile(np.array(pred)[:, np.newaxis, :], (1, seq_len, 1))
    # p_np = np.random.random((batch_size, 6))
    print("forward pass")
    out = forward_kine3(p_np, states_init, target)
    print(out)
    print("out ok")
    grad_output = np.zeros((batch_size, seq_len, 2 * 9 + 1))
    print("backward pass")
    backward = np.array(
        tartempion.backward_pass(
            workspace,
            rmodel,
            grad_output,
            n_threads,
            grad_output.shape[0],
        )
    )
    arr = np.array(workspace.get_q())
    print(arr.shape)
    print("backward ok")
    p_grad = np.array(workspace.grad_p())
    p_grad = np.reshape(p_grad, (batch_size, seq_len, rmodel.nq))
    norms = np.linalg.norm(p_grad.reshape(batch_size, -1), axis=1)
    i_max = np.argmax(norms)
    fd = fd_dLdq2(forward_kine2, p_np[i_max], 1e-6)
    ana = p_grad
    # print(p_grad.sum(1))
    # print("1", p_grad[i_max, 100, :])
    # print("2", p_grad[i_max, 200, :])
    # print("3", p_grad[i_max, 300, :])
    # print("320", p_grad[i_max, 320, :])
    # print("4", p_grad[i_max, 400, :])

    # import viewer

    # viz = viewer.Viewer(rmodel, gmodel, vmodel, True)
    # viz.viz.viewer["start"].set_object(  # type: ignore
    #     g.Sphere(0.1),
    #     g.MeshLambertMaterial(
    #         color=0x0000FF, transparent=True, opacity=0.5
    #     ),  # vert transparent
    # )
    # viz.viz.viewer["start2"].set_object(  # type: ignore
    #     g.Sphere(0.1),
    #     g.MeshLambertMaterial(
    #         color=0x00FFFF, transparent=True, opacity=0.5
    #     ),  # vert transparent
    # )

    # viz.viz.viewer["start"].set_transform(target[i_max].homogeneous)
    # viz.viz.viewer["start2"].set_transform(pin.exp6(p_np[i_max, 0]).homogeneous)

    # print("target", target[i_max])
    # print("state init", states_init[i_max])
    # print("p_np SE3", pin.exp6(pin.Motion(p_np[i_max, 0])))
    # print(
    #     "p_np motion", pin.exp6(pin.log6(pin.exp6(pin.Motion(p_np[i_max, 0]))).vector)
    # )
    # print("p_np", p_np[0, 0])
    # print("p_np motion", pin.log6(pin.exp6(pin.Motion(p_np[i_max, 0]))).vector)
    cond = []
    condQ = []
    for plot_time in range(0, arr.shape[1]):
        pass
        # if plot_time > 290:
        #     input()
        # J = pin.computeFrameJacobian(
        #     rmodel, rmodel.data, arr[i_max, plot_time], tool_id, pin.LOCAL
        # )
        # cond.append(np.linalg.cond(J))
        # condQ.append(np.linalg.cond(J.T @ J + q_reg * np.identity(rmodel.nq)))
        # pin.framesForwardKinematics(rmodel, rmodel.data, arr[i_max, plot_time])

        # errs.append(
        #     np.sum(
        #         np.square(
        #             pin.log6(
        #                 rmodel.data.oMf[tool_id].actInv(
        #                     pin.exp6(pin.Motion(p_np[i_max, 0]))
        #                 )
        #             ).vector
        #         )
        #     )
        # )
        # viz.display(arr[i_max, plot_time])
        # time.sleep(dt / 10)

    fig, ax1 = plt.subplots()
    ax1.plot(np.linalg.norm(p_grad[i_max, :, :], axis=1), label="||p_grad||")
    ax1.set_yscale("log")
    ax1.plot(cond, "r-", label="cond Jac")
    ax1.plot(condQ, "g-", label="cond Q")
    ax1.plot(np.abs(p_grad - fd)[0, :, :].mean(-1), label="absolute error")
    ax1.plot(np.linalg.norm(fd[0, :, :], axis=1), label="||fd||")
    ax1.legend(loc="lower left")

    v1 = p_grad[0]
    v2 = fd[0]
    cosalign = np.einsum("ij,ij->i", v1, v2) / (
        np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    )

    ax2 = ax1.twinx()
    ax2.plot(cosalign, "r-", label="cosalign")
    ax2.legend()

    plt.show()

if n != 0:
    times = np.arange(len(errs)) * dt

    plt.plot(times, errs)
    plt.title(
        "IK Error Evolution with nonreachable Target and with Analytic Gradients not Matching Finite-Difference Gradient"
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Squared Norm of Log Error")
    plt.show()

# if n != 0:
#     plt.plot(histerrr)
#     plt.yscale("log")
#     plt.show()


seq_len = 1500
batch_size = 1

n = 500000
histerra = []
histerrr = []
p_np = p_np[0, 0, :][np.newaxis, :]
for ii in tqdm(range(n)):
    states_init = np.random.randn(*(batch_size, rmodel.nq)).astype(np.float64)
    target = [pin.SE3.Random() for _ in range(batch_size)]
    pred = []
    for i in range(len(target)):
        xi = 1e-5 * np.random.randn(6)
        dT = pin.exp6(xi)
        pred.append(pin.log6(target[i] * dT))
    p_np = np.array(pred)
    normalizer = tartempion.Normalizer()
    reach = 0.7
    reach_eps = 0.01
    p_np = np.array(normalizer.normalize(p_np, reach, 1e-8))
    p_np = p_np.reshape((batch_size, 6))

    # p_np = np.random.random((batch_size, 6))
    # xyz = p_np[:, :3]  # premières composantes
    # norm = np.linalg.norm(xyz, axis=1, keepdims=True)
    # scale = np.minimum(0.8 / norm, 1.0)
    # p_np[:, :3] = xyz * scale
    xi = 5e-4 * np.random.randn(6)
    dT = pin.exp6(xi)
    target = [pin.exp6(p_np[0]) * dT]

    # data = np.load("cosalign_failure_debug2.npz")
    # # accéder aux arrays
    # p_np = data["p_np"]
    # target = data["target"]
    # target = [pin.SE3(target[0])]
    # states_init = data["states_init"]
    # print(p_np.shape, target.shape, states_init.shape)
    if np.linalg.norm(pin.exp6(p_np[0, :]).translation) > reach + reach_eps:
        print(np.linalg.norm(pin.exp6(p_np[0, :]).translation))
        raise
        continue
    # if np.linalg.norm(target[0].translation) > reach:
    #     continue

    fd = fd_dLdq2(forward_kine2, np.tile(p_np[:, :], (seq_len, 1)), 1e-6)
    out = forward_kine(p_np)
    print("out", out)
    print("time", ii)
    grad_output = np.zeros((batch_size, seq_len, 2 * 9 + 1))
    backward = np.array(
        tartempion.backward_pass(
            workspace,
            rmodel,
            grad_output,
            n_threads,
            grad_output.shape[0],
        )
    )
    p_grad = np.array(workspace.grad_p())
    p_grad = np.reshape(p_grad, (batch_size, seq_len, rmodel.nq))
    # fd1, fd2 = numeric_gradient_forward(
    #     forward,
    #     states_init[0].copy(),
    #     pin.Motion(p_np[0].copy()),
    #     target[0].copy(),
    #     1e-6,
    # )
    ana = p_grad
    fd = fd
    max_abs_error = np.max(np.abs(ana.flatten() - fd.flatten()))
    max_rel_error = np.max(
        np.abs(p_grad.flatten() - fd.flatten())
        / (np.maximum(np.abs(fd.flatten()), 1e-5))
    )
    histerra.append(max_abs_error)
    histerrr.append(max_rel_error)
    # print("ana python", analytical(states_init[0], p_np[0], target[0], debug=True)[1])
    # print("ana cpp", p_grad)
    # print("fd cpp", fd)
    # input()
    # print("fd python", fd2)
    # print("out cpp", out)
    # print("out python", forward(states_init[0], p_np[0], target[0]))
    # print(max_rel_error)
    # print(max_abs_error)
    # if max_rel_error > 1e-2:
    # print(p_grad)
    # print(fd)
    # print("1", p_grad[:, 100, :])
    # print(fd[:, 100, :])
    # print("2", p_grad[:, 200, :])
    # print(fd[:, 200, :])
    # print("3", p_grad[:, 300, :])
    # print(fd[:, 300, :])
    # print("320", p_grad[:, 320, :])
    # print(fd[:, 320, :])
    # print("4", p_grad[:, 400, :])
    # print(fd[:, 400, :])

    # print(np.abs(p_grad - fd) / np.abs(fd))
    if False:
        viz = viewer.Viewer(rmodel, gmodel, vmodel, True)
        viz.viz.viewer["start"].set_object(  # type: ignore
            g.Sphere(0.1),
            g.MeshLambertMaterial(
                color=0x0000FF, transparent=True, opacity=0.5
            ),  # vert transparent
        )
        viz.viz.viewer["start2"].set_object(  # type: ignore
            g.Sphere(0.1),
            g.MeshLambertMaterial(
                color=0x00FFFF, transparent=True, opacity=0.5
            ),  # vert transparent
        )
        arr = np.array(workspace.get_q())
        i_max = 0
        viz.viz.viewer["start"].set_transform(target[i_max].homogeneous)
        viz.viz.viewer["start2"].set_transform(pin.exp6(p_np[i_max, :]).homogeneous)

        print("target", target[i_max])
        print("state init", states_init[i_max])
        print("p_np", p_np[i_max, :])
        for epo in range(5):
            for plot_time in range(0, arr.shape[1]):
                # if epo == 0 and abs(plot_time - 90) < 10:
                #     input()
                viz.display(arr[i_max, plot_time])
                time.sleep(dt / 10)
    fig, ax1 = plt.subplots()

    # courbe ||p_grad||
    ax1.plot(np.linalg.norm(p_grad[0, :, :], axis=1), label="||p_grad||")
    ax1.set_yscale("log")
    ax1.legend()

    # calcul cosalign
    v1 = p_grad[0]
    v2 = fd[0]
    cosalign = np.einsum("ij,ij->i", v1, v2) / (
        np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    )

    # indices où fd est non nul
    mask = np.linalg.norm(v2, axis=1) > 1e-12
    idx_nonzero = np.where(mask)[0]

    # scatter (croix) sur ax1, même échelle que ||p_grad||
    ax1.scatter(
        idx_nonzero,
        np.linalg.norm(v2[idx_nonzero], axis=1),
        marker="x",
        color="red",
        zorder=5,
        label="fd non nul",
    )

    # annotation cosalign autour des croix
    for i in idx_nonzero:
        ax1.annotate(
            f"{cosalign[i]:.2f}",
            (i, np.linalg.norm(v2[i])),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            color="blue" if cosalign[i] > 0.98 else "red",
        )

    # plt.show()

    threshold = 0.98
    cosalign_nonzero = cosalign[idx_nonzero]
    bad_idx = idx_nonzero[cosalign_nonzero < threshold]
    if np.any(cosalign_nonzero < threshold):
        print(fd)
        print(p_grad)
        mask = fd != 0
        fd_sum_nonzero = np.where(mask, fd, 0).sum(axis=1)
        p_grad_sum_nonzero = np.where(mask, p_grad, 0).sum(axis=1)
        print("fd", fd_sum_nonzero)
        print("ana", p_grad_sum_nonzero)

        np.savez(
            "cosalign_failure_debug2.npz",
            p_np=p_np,
            target=target,
            states_init=states_init,
        )
        print(np.linalg.norm(fd_sum_nonzero - p_grad_sum_nonzero, np.inf))
        if np.linalg.norm(fd_sum_nonzero - p_grad_sum_nonzero, np.inf) < 1e-6:
            print("ok")
        else:
            plt.show()
            raise ValueError(
                f"Cosalign pour certains points non nuls descend sous {threshold}! "
                f"Données sauvegardées dans cosalign_failure_debug.npz"
            )
    plt.close("all")  # Ferme toutes les figures

    plt.clf()

    #     input()
    # input()
plt.plot(histerrr)
plt.yscale("log")
plt.show()
