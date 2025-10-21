import numpy as np
import sys
import os
import platform
import pinocchio as pin


np.random.seed(21)
pin.seed(21)

system = platform.system()
paths = []
if system == "Linux":
    paths.append(
        "/lustre/fswork/projects/rech/tln/urh44lu/pinocchio-minimal-main/build/python"
    )
elif system == "Darwin":  # macOS
    paths.append("/Users/mscheffl/Desktop/pinocchio-minimal-main/build/python")
else:
    raise RuntimeError(f"Système non supporté : {system}")
for p in paths:
    if os.path.exists(p):
        if p not in sys.path:
            sys.path.insert(0, p)
import tartempion


def finite_difference_gradient(workspace, fixed, updated, epsilon=1e-4):
    batch_size = updated.shape[0]
    grad_numerical = np.zeros_like(updated)
    for batch_idx in range(batch_size):
        for j in range(updated.shape[1]):
            updated_plus = updated.copy()
            updated_plus[batch_idx, j] += epsilon
            losses_plus = workspace.SE3_loss(fixed, updated_plus)
            updated_minus = updated.copy()
            updated_minus[batch_idx, j] -= epsilon
            losses_minus = workspace.SE3_loss(fixed, updated_minus)
            grad_numerical[batch_idx, j] = (
                losses_plus[batch_idx] - losses_minus[batch_idx]
            ) / (2 * epsilon)

    return grad_numerical


def SE3_loss_python(updated, frozen):
    """Forward pass en Python"""
    batch_size = frozen.shape[0]
    losses = np.zeros(batch_size)
    frozen_SE3 = []
    updated_SE3 = []
    Jexp = []

    for batch_id in range(batch_size):
        updated_ = pin.exp6(pin.Motion(updated[batch_id]))
        frozen_ = pin.exp6(pin.Motion(frozen[batch_id]))

        frozen_SE3.append(frozen_)
        updated_SE3.append(updated_)

        diff = frozen_.actInv(updated_)

        log_diff = pin.log6(diff).vector
        losses[batch_id] = np.dot(log_diff, log_diff)
        Jexp.append(updated[batch_id])
    return losses, frozen_SE3, updated_SE3, Jexp


def d_SE3_loss_python(frozen_SE3, updated_SE3, Jexp):
    """Backward pass en Python - VERSION À CORRIGER"""
    batch_size = len(frozen_SE3)
    grad = np.zeros((batch_size, 6))

    for batch_id in range(batch_size):
        # frozen^{-1} * updated
        diff = frozen_SE3[batch_id].actInv(updated_SE3[batch_id])

        # Jacobienne de log6
        Jlog = pin.Jlog6(diff)

        # Jacobienne de exp6
        log_updated = pin.log6(updated_SE3[batch_id])

        # Gradient de la loss par rapport à log(diff)
        log_diff = pin.log6(diff).vector
        grad_e = 2.0 * log_diff  # d(||e||²)/de = 2*e
        # Chain rule
        grad[batch_id] = grad_e.T @ Jlog @ Jexp[batch_id]

    return grad


# Test
SE3_loss_workspace = tartempion.SE3_loss_workspace()

# Données de test
batch_size = 30000
dim = 6

np.random.seed(42)
fixed_np = np.random.randn(batch_size, dim).astype(np.float64)
updated_np = np.random.randn(batch_size, dim).astype(np.float64)


# print(updated_np[117])
# print(fixed_np[117])
# print(pin.log6(pin.exp6(pin.Motion(updated_np[117]))).vector)
# print(pin.log6(pin.exp6(pin.Motion(fixed_np[117]))).vector)
# print(pin.exp6(pin.Motion(fixed_np[117])))
# print(pin.exp6(pin.Motion(updated_np[117])))
# # for i in range(len(updated_np)):
# #     updated_np[i] = pin.log6(pin.exp6(pin.Motion(updated_np[i]))).vector
# #     fixed_np[i] = pin.log6(pin.exp6(pin.Motion(fixed_np[i]))).vector
# print(
#     (
#         pin.log6(
#             pin.exp6(pin.Motion(fixed_np[117])).actInv(
#                 pin.exp6(pin.Motion(updated_np[117]))
#             )
#         ).vector
#         ** 2
#     ).sum()
# )


losses = SE3_loss_workspace.SE3_loss(updated_np, fixed_np)
# print(losses[117])

# print("=" * 60)
# print("VÉRIFICATION DES GRADIENTS (LOSS BATCHÉE)")
# print("=" * 60)

# # 1. Calcul des losses
# print(f"\nLosses shape: {losses.shape}")
# print(f"Losses: {losses}")
# print(f"Loss totale (sum): {np.sum(losses):.6f}")

# # 2. Gradient analytique
grad_analytical = np.array(SE3_loss_workspace.d_SE3_loss())
# print(f"\nGradient analytique shape: {grad_analytical.shape}")
# print(f"Gradient analytique:\n{grad_analytical}")

# # 3. Gradient numérique
# print("\nCalcul du gradient numérique...")
grad_numerical = finite_difference_gradient(
    SE3_loss_workspace, fixed_np, updated_np, 1e-5
)
# print(f"\nGradient numérique shape: {grad_numerical.shape}")
# print(f"Gradient numérique:\n{grad_numerical}")

# # 3. Gradient numérique
# print("\nCalcul du gradient numérique...")
losses, frozen_SE3, updated_SE3, Jexp = SE3_loss_python(updated_np, fixed_np)
grad_python = d_SE3_loss_python(frozen_SE3, updated_SE3, Jexp)
# print("loss python", losses)
# print(f"\nGradient python shape: {grad_python.shape}")
# print(f"Gradient python:\n{grad_python}")

# 4. Comparaison
diff = np.abs(grad_analytical - grad_numerical)
rel_error = diff / (np.abs(grad_numerical) + 1e-8)

print(grad_numerical[117])
print(grad_python[117])
print(grad_analytical[117])
print(grad_numerical[182])
print(grad_python[182])
print(grad_analytical[182])

# print("\n" + "=" * 60)
# print("RÉSULTATS")
# print("=" * 60)
# print(f"\nDifférence absolue max: {np.max(diff):.2e}")
# print(f"Différence absolue moyenne: {np.mean(diff):.2e}")
# print(f"\nErreur relative max: {np.max(rel_error):.2e}")
# print(f"Erreur relative moyenne: {np.mean(rel_error):.2e}")

# Verdict par élément du batch
# print("\n--- Analyse par batch ---")
# for i in range(batch_size):
#     batch_diff = diff[i]
#     batch_rel_error = rel_error[i]
#     print(
#         f"Batch {i}: max_diff={np.max(batch_diff):.2e}, max_rel_err={np.max(batch_rel_error):.2e}"
#     )

# Affichage des plus grandes erreurs
print("\n--- Top 5 des plus grandes erreurs ---")
flat_indices = np.argsort(diff.ravel())[-5:]
for flat_idx in reversed(flat_indices):
    idx = np.unravel_index(flat_idx, diff.shape)
    print(
        f"Position {idx}: "
        f"analytical={grad_analytical[idx]:.6e}, "
        f"numerical={grad_numerical[idx]:.6e}, "
        f"diff={diff[idx]:.2e}"
    )

# Verdict global
threshold = 1e-5
if np.max(rel_error) < threshold:
    print(f"\n✅ GRADIENT CORRECT (erreur < {threshold})")
elif np.max(rel_error) < 1e-8:
    print(f"\n⚠️  GRADIENT ACCEPTABLE (erreur < 1e-3)")
else:
    print(f"\n❌ GRADIENT INCORRECT (erreur > 1e-3)")
