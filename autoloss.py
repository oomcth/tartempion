import tartempion
import numpy as np
import torch
from torch.autograd import Function
import pinocchio as pin


class torch_SE3_loss(Function):
    @staticmethod
    def forward(
        ctx,
        fixed: torch.Tensor,
        updated: torch.Tensor,
        SE3_loss_workspace: tartempion.SE3_loss_workspace,
        type=2,
        R: torch.Tensor = None,
        target=None,
    ):
        ctx.type = type
        if type == 1:
            fixed_np = fixed.detach().cpu().numpy()
            updated_np = updated.detach().cpu().numpy()
            ctx.SE3_loss_workspace = SE3_loss_workspace
            out = SE3_loss_workspace.SE3_loss(updated_np, fixed_np)
            return torch.from_numpy(out).to(fixed.device).to(fixed.dtype)
        if type == 2:
            ctx.SE3_loss_workspace = SE3_loss_workspace
            updated_np = updated.detach().cpu().numpy()
            R_np = R.detach().cpu().numpy()
            pred = []
            for i in range(len(updated)):
                pred.append(pin.SE3(R_np[i], updated_np[i]))
            out = SE3_loss_workspace.SE3_loss_2(pred, target)
            return torch.from_numpy(out).to(fixed.device).to(fixed.dtype)

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ):
        grad = np.array(ctx.SE3_loss_workspace.d_SE3_loss())
        has_nan = np.isnan(grad).any()
        has_large = np.abs(grad).max() > 1e5
        if has_nan or has_large:
            print(grad)
            print(np.abs(grad).max())
            print(np.isnan(grad).any())
            raise
        if ctx.type == 1:
            return (
                None,
                torch.from_numpy(grad).to(grad_output.device).to(grad_output.dtype),
                None,
                None,
                None,
                None,
            )
        if ctx.type == 2:
            return (
                None,
                torch.from_numpy(grad).to(grad_output.device).to(grad_output.dtype),
                None,
                None,
                None,
                None,
            )


class torch_SE3_loss_2(Function):
    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        a1: torch.Tensor,
        a2: torch.Tensor,
        SE3_loss_workspace: tartempion.SE3_loss_workspace,
        target,
    ):
        ctx.type = type
        ctx.SE3_loss_workspace = SE3_loss_workspace
        ctx.t_np = t.cpu().detach().numpy()
        ctx.a1_np = a1.cpu().detach().numpy()
        ctx.a2_np = a2.cpu().detach().numpy()
        out = SE3_loss_workspace.SE3_loss_3(ctx.t_np, ctx.a1_np, ctx.a2_np, target)
        return torch.from_numpy(out).to(t.device).to(t.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_t = np.array(ctx.SE3_loss_workspace.d_t_loss())
        grad_a1 = np.array(ctx.SE3_loss_workspace.d_a1_loss())
        grad_a2 = np.array(ctx.SE3_loss_workspace.d_a2_loss())

        g_t = torch.from_numpy(grad_t).to(grad_output.device).to(grad_output.dtype)
        g_a1 = torch.from_numpy(grad_a1).to(grad_output.device).to(grad_output.dtype)
        g_a2 = torch.from_numpy(grad_a2).to(grad_output.device).to(grad_output.dtype)

        grad_clip_max = 10.0

        def clip_batchwise(g):
            norms = g.norm(dim=1, keepdim=True)
            scale = torch.clamp(grad_clip_max / (norms + 1e-12), max=1.0)
            return g * scale

        g_t = clip_batchwise(g_t)
        g_a1 = clip_batchwise(g_a1)
        g_a2 = clip_batchwise(g_a2)

        return g_t, g_a1, g_a2, None, None


if __name__ == "__main__":
    import torch
    import numpy as np

    torch.set_printoptions(precision=5, sci_mode=False, linewidth=150)

    # 🧩 À adapter à ton projet :
    import tartempion

    workspace = tartempion.SE3_loss_workspace()

    B = 400
    torch.manual_seed(0)
    target = B * [pin.SE3.Random()]

    t = torch.randn(B, 3, dtype=torch.double, requires_grad=True)
    a1 = torch.randn(B, 3, dtype=torch.double, requires_grad=True)
    a2 = torch.randn(B, 3, dtype=torch.double, requires_grad=True)

    loss_fn = torch_SE3_loss_2.apply
    loss_vals = loss_fn(t, a1, a2, workspace, target)
    loss_scalar = loss_vals.sum()
    loss_scalar.backward()

    grad_t_torch = t.grad.detach().clone()
    grad_a1_torch = a1.grad.detach().clone()
    grad_a2_torch = a2.grad.detach().clone()

    print("Loss:", loss_scalar.item())
    print("‖grad_t‖  =", grad_t_torch.norm().item())
    print("‖grad_a1‖ =", grad_a1_torch.norm().item())
    print("‖grad_a2‖ =", grad_a2_torch.norm().item())

    # ================================================================
    #  FONCTION DE PERTE POUR LES DIFFÉRENCES FINIES
    # ================================================================
    def compute_loss(tt, aa1, aa2):
        out = loss_fn(tt, aa1, aa2, workspace, target)
        return out.sum()

    # ================================================================
    #  GRADIENTS NUMÉRIQUES (FINITE DIFFERENCE)
    # ================================================================
    def numerical_grad(param, fn, eps=1e-6):
        grad_fd = torch.zeros_like(param, dtype=torch.double)
        for b in range(param.shape[0]):
            for i in range(param.shape[1]):
                plus = param.clone().detach()
                minus = param.clone().detach()
                plus[b, i] += eps
                minus[b, i] -= eps
                Lp = fn(plus)
                Lm = fn(minus)
                grad_fd[b, i] = (Lp - Lm) / (2.0 * eps)
        return grad_fd

    grad_t_fd = numerical_grad(t.clone().detach(), lambda tt: compute_loss(tt, a1, a2))
    grad_a1_fd = numerical_grad(
        a1.clone().detach(), lambda aa1: compute_loss(t, aa1, a2)
    )
    grad_a2_fd = numerical_grad(
        a2.clone().detach(), lambda aa2: compute_loss(t, a1, aa2)
    )

    # ================================================================
    #  COMPARAISON DES GRADIENTS
    # ================================================================
    def compare_gradients(name, grad_fd, grad_torch):
        diff_norm = torch.linalg.norm(grad_fd - grad_torch)
        rel_err = diff_norm / (torch.linalg.norm(grad_torch) + 1e-12)
        print(f"\nGradient {name}")
        print("‖Torch‖ =", grad_torch.norm().item())
        print("‖FD‖    =", grad_fd.norm().item())
        print("‖diff‖  =", diff_norm.item())
        print("Rel_err =", rel_err.item())

    compare_gradients("t", grad_t_fd, grad_t_torch)
    compare_gradients("a1", grad_a1_fd, grad_a1_torch)
    compare_gradients("a2", grad_a2_fd, grad_a2_torch)

    print("\n✅  Vérification par différences finies terminée.")

    # ================================================================
    #  STRESS TEST NUMÉRIQUE SUR UN GRAND ÉCHANTILLON
    # ================================================================
    print("\n🧪  Démarrage du stress-test numérique ...")

    torch.manual_seed(123)
    num_batches = 1000  # nombre de lots à tester
    batch_size = 1000  # taille d’un lot (tu peux pousser à 10_000 ou +)
    dtype = torch.double

    # bornes de génération aléatoire (selon ton domaine SE3)
    abs_max = 10.0

    max_loss = 0.0
    max_grad = 0.0
    num_nan = 0
    target = batch_size * [pin.SE3.Random()]
    for step in range(num_batches):
        t = (2 * abs_max) * torch.rand(batch_size, 3, dtype=dtype) - abs_max
        a1 = (2 * abs_max) * torch.rand(batch_size, 3, dtype=dtype) - abs_max
        a2 = (2 * abs_max) * torch.rand(batch_size, 3, dtype=dtype) - abs_max

        t.requires_grad_(True)
        a1.requires_grad_(True)
        a2.requires_grad_(True)

        try:
            loss = loss_fn(t, a1, a2, workspace, target)
            loss_sum = loss.mean()
            loss_sum.backward()

            lval = loss_sum.detach().item()
            gn_t = t.grad.norm().item()
            gn_a1 = a1.grad.norm().item()
            gn_a2 = a2.grad.norm().item()

            if not (
                np.isfinite(lval)
                and np.isfinite(gn_t)
                and np.isfinite(gn_a1)
                and np.isfinite(gn_a2)
            ):
                num_nan += 1
                print(f"⚠️ NaN ou Inf détecté au batch {step}")
                continue

            max_loss = max(max_loss, abs(lval))
            max_grad = max(max_grad, gn_t, gn_a1, gn_a2)

            if step % 100 == 0:
                print(f"[{step:04d}/{num_batches}] loss={lval:.4e}  ‖grad‖≈{gn_t:.4e}")

        except Exception as e:
            print(f"❌ Exception au batch {step}: {e}")
            num_nan += 1

    print("\n=== Résumé du stress test ===")
    print(f"Batches testés : {num_batches}")
    print(f"Nombre d'échecs ou NaN : {num_nan}")
    print(f"Loss max observée : {max_loss:.4e}")
    print(f"Norme de gradient max : {max_grad:.4e}")
    print("✅  Fin du stress test.")
