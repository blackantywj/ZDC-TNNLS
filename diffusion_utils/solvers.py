import torch
from functools import partial
from diffusion_utils.dpm_solver_pytorch import DPM_Solver

def create_solver(config):
    if config.dynamic.solver == "euler":
        return EulerDiffEqSolver
    if config.dynamic.solver == "dpmsolver":
        return DPM_Solver
    if config.dynamic.solver == "dpmsolver++":
        return DPM_Solver_PlusPlus

class EulerDiffEqSolver:
    def __init__(self, dynamic, score_fn, ode_sampling=False):
        self.dynamic = dynamic
        self.score_fn = score_fn
        self.ode_sampling = ode_sampling

    def step(self, x_t, t, next_t, **kwargs):
        """
        Implement reverse SDE/ODE Euler solver
        """

        """
        x_mean = deterministic part
        x = x_mean + noise (yet another noise sampling)
        """
        dt = (next_t - t).view(-1, 1, 1)
        noise = torch.randn_like(x_t)
        drift, diffusion, score_output = self.dynamic.reverse_params(x_t, t, partial(self.score_fn, **kwargs), self.ode_sampling)
        x_mean = x_t + drift * dt
        x = x_mean + diffusion.view(-1, 1, 1) * torch.sqrt(-dt) * noise
        return {
            "x": x,
            "x_mean": x_mean,
            "x_0": score_output["x_0"],
        }

class EulerSolverX0(EulerDiffEqSolver):
    """Solver for models that predict x0 directly."""
    def step(self, x_t, t, next_t, **kwargs):
        dt = (next_t - t).view(-1, 1, 1)
        score_fn = partial(self.score_fn, **kwargs)
        x0_pred = score_fn(x_t=x_t, t=t)['x_0']                  # ˆx₀ :contentReference[oaicite:7]{index=7}
        beta_t    = self.dynamic.scheduler.beta_t(t).view(-1, 1, 1)
        alpha_cum = self.dynamic.scheduler.params(t)[0].view(-1, 1, 1)
        score     = (alpha_cum.sqrt()*x0_pred - x_t)/(1.0-alpha_cum)  # Tweedie :contentReference[oaicite:8]{index=8}
        drift     = -0.5*beta_t*x_t - beta_t*score                   # VP SDE :contentReference[oaicite:9]{index=9}
        diffusion = beta_t.sqrt().view(-1)
        noise     = torch.randn_like(x_t)
        x_mean    = x_t + drift*dt
        x         = x_mean + diffusion.view(-1,1,1)*torch.sqrt(-dt)*noise
        return {"x": x, "x_mean": x_mean, "x_0": x0_pred}

class DDIMSolver(EulerSolverX0):
    """
    DDIM solver for models that predict x0 directly.
    eta controls the amount of noise (0 for deterministic).
    """
    def __init__(self, dynamic, score_fn, ode_sampling=False, eta: float = 0.0):
        super().__init__(dynamic, score_fn, ode_sampling)
        self.eta = eta

    def step(self, x_t, t, next_t, **kwargs):
        # predictions
        score_fn = partial(self.score_fn, **kwargs)
        x0_pred = score_fn(x_t=x_t, t=t)['x_0']

        # schedule parameters
        alpha_t     = self.dynamic.scheduler.params(t)[0].view(-1, 1, 1)  # cumulative alpha_t
        alpha_next  = self.dynamic.scheduler.params(next_t)[0].view(-1, 1, 1)  # cumulative alpha_{t-1}

        # compute noise residual epsilon
        sigma_t = torch.sqrt(1.0 - alpha_t)
        eps_pred = (x_t - alpha_t.sqrt() * x0_pred) / sigma_t

        # ddim update
        x_pred = alpha_next.sqrt() * x0_pred + torch.sqrt(1.0 - alpha_next - self.eta**2 * (1.0 - alpha_t)) * eps_pred

        if self.eta > 0:
            # add stochastic noise
            noise = torch.randn_like(x_t)
            x = x_pred + self.eta * torch.sqrt(1.0 - alpha_t) * noise
        else:
            x = x_pred

        return {"x": x, "x_mean": x_pred, "x_0": x0_pred}

class DPM_Solver_PlusPlus:
    def __init__(self, dynamic, score_fn, ode_sampling=False):
        """
        二阶 DPM-Solver++（Predictor-Corrector 结构），纯 ODE 版本。
        Args:
            dynamic:      用户自定义的动态模型对象，需实现 reverse_params / get_alpha 等
            score_fn:     打分函数，调用时会被包装成 partial(score_fn, **kwargs)
            ode_sampling: 是否走纯 ODE（True 表示不加噪声），传递给 dynamic
        """
        self.dynamic = dynamic
        self.score_fn = score_fn
        self.ode_sampling = ode_sampling

    def step(self, x_t, t, next_t, **kwargs):
        """
        二阶 DPM-Solver++ 一步更新。
        返回字典中包含：
          - "x_mean": 纯 ODE 下的二阶修正结果
          - "x":      如果需要添加噪声可在 "x_mean" 上加噪；当前纯 ODE, 等同于 "x_mean"
          - "x_0":    第一次（一阶 predictor）得到的 \hat x_0
          - "x_0_tilda": 第二次（corrector）处得到的 \hat x_0^{(tilda)}
        Args:
            x_t:      当前时刻样本 (B, C, H, W) 或类似
            t:        当前 step, 形状 (B,) 或 (B,1)
            next_t:   下一个 step, 形状同 t
            **kwargs: 传给 score_fn 的其他参数（例如 cond, cond_mask, attention_mask 等）
        """
        # ———— 第一步（predictor）：在 (x_t, t) 处预测 x0 并做一阶更新 ————
        drift, diffusion, score_output = self.dynamic.reverse_params(
            x_t, t, partial(self.score_fn, **kwargs), self.ode_sampling
        )
        x0_pred = score_output["x_0"]

        # 计算 alpha(t) 和 alpha(next_t)
        alpha_t, _ = self.dynamic.scheduler.params(t)
        alpha_next, _ = self.dynamic.scheduler.params(next_t)
        sqrt_ratio = torch.sqrt(alpha_next / alpha_t)
        coef = (alpha_next - alpha_t) / torch.sqrt(alpha_t)

        # predictor：一阶更新得到 x_tilde
        x_tilde = sqrt_ratio * x_t - coef * x0_pred

        # ———— 第二步（corrector）：在 (x_tilde, next_t) 处重新预测 x0 ————
        drift_tilda, diffusion_tilda, score_output_tilda = self.dynamic.reverse_params(
            x_tilde, next_t, partial(self.score_fn, **kwargs), self.ode_sampling
        )
        x0_tilda = score_output_tilda["x_0"]

        # ———— 第三步：结合两次预测结果做二阶修正 ————
        delta = alpha_next - alpha_t
        # DPM-Solver++ 二阶修正项
        correct_term = 0.5 * delta / torch.sqrt(alpha_next) * (x0_tilda - x0_pred)

        x_mean = sqrt_ratio * x_t - coef * x0_pred - correct_term

        # 同样，如果想在 x_mean 上加弱随机噪声，可以把下面几行取消注释：
        # noise = torch.randn_like(x_t)
        # sigma_next = self.dynamic.get_sigma(next_t)
        # x = x_mean + sigma_next * torch.sqrt(next_t - t) * noise

        # 这里只做纯 ODE
        x = x_mean

        return {
            "x": x,
            "x_mean": x_mean,
            "x_0": x0_pred,
            "x_0_tilda": x0_tilda,
        }
