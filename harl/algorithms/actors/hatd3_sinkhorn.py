# harl/algorithms/actors/hatd3_sinkhorn.py
import torch
from geomloss import SamplesLoss
from harl.algorithms.actors.hatd3 import HATD3  # base TD3 actor


class HATD3Sinkhorn(HATD3):
    """
    HAML (deterministic) actor update with Sinkhorn divergence on pushforward action measures.
    Supports:
      - Hard constraint (trust-region) via dual variable:   loss = -(Q) + λ ( S - ρ^2 )
      - Hard constraint via projection (line search on actions)
      - Penalty form:                                       loss = -(Q - λ S)
    """
    def __init__(self, *args, sinkhorn_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        sc = sinkhorn_cfg or {}
        # Sinkhorn geometry / solver
        self.blur     = sc.get("blur", 0.05)        # ε = blur**p, actions normalized to [-1,1]
        self.p        = sc.get("p", 2)
        self.scaling  = sc.get("scaling", 0.8)
        self.backend  = sc.get("backend", "auto")
        self.iters    = sc.get("iters", 50)         # (GeomLoss handles internally)
        # Modes / constraint
        self.mode     = sc.get("mode", "dual")      # "dual" | "projection" | "penalty"
        self.use_dual = sc.get("use_dual", True)    # kept for backward-compat; overrides mode if True
        self.radius   = float(sc.get("radius", 0.10))   # target sqrt(S)
        self.target   = self.radius ** 2                # target on S
        self.dual_lr  = float(sc.get("dual_lr", 0.01))
        self.max_proj_halves = int(sc.get("max_proj_halves", 8))
        # Penalty / initial λ
        self.lam      = float(sc.get("lam", 0.10))

        self.sink = SamplesLoss(
            loss="sinkhorn", p=self.p, blur=self.blur,
            debias=True, scaling=self.scaling, backend=self.backend
        )

    @torch.no_grad()
    def _anchor_actions(self, obs_batch):
        # Fixed pushforward reference for this inner step (detached)
        return self.actor(obs_batch).clamp_(-1, 1).detach()

    def _get_opt(self):
        # Support either attribute name
        return getattr(self, "actor_opt", None) or getattr(self, "actor_optimizer", None)

    def _grad_clip(self):
        max_gn = getattr(self, "max_grad_norm", 10.0)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_gn)

    def _critic_q_mean(self, obs, a_new, others=None):
        # Centralized critic should return twin critics for TD3
        q1, q2 = self.critic(obs, a_new, others)  # shape [B,1] each
        return torch.min(q1, q2).mean()

    def actor_update_with_sinkhorn(self, batch):
        """
        One actor update with Sinkhorn-based HADF.
        Expects batch keys: "obs", "others" (centralized inputs for critic).
        """
        obs    = batch["obs"]                         # [B, ...]
        others = batch.get("others", None)

        # (A) Anchor: pushforward of μ_old over current state batch (detached)
        a_old = self._anchor_actions(obs)             # [B, act_dim]

        # (B) Current policy actions (requires grad)
        a_new = self.actor(obs)                       # [B, act_dim]

        # (C) Critic improvement term (maximize Q)
        q_pi = self._critic_q_mean(obs, a_new, others)  # scalar

        # (D) Sinkhorn divergence between pushforward action sets
        S = self.sink(a_old, a_new)                   # scalar, debiased

        # Resolve mode: prefer explicit 'mode', but keep use_dual=True for backward-compat
        mode = "dual" if self.use_dual else self.mode.lower()

        # (E) Build loss per mode
        if mode == "dual":
            # Hard constraint via Lagrangian: minimize -(Q) + λ (S - ρ^2)
            loss_actor = -(q_pi) + self.lam * (S - self.target)

        elif mode == "projection":
            # Hard constraint via projection: backtrack α so S(a_old, a_proj) <= ρ^2
            if S.item() > self.target:
                alpha = 1.0
                a_proj = a_new
                for _ in range(self.max_proj_halves):
                    alpha *= 0.5
                    a_proj = a_old + alpha * (a_new - a_old)  # α is a Python float (no grad through LS)
                    S_proj = self.sink(a_old, a_proj)
                    if S_proj.item() <= self.target:
                        break
                # Recompute Q on projected actions; gradient flows via a_new → a_proj
                q_pi_proj = self._critic_q_mean(obs, a_proj, others)
                loss_actor = -(q_pi_proj)
                S = S_proj  # log the enforced divergence
            else:
                loss_actor = -(q_pi)

        else:
            # Penalty form: minimize -(Q - λ S)
            loss_actor = -(q_pi - self.lam * S)

        # (F) Optimize
        opt = self._get_opt()
        assert opt is not None, "Actor optimizer not found (expected actor_opt or actor_optimizer)."
        opt.zero_grad(set_to_none=True)
        loss_actor.backward()
        self._grad_clip()
        opt.step()

        # (G) Dual λ update (only in 'dual' mode)
        S_post = S
        if mode == "dual":
            with torch.no_grad():
                a_new_post = self.actor(obs)                  # post-step actions
                S_post = self.sink(a_old, a_new_post)
                self.lam = max(0.0, float(self.lam + self.dual_lr * (S_post.item() - self.target)))

        # Logs
        return {
            "actor/q_pi": float(q_pi.item()),
            "actor/loss": float(loss_actor.item()),
            "actor/sinkhorn": float(S_post.item()),
            "actor/sinkhorn_sqrt": float(torch.sqrt(S_post).item()),
            "actor/lam": float(self.lam),
            "actor/target_radius": float(self.radius),
            "actor/eps_ot": float(self.blur ** self.p),
            "actor/mode": mode,
        }
