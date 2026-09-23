"""Runner for off-policy HARL algorithms."""
import torch
import numpy as np
import torch.nn.functional as F
from harl.runners.off_policy_base_runner import OffPolicyBaseRunner
from geomloss import SamplesLoss
from harl.utils.trpo_util import (
    flat_grad,
    flat_params,
    conjugate_gradient,
    fisher_vector_product,
    update_model,
    kl_divergence,
)

class OffPolicyHARunner(OffPolicyBaseRunner):
    """Runner for off-policy HA algorithms."""
    train_step = 0
    def train(self):
        """Train the model"""
        # -------- Sinkhorn HADF setup (lazy) --------
        algo_cfg = self.algo_args.get("algo", {}) if hasattr(self, "algo_args") else {}
        sink_cfg = algo_cfg.get("sinkhorn", {}) if isinstance(algo_cfg, dict) else {}
        use_sink = bool(sink_cfg.get("enabled", False))

        if use_sink:
            # GeomLoss parameters (maps to YAML)
            p       = sink_cfg.get("p", 2)
            blur    = sink_cfg.get("blur", 0.05)      # ε = blur^p (actions normalized to [-1,1])
            scaling = sink_cfg.get("scaling", 0.8)
            backend = sink_cfg.get("backend", "auto")
            # Wasserstein (blur=0) and MMD (blur=∞) are special cases
            self._sinkhorn_loss = SamplesLoss(
                loss="sinkhorn", p=p, blur=blur, debias=True, scaling=scaling, backend=backend
            )
            # Modes / dual state
            self._sink_mode     = str(sink_cfg.get("mode", "dual")).lower()  # "dual" | "projection" | "penalty"
            self._sink_use_dual = bool(sink_cfg.get("use_dual", self._sink_mode == "dual"))
            self._sink_radius   = float(sink_cfg.get("radius", 0.10))  # target sqrt(S)
            self._sink_dual_lr  = float(sink_cfg.get("dual_lr", 0.01))
            self._sink_lam      = float(sink_cfg.get("lam", 0.10))
            self._sink_max_proj_halves = int(sink_cfg.get("max_proj_halves", 8))
            self.ls_step = sink_cfg.get("ls_step",10)
            self.accept_ratio = sink_cfg.get("accept_ratio", 0.5)
            self.backtrack_coeff = sink_cfg.get("backtrack_coeff", 0.8)
            self.step_size = sink_cfg.get("step_size", 0.01)


        self.total_it += 1
        data = self.buffer.sample()
        (
            sp_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_obs,  # (n_agents, batch_size, dim)
            sp_actions,  # (n_agents, batch_size, dim)
            sp_available_actions,  # (n_agents, batch_size, dim)
            sp_reward,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_done,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_valid_transition,  # (n_agents, batch_size, 1)
            sp_term,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_next_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_next_obs,  # (n_agents, batch_size, dim)
            sp_next_available_actions,  # (n_agents, batch_size, dim)
            sp_gamma,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
        ) = data
        # train critic
        self.critic.turn_on_grad()
        if self.args["algo"] == "hasac":
            next_actions = []
            next_logp_actions = []
            for agent_id in range(self.num_agents):
                next_action, next_logp_action = self.actor[
                    agent_id
                ].get_actions_with_logprobs(
                    sp_next_obs[agent_id],
                    sp_next_available_actions[agent_id]
                    if sp_next_available_actions is not None
                    else None,
                )
                next_actions.append(next_action)
                next_logp_actions.append(next_logp_action)
            self.critic.train(
                sp_share_obs,
                sp_actions,
                sp_reward,
                sp_done,
                sp_valid_transition,
                sp_term,
                sp_next_share_obs,
                next_actions,
                next_logp_actions,
                sp_gamma,
                self.value_normalizer,
            )
        else:
            next_actions = []
            for agent_id in range(self.num_agents):
                next_actions.append(
                    self.actor[agent_id].get_target_actions(sp_next_obs[agent_id])
                )
            self.critic.train(
                sp_share_obs,
                sp_actions,
                sp_reward,
                sp_done,
                sp_term,
                sp_next_share_obs,
                next_actions,
                sp_gamma,
            )
        self.critic.turn_off_grad()
        sp_valid_transition = torch.tensor(sp_valid_transition, device=self.device)
        if self.total_it % self.policy_freq == 0:
            # train actors
            if self.args["algo"] == "hasac":
                actions = []
                logp_actions = []
                with torch.no_grad():
                    for agent_id in range(self.num_agents):
                        action, logp_action = self.actor[
                            agent_id
                        ].get_actions_with_logprobs(
                            sp_obs[agent_id],
                            sp_available_actions[agent_id]
                            if sp_available_actions is not None
                            else None,
                        )
                        actions.append(action)
                        logp_actions.append(logp_action)
                # actions shape: (n_agents, batch_size, dim)
                # logp_actions shape: (n_agents, batch_size, 1)
                if self.fixed_order:
                    agent_order = list(range(self.num_agents))
                else:
                    agent_order = list(np.random.permutation(self.num_agents))
                for agent_id in agent_order:
                    self.actor[agent_id].turn_on_grad()
                    # train this agent
                    actions[agent_id], logp_actions[agent_id] = self.actor[
                        agent_id
                    ].get_actions_with_logprobs(
                        sp_obs[agent_id],
                        sp_available_actions[agent_id]
                        if sp_available_actions is not None
                        else None,
                    )
                    if self.state_type == "EP":
                        logp_action = logp_actions[agent_id]
                        actions_t = torch.cat(actions, dim=-1)
                    elif self.state_type == "FP":
                        logp_action = torch.tile(
                            logp_actions[agent_id], (self.num_agents, 1)
                        )
                        actions_t = torch.tile(
                            torch.cat(actions, dim=-1), (self.num_agents, 1)
                        )
                    value_pred = self.critic.get_values(sp_share_obs, actions_t)
                    if self.algo_args["algo"]["use_policy_active_masks"]:
                        if self.state_type == "EP":
                            actor_loss = (
                                -torch.sum(
                                    (value_pred - self.alpha[agent_id] * logp_action)
                                    * sp_valid_transition[agent_id]
                                )
                                / sp_valid_transition[agent_id].sum()
                            )
                        elif self.state_type == "FP":
                            valid_transition = torch.tile(
                                sp_valid_transition[agent_id], (self.num_agents, 1)
                            )
                            actor_loss = (
                                -torch.sum(
                                    (value_pred - self.alpha[agent_id] * logp_action)
                                    * valid_transition
                                )
                                / valid_transition.sum()
                            )
                    else:
                        actor_loss = -torch.mean(
                            value_pred - self.alpha[agent_id] * logp_action
                        )
                    self.actor[agent_id].actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor[agent_id].actor_optimizer.step()
                    self.actor[agent_id].turn_off_grad()
                    # train this agent's alpha
                    if self.algo_args["algo"]["auto_alpha"]:
                        log_prob = (
                            logp_actions[agent_id].detach()
                            + self.target_entropy[agent_id]
                        )
                        alpha_loss = -(self.log_alpha[agent_id] * log_prob).mean()
                        self.alpha_optimizer[agent_id].zero_grad()
                        alpha_loss.backward()
                        self.alpha_optimizer[agent_id].step()
                        self.alpha[agent_id] = torch.exp(
                            self.log_alpha[agent_id].detach()
                        )
                    actions[agent_id], _ = self.actor[
                        agent_id
                    ].get_actions_with_logprobs(
                        sp_obs[agent_id],
                        sp_available_actions[agent_id]
                        if sp_available_actions is not None
                        else None,
                    )
                # train critic's alpha
                if self.algo_args["algo"]["auto_alpha"]:
                    self.critic.update_alpha(logp_actions, np.sum(self.target_entropy))
            else:
                if self.args["algo"] == "had3qn":
                    actions = []
                    with torch.no_grad():
                        for agent_id in range(self.num_agents):
                            actions.append(
                                self.actor[agent_id].get_actions(
                                    sp_obs[agent_id], False
                                )
                            )
                    # actions shape: (n_agents, batch_size, 1)
                    update_actions, get_values = self.critic.train_values(
                        sp_share_obs, actions
                    )
                    if self.fixed_order:
                        agent_order = list(range(self.num_agents))
                    else:
                        agent_order = list(np.random.permutation(self.num_agents))
                    for agent_id in agent_order:
                        self.actor[agent_id].turn_on_grad()
                        # actor preds
                        actor_values = self.actor[agent_id].train_values(
                            sp_obs[agent_id], actions[agent_id]
                        )
                        # critic preds
                        critic_values = get_values()
                        # update
                        actor_loss = torch.mean(F.mse_loss(actor_values, critic_values))
                        self.actor[agent_id].actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor[agent_id].actor_optimizer.step()
                        self.actor[agent_id].turn_off_grad()
                        update_actions(agent_id)

                elif self.args["algo"] in ["haddpg", "hatd3"]:
                    actions = []
                    with torch.no_grad():
                        for agent_id in range(self.num_agents):
                            actions.append(
                                self.actor[agent_id].get_actions(
                                    sp_obs[agent_id], False
                                )
                            )
                    # actions shape: (n_agents, batch_size, dim)
                    if self.fixed_order:
                        agent_order = list(range(self.num_agents))
                    else:
                        agent_order = list(np.random.permutation(self.num_agents))
                    for agent_id in agent_order:
                        self.actor[agent_id].turn_on_grad()
                        # train this agent
                        actions[agent_id] = self.actor[agent_id].get_actions(
                            sp_obs[agent_id], False
                        )
                        actions_t = torch.cat(actions, dim=-1)
                        value_pred = self.critic.get_values(sp_share_obs, actions_t)
                        actor_loss = -torch.mean(value_pred)
                        self.actor[agent_id].actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor[agent_id].actor_optimizer.step()
                        self.actor[agent_id].turn_off_grad()
                        actions[agent_id] = self.actor[agent_id].get_actions(
                            sp_obs[agent_id], False
                        )

                elif self.args["algo"] == "hatd3_sinkhorn":
                    # === Deterministic (HADDPG / HATD3 backbone) with Sinkhorn Modes ===
                    # change sp_actions to torch tensor
                    sp_actions_ts = torch.tensor(sp_actions, device=self.device).clone().detach()
                    actions = []
                    with torch.no_grad():
                        for agent_id in range(self.num_agents):
                            actions.append(
                                self.actor[agent_id].get_target_actions(sp_obs[agent_id])
                            )
                            # actions.append(
                            #     self.old_actor[agent_id].get_actions(
                            #         sp_obs[agent_id], False
                            #     )
                            # )
                            # actions.append(
                            #     sp_actions_ts[agent_id]
                            # )

                    # print 1st item in batch for debugging
                    # print(f"- sp_actions:{sp_actions[0][0]}")
                    # print(f"- init actions:{actions[0][0]}")
                    agent_order = list(range(self.num_agents)) if self.fixed_order \
                                else list(np.random.permutation(self.num_agents))

                    rho = self._sink_radius 
                    mode = str(sink_cfg.get("mode", "projection")).lower() # options: "dual" | "projection" | "penalty"

                    # print(f"**** Sinkhorn HADF mode: {mode}")
                    for agent_id in agent_order:
                        self.actor[agent_id].turn_on_grad()

                        a_old = actions[agent_id]  # baseline μ_old(s) (detached)

                        # (1) New actions for this agent (requires grad)
                        a_new = self.actor[agent_id].get_actions(sp_obs[agent_id], False)  # [B, act_dim]
                        actions[agent_id] = a_new

                        # (2) Q term: centralized critic with joint actions
                        actions_t  = torch.cat(actions, dim=-1)
                        value_pred = self.critic.get_values(sp_share_obs, actions_t)  # [B,1]
                        actor_loss     = -torch.mean(value_pred)  # minimize -(Q)
                        
                        # (3) Apply Sinkhorn HADF (mode-dependent)
                        
                        if use_sink:
                            if mode == "dual":
                                # Dual (hard trust region): L = -(Q) + λ (S - ρ)
                                lam = float(sink_cfg.get("sink_lam", 0.10))
                                S = self._sinkhorn_loss(a_old, a_new)
                                actor_loss = actor_loss + lam * (S - rho)

                                # Optimize actor (standard step)
                                self.actor[agent_id].actor_optimizer.zero_grad()
                                actor_loss.backward()
                                self.actor[agent_id].actor_optimizer.step()
                                self.actor[agent_id].turn_off_grad()

                                # Dual λ update with post-step actions: λ ← [λ + η (S_post - ρ²)]_+
                                with torch.no_grad():
                                    a_new_post = self.actor[agent_id].get_actions(sp_obs[agent_id], False)
                                #     S_post = self._sinkhorn_loss(a_old, a_new_post)
                                #     lam_new = lam + self._sink_dual_lr * (S_post.item() - rho)
                                #     lam_new = max(lam_new, 0.0)
                                    
                                # Refresh for later agents
                                actions[agent_id] = a_new_post
                                # print(f"a_old[0]:{a_old[0]}, a_new[0]:{a_new[0]}, a_new_post[0]:{a_new_post[0]}, S:{S.item():.4f}, actor_loss:{actor_loss.item():.4f}")

                        

                            elif mode == "projection":
                                old_actor = self.actor[agent_id]
                                # compute gradident of actor_loss explicitly
                                actor_loss_grad = torch.autograd.grad(actor_loss, self.actor[agent_id].actor.parameters(), allow_unused=True)
                                actor_loss_grad = flat_grad(actor_loss_grad)
                                max_actor_update = actor_loss_grad * self.step_size
                                params = flat_params(self.actor[agent_id].actor)

                                flag = False
                                fraction = 1.0
                                expected_improve = actor_loss
                                for i in range(self.ls_step):
                                    new_params = params - fraction * max_actor_update
                                    update_model(self.actor[agent_id].actor, new_params)
                                    a_new_update = self.actor[agent_id].get_actions(sp_obs[agent_id], False)  # [B, act_dim]
                                    actions[agent_id] = a_new_update
                                    actions_t  = torch.cat(actions, dim=-1)
                                    value_pred = self.critic.get_values(sp_share_obs, actions_t)
                                    new_actor_loss = -torch.mean(value_pred)
                                    actor_loss_improve = new_actor_loss - actor_loss
                                    S = self._sinkhorn_loss(a_new, a_new_update)
                                    # print(f"step:{i}, S={S.item():.4f} > {rho:.4f}, actor_loss:{actor_loss}, new_loss:{new_actor_loss}, actor_loss_improve={actor_loss_improve.item():.6f} > 0")
                                  
                                    if (S.item() <= rho) and (actor_loss_improve / expected_improve > self.accept_ratio) : #and (actor_loss_improve.item() > 0):
                                        # print(f"Projection LS succeeded, S={S.item():.4f} <= {rho:.4f}, Δactor_loss={actor_loss_improve.item():.6f} < 0")
                                        flag = True
                                        break
                                    expected_improve *= self.backtrack_coeff
                                    fraction *= self.backtrack_coeff
                                if not flag:
                                    print(f"*** Projection LS failed agent {agent_id}, S={S.item():.4f} > {rho:.4f}, actor_loss:{actor_loss}, new_loss:{new_actor_loss}")
                                    params = flat_params(old_actor.actor)
                                    update_model(self.actor[agent_id].actor, params)
                                    # print(f"policy update does not impove the surrogate agent {agent_id}")

                                # (6) Refresh for later agents
                                actions[agent_id] = self.actor[agent_id].get_actions(sp_obs[agent_id], False)

                            else:
                                pass

                elif self.args["algo"] == "hatd3_sinkhorn_igm":
                    # === Deterministic (HADDPG / HATD3 backbone) with Sinkhorn Modes ===

                    if self.train_step < 1e6:
                        # print(f"Warmup policy gradient step {self.train_step}")
                        actions = []
                        for agent_id in range(self.num_agents):
                            self.actor[agent_id].turn_on_grad()
                            self.actor[agent_id].actor_optimizer.zero_grad()
                            actions.append(self.actor[agent_id].get_actions(sp_obs[agent_id], False))

                        actions_t  = torch.cat(actions, dim=-1)
                        value_pred = self.critic.get_values(sp_share_obs, actions_t)  # [B,1]
                        actor_loss     = -torch.mean(value_pred)  # minimize -(Q) = maximize Q

                        actor_loss.backward()

                        for agent_id in range(self.num_agents):
                            self.actor[agent_id].actor_optimizer.step()

                    else:
                        # print(f"Sinkhorn policy gradient step {self.train_step}")
                        # change sp_actions to torch tensor
                        sp_actions_ts = torch.tensor(sp_actions, device=self.device).clone().detach()
                        actions = []
                        with torch.no_grad():
                            for agent_id in range(self.num_agents):
                                actions.append(
                                    self.actor[agent_id].get_target_actions(sp_obs[agent_id])
                                )

                        for agent_id in range(self.num_agents):
                            self.actor[agent_id].turn_off_grad()
                        
                        agent_order = list(range(self.num_agents)) if self.fixed_order \
                                    else list(np.random.permutation(self.num_agents))

                        rho = self._sink_radius 
                        mode = str(sink_cfg.get("mode", "projection")).lower() # options: "dual" | "projection" | "penalty"

                        # print(f"**** Sinkhorn HADF mode: {mode}")
                        for agent_id in agent_order:
                            self.actor[agent_id].turn_on_grad()

                            a_old = actions[agent_id]  # baseline μ_old(s) (detached)

                            # (1) New actions for this agent (requires grad)
                            a_new = self.actor[agent_id].get_actions(sp_obs[agent_id], False)  # [B, act_dim]
                            actions[agent_id] = a_new

                            # (2) Q term: centralized critic with joint actions
                            actions_t  = torch.cat(actions, dim=-1)
                            value_pred = self.critic.get_values(sp_share_obs, actions_t)  # [B,1]
                            actor_loss     = -torch.mean(value_pred)  # minimize -(Q)
                            
                            # (3) Apply Sinkhorn HADF (mode-dependent)
                            
                            if use_sink:
                                if mode == "dual":
                                    # Dual (hard trust region): L = -(Q) + λ (S - ρ)
                                    lam = float(sink_cfg.get("sink_lam", 0.10))
                                    S = self._sinkhorn_loss(a_old, a_new)
                                    actor_loss = actor_loss + lam * (S - rho)

                                    # Optimize actor (standard step)
                                    self.actor[agent_id].actor_optimizer.zero_grad()
                                    actor_loss.backward()
                                    self.actor[agent_id].actor_optimizer.step()
                                    self.actor[agent_id].turn_off_grad()

                                    # Dual λ update with post-step actions: λ ← [λ + η (S_post - ρ²)]_+
                                    with torch.no_grad():
                                        a_new_post = self.actor[agent_id].get_actions(sp_obs[agent_id], False)
                                    #     S_post = self._sinkhorn_loss(a_old, a_new_post)
                                    #     lam_new = lam + self._sink_dual_lr * (S_post.item() - rho)
                                    #     lam_new = max(lam_new, 0.0)
                                        
                                    # Refresh for later agents
                                    actions[agent_id] = a_new_post
                                    # print(f"a_old[0]:{a_old[0]}, a_new[0]:{a_new[0]}, a_new_post[0]:{a_new_post[0]}, S:{S.item():.4f}, actor_loss:{actor_loss.item():.4f}")

                            

                                elif mode == "projection":
                                    old_actor = self.actor[agent_id]
                                    # compute gradident of actor_loss explicitly
                                    actor_loss_grad = torch.autograd.grad(actor_loss, self.actor[agent_id].actor.parameters(), allow_unused=True)
                                    actor_loss_grad = flat_grad(actor_loss_grad)
                                    max_actor_update = actor_loss_grad * self.step_size
                                    params = flat_params(self.actor[agent_id].actor)

                                    flag = False
                                    fraction = 1.0
                                    expected_improve = actor_loss
                                    for i in range(self.ls_step):
                                        new_params = params - fraction * max_actor_update
                                        update_model(self.actor[agent_id].actor, new_params)
                                        a_new_update = self.actor[agent_id].get_actions(sp_obs[agent_id], False)  # [B, act_dim]
                                        actions[agent_id] = a_new_update
                                        actions_t  = torch.cat(actions, dim=-1)
                                        value_pred = self.critic.get_values(sp_share_obs, actions_t)
                                        new_actor_loss = -torch.mean(value_pred)
                                        actor_loss_improve = new_actor_loss - actor_loss
                                        S = self._sinkhorn_loss(a_new, a_new_update)
                                        # print(f"step:{i}, S={S.item():.4f} > {rho:.4f}, actor_loss:{actor_loss}, new_loss:{new_actor_loss}, actor_loss_improve={actor_loss_improve.item():.6f} > 0")
                                    
                                        if (S.item() <= rho) and (actor_loss_improve / expected_improve > self.accept_ratio) : #and (actor_loss_improve.item() > 0):
                                            # print(f"Projection LS succeeded, S={S.item():.4f} <= {rho:.4f}, Δactor_loss={actor_loss_improve.item():.6f} < 0")
                                            flag = True
                                            break
                                        expected_improve *= self.backtrack_coeff
                                        fraction *= self.backtrack_coeff
                                    if not flag:
                                        print(f"*** Projection LS failed agent {agent_id}, S={S.item():.4f} > {rho:.4f}, actor_loss:{actor_loss}, new_loss:{new_actor_loss}")
                                        params = flat_params(old_actor.actor)
                                        update_model(self.actor[agent_id].actor, params)
                                        # print(f"policy update does not impove the surrogate agent {agent_id}")

                                    # (6) Refresh for later agents
                                    actions[agent_id] = self.actor[agent_id].get_actions(sp_obs[agent_id], False)

                                else:
                                    pass
                # soft update target networks of actor and critic
                for agent_id in range(self.num_agents):
                    self.actor[agent_id].soft_update()
            self.critic.soft_update()
        
        self.train_step += 1
