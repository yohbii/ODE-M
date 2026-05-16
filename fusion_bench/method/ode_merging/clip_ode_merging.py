from collections import defaultdict
from copy import deepcopy
import logging
import math
import random
import re
from typing import Dict, List, Optional, Sequence, Union, cast

from omegaconf import DictConfig
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import lightning as L
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPVisionModel

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.dataset import CLIPDataset
from fusion_bench.taskpool.clip_vision.taskpool import CLIPVisionModelTaskPool
from fusion_bench.mixins import LightningFabricMixin, SimpleProfilerMixin, CLIPClassificationMixin

from .utils import get_param, get_param_gradients, get_param_names_to_merge, make_calibration_dataloader, sample_subset, TaskTaggedDataset

log = logging.getLogger(__name__)


class ODEMergingForCLIPVisionModel(
    BaseAlgorithm,
    CLIPClassificationMixin,
    LightningFabricMixin,
    SimpleProfilerMixin,
):
    """
    ODE Merging for CLIP Vision Model
    """

    _config_mapping = BaseAlgorithm._config_mapping | {
        "_dataloader_kwargs": "dataloader_kwargs",
        "_merge_mode": "merge_mode",
        "_num_samples": "num_samples",
        "_exclude_param_names_regex": "exclude_param_names_regex",
        "_exclude_model_names_regex": "exclude_model_names_regex",
        "_intetgration": "integration",
        "_macro_grad_norm_power": "macro_grad_norm_power",
        "_post_refine": "post_refine",
        "_post_refine_steps": "post_refine_steps",
        "_post_refine_lr": "post_refine_lr",
        "_post_refine_anchor": "post_refine_anchor",
        "_post_refine_radius": "post_refine_radius",
        "_post_refine_task_tol": "post_refine_task_tol",
        "_post_refine_grad_clip": "post_refine_grad_clip",
        "_post_refine_probe_batches": "post_refine_probe_batches",
    }
    _clip_processor: CLIPProcessor = None

    def __init__(
            self,
            dataloader_kwargs: DictConfig, 
            integration: DictConfig,
            seed: Optional[int] = None,
            num_samples: int = 1024,
            merge_mode: str = "sequential",
            exclude_param_names_regex: List[str] = [],
            exclude_model_names_regex: List[str] = [],
            evaluate_on_every_step: bool = False,
            barrier_budget: float = 0.08,
            barrier_tight_power: float = 2.0,
            barrier_probe_batches: int = 8,
            protect_old_only: bool = False,
            qp_num_constraints: int = 2,
            qp_sens_constraints: int = 2,
            qp_reg: float = 1e-6,
            qp_tol: float = 1e-6,
            dv_over_u_target: float = 0.24,
            dv_over_u_ramp: float = 0.0,
            eta_max: float = 1e6,
            eta_min: float = 0.0,
            macro_grad_batches: int = 8,
            sens_grad_batches: int = 1,
            constraint_grad_batches: int = 1,
            allowed_inc_per_step: float = 0.0,
            step_backtrack_steps: int = 8,
            step_shrink: float = 0.7,
            step_min: float = 1e-4,
            step_max: float = 1.0,
            step_grow: float = 1.25,
            macro_w_old: float = 1.0,
            macro_w_new: float = 1.0,
            macro_grad_norm_power: float = 0.5,
            post_refine: bool = False,
            post_refine_steps: int = 30,
            post_refine_lr: float = 1e-6,
            post_refine_anchor: float = 1e-3,
            post_refine_radius: float = 1e-3,
            post_refine_task_tol: float = 0.02,
            post_refine_grad_clip: float = 1.0,
            post_refine_probe_batches: int = 4,
            **kwargs,
        ):
        super().__init__(kwargs)
        self.seed = seed
        self.num_samples = num_samples
        self.dataloader_kwargs = dataloader_kwargs
        self.merge_mode = merge_mode
        self.intergration = integration
        self.exclude_param_names_regex = exclude_param_names_regex
        self.exclude_model_names_regex = exclude_model_names_regex
        self.evaluate_on_every_step = evaluate_on_every_step

        self.barrier_budget = barrier_budget
        self.barrier_tight_power = barrier_tight_power
        self.barrier_probe_batches = barrier_probe_batches
        self.protect_old_only = protect_old_only
        self.qp_num_constraints = qp_num_constraints
        self.qp_sens_constraints = qp_sens_constraints
        self.qp_reg = qp_reg
        self.qp_tol = qp_tol
        self.dv_over_u_target = dv_over_u_target
        self.dv_over_u_ramp = dv_over_u_ramp
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.macro_grad_batches = macro_grad_batches
        self.sens_grad_batches = sens_grad_batches
        self.constraint_grad_batches = constraint_grad_batches
        self.allowed_inc_per_step = allowed_inc_per_step
        self.step_backtrack_steps = step_backtrack_steps
        self.step_shrink = step_shrink
        self.step_min = step_min
        self.step_max = step_max
        self.step_grow = step_grow
        self.macro_w_old = macro_w_old
        self.macro_w_new = macro_w_new
        self.macro_grad_norm_power = macro_grad_norm_power
        self.post_refine = post_refine
        self.post_refine_steps = post_refine_steps
        self.post_refine_lr = post_refine_lr
        self.post_refine_anchor = post_refine_anchor
        self.post_refine_radius = post_refine_radius
        self.post_refine_task_tol = post_refine_task_tol
        self.post_refine_grad_clip = post_refine_grad_clip
        self.post_refine_probe_batches = post_refine_probe_batches


    def run(self, modelpool: Union[BaseModelPool, Dict[str, nn.Module]]) -> CLIPVisionModel:
        log.info("Running ODE Merging Algorithm")

        if self.seed:
            L.seed_everything(self.seed)
        self.accelerator = self.fabric.device

        if isinstance(modelpool, (dict, list, tuple)):
            modelpool = BaseModelPool(modelpool)
        self.modelpool = modelpool
        with self.profile("select_and_shuffle_models"):
            names = self.select_model_names()
            if len(names) < 2:
                raise ValueError(
                    f"Need at least 2 models to merge, got {len(names)} after filtering."
                )

            random.seed(self.seed)
            random.shuffle(names)
        log.info(f"Selected models: {names}")

        self.setup_zero_shot_classification_head()
        self.taskpool = cast(CLIPVisionModelTaskPool, self._program.taskpool)
        self._test_datasets = deepcopy(self.taskpool._test_datasets)
    

        if self.merge_mode == "sequential":
            merged_model = self.merge_sequential(names)
        elif self.merge_mode == "tree":
            pass
        else:
            raise ValueError(f"Unknown merge mode: {self.merge_mode}")
            
        return merged_model


    def select_model_names(self) -> List[str]:
        """
        Filter model names from the modelpool by exclude_model_names_regex.

        Args:
            None.
        
        Returns:
            List[str]: The selected model names.
        """
        names = list(getattr(self.modelpool, "model_names", []))
        if not names:
            # fallback if pool uses different attr
            names = list(getattr(self.modelpool, "models", {}).keys())

        if not names:
            raise ValueError("Could not find model names from modelpool (model_names/models).")

        # Apply regex exclusions (optional)
        if self.exclude_model_names_regex:
            patterns = [re.compile(p) for p in self.exclude_model_names_regex]
            kept = []
            for n in names:
                if any(p.search(n) for p in patterns):
                    continue
                kept.append(n)
            names = kept
        names = sorted(names)

        return names
    

    def merge_sequential(self, names: Sequence[str]) -> nn.Module:
        merged_model = self.modelpool.load_model(names[0]).to(self.accelerator)

        if self.evaluate_on_every_step:
            with self.profile("evaluating model"):
                self.taskpool._is_setup = False
                self.taskpool._test_datasets = DictConfig(
                    {names[0]: self._test_datasets[names[0]]}
                )
                report = self.taskpool.evaluate(deepcopy(merged_model))
        
        dataset = CLIPDataset(self.modelpool.load_train_dataset(names[0]), self.clip_processor)
        datasets = {names[0]: dataset}
        for model_idx, name in tqdm(
            enumerate(names[1:]),
            desc="Merging models",
            total=len(names) - 1,
        ):
            model_idx += 1
            model = self.modelpool.load_model(name).to(self.accelerator)
            for task in list(datasets.keys()):
                datasets[task] = sample_subset(datasets[task], self.num_samples // (model_idx + 1))
            task_dataset = CLIPDataset(
                sample_subset(self.modelpool.load_train_dataset(name), self.num_samples // (model_idx + 1)),
                self.clip_processor
            )
            datasets[name] = task_dataset
            log.info(f"Current datasets: {datasets}")
            merged_model = self.merge(
                model_idx,
                merged_model,
                model,
                datasets,
                names[: model_idx + 1],
            )

            if self.evaluate_on_every_step:
                with self.profile("evaluating model"):
                    self.taskpool._is_setup = False
                    self.taskpool._test_datasets = DictConfig(
                        {
                            n: self._test_datasets[n]
                            for n in names[: model_idx + 1]
                        }
                    )
                    report = self.taskpool.evaluate(deepcopy(merged_model))
                    print(report)

        if self.post_refine:
            with self.profile("post_refine"):
                merged_model = self._post_refine_model(merged_model, datasets, names)

        return merged_model
    

    def merge(self, model_idx, merged_model: nn.Module, model: nn.Module, datasets: any, tasks: Sequence[str]) -> nn.Module:
        if self.intergration.method == "euler":
            self._merge_euler(model_idx, merged_model, model, datasets, tasks)
        else:
            raise ValueError(f"Unknown integration method: {self.intergration.method}")
        
        return merged_model


    def _merge_euler(
        self,
        model_idx: int,
        merged_model: nn.Module,
        model: nn.Module,
        datasets: any,
        tasks: Sequence[str],
    ) -> nn.Module:
        eps = 1e-12
        device = self.fabric.device

        @torch.no_grad()
        def make_probe_batches(max_batches_per_task: int):
            probe = {}
            for task in tasks:
                dl = DataLoader(datasets[task], shuffle=False, **self.dataloader_kwargs)
                batches = []
                for b_idx, batch in enumerate(dl):
                    if b_idx >= max_batches_per_task:
                        break
                    batches.append(batch)
                probe[task] = batches
            return probe

        @torch.no_grad()
        def probe_task_loss(m: nn.Module, task: str, probe) -> float:
            batches = probe[task]
            if len(batches) == 0:
                return 0.0
            s = 0.0
            for batch in batches:
                s += float(self.compute_loss(m, batch, task))
            return s / len(batches)

        @torch.no_grad()
        def probe_all_task_losses(m: nn.Module, probe):
            return {task: probe_task_loss(m, task, probe) for task in tasks}

        def _backward_task_avg_loss(task: str, max_batches: int | None, denom_scale: float):
            dl = DataLoader(datasets[task], shuffle=False, **self.dataloader_kwargs)
            n = 0
            for b_idx, batch in enumerate(dl):
                if max_batches is not None and b_idx >= max_batches:
                    break
                loss = self.compute_loss(merged_model, batch, task)
                (loss / denom_scale).backward()
                n += 1
            return max(n, 1)

        @torch.no_grad()
        def dot_dict(a: dict, b: dict) -> float:
            s = 0.0
            for k in a.keys():
                s += float((a[k].double() * b[k].double()).sum())
            return s

        @torch.no_grad()
        def axpy_(y: dict, a: float, x: dict):
            for k in y.keys():
                y[k].add_(x[k], alpha=a)

        @torch.no_grad()
        def copy_dict_(dst_params: dict, src_params: dict):
            for k, p in dst_params.items():
                p.copy_(src_params[k])

        # Integration grid: full [0,1], stop at T_end
        T_end = 1.0 / (model_idx + 1)
        N_full = math.ceil(1.0 / self.intergration.step_size)
        time_grid = torch.linspace(0.0, 1.0, steps=N_full + 1, device=device)
        i_max = int(torch.searchsorted(time_grid, torch.tensor(T_end, device=device), right=True).item()) - 1
        i_max = max(i_max, 0)

        # Params
        param_names = get_param_names_to_merge(
            [n for n, _ in merged_model.named_parameters()],
            self.exclude_param_names_regex,
        )
        params = get_param(merged_model, param_names)
        target_params = get_param(model, param_names)

        B_max = float(getattr(self, "barrier_budget", 0.08))
        p_tight = float(getattr(self, "barrier_tight_power", 2.0))
        probe_batches = int(getattr(self, "barrier_probe_batches", 8))

        protect_old_only = bool(getattr(self, "protect_old_only", True))
        new_task = tasks[-1] if len(tasks) > 0 else None
        old_tasks = list(tasks[:-1]) if len(tasks) > 1 else list(tasks)
        protect_tasks = old_tasks if (protect_old_only and len(old_tasks) > 0) else list(tasks)

        m_excess = int(getattr(self, "qp_num_constraints", 2))
        m_sens = int(getattr(self, "qp_sens_constraints", 2))
        qp_reg = float(getattr(self, "qp_reg", 1e-6))
        qp_tol = float(getattr(self, "qp_tol", 1e-6))

        # Macro deviation
        dv_target = float(getattr(self, "dv_over_u_target", 0.24))
        dv_ramp = float(getattr(self, "dv_over_u_ramp", 0.0))
        eta_max = float(getattr(self, "eta_max", 1e6))
        eta_min = float(getattr(self, "eta_min", 0.0))

        # gradient budgets
        macro_batches = int(getattr(self, "macro_grad_batches", 8))
        sens_batches = int(getattr(self, "sens_grad_batches", 1))
        constraint_batches = int(getattr(self, "constraint_grad_batches", 1))

        # constraint slope cap
        allowed_inc_per_step = float(getattr(self, "allowed_inc_per_step", 0.0))

        # step line search
        max_bt = int(getattr(self, "step_backtrack_steps", 8))
        shrink = float(getattr(self, "step_shrink", 0.7))
        step_min = float(getattr(self, "step_min", 1e-4))
        step_max = float(getattr(self, "step_max", 1.0))
        step_grow = float(getattr(self, "step_grow", 1.25))

        # macro task weights 
        w_old = float(getattr(self, "macro_w_old", 1.0))
        w_new = float(getattr(self, "macro_w_new", 1.0))
        macro_norm_power = float(getattr(self, "macro_grad_norm_power", 0.5))

        probe = make_probe_batches(probe_batches)
        with torch.no_grad():
            L0_task = probe_all_task_losses(merged_model, probe)
            L1_task = probe_all_task_losses(model, probe)

        macro_tasks = list(tasks) 
        if len(macro_tasks) == 0:
            macro_tasks = list(protect_tasks)

        for i in range(i_max):
            t = float(time_grid[i].item())
            h = float((time_grid[i + 1] - time_grid[i]).item())
            t_next = t + h
            rem = max(1.0 - t, 1e-8)

            with torch.no_grad():
                B_t = B_max * (max(1.0 - t, 0.0) ** p_tight)
                B_t_next = B_max * (max(1.0 - t_next, 0.0) ** p_tight)

                L_cur = {task: probe_task_loss(merged_model, task, probe) for task in protect_tasks}

                L_max_t = {}
                L_max_next = {}
                for task in protect_tasks:
                    L0j = L0_task[task]
                    L1j = L1_task[task]
                    if protect_old_only and (new_task is not None) and (task != new_task):
                        L_ref_t = L0j
                        L_ref_next = L0j
                    else:
                        L_ref_t = (1.0 - t) * L0j + t * L1j
                        L_ref_next = (1.0 - t_next) * L0j + t_next * L1j

                    L_max_t[task] = L_ref_t + B_t
                    L_max_next[task] = L_ref_next + B_t_next

                excess_sorted = sorted(
                    [(task, L_cur[task] - L_max_t[task]) for task in protect_tasks],
                    key=lambda x: x[1],
                    reverse=True,
                )
                excess_tasks = [task for task, _ in excess_sorted[: max(1, min(m_excess, len(excess_sorted)))]]

            # precompute u
            with torch.no_grad():
                u_dict = {}
                for name, p in params.items():
                    tgt = target_params[name]
                    u_dict[name] = (tgt - p) / rem
                u_norm2 = max(dot_dict(u_dict, u_dict), eps)
                u_norm = math.sqrt(u_norm2)

            # top tasks with largest positive g_task^T u
            # compute per-task gradients
            sens_scores = []
            if m_sens > 0 and len(protect_tasks) > 0:
                with self.profile("compute sensitivity grads"):
                    for task in protect_tasks:
                        merged_model.zero_grad()
                        _backward_task_avg_loss(task, max_batches=sens_batches, denom_scale=1.0)
                        g_tmp = get_param_gradients(merged_model, param_names)
                        d = dot_dict(g_tmp, u_dict)  # dL/dt approx
                        sens_scores.append((task, float(d)))

                sens_scores.sort(key=lambda x: x[1], reverse=True)
                sens_tasks = [task for task, d in sens_scores if d > 0.0][:m_sens]
            else:
                sens_tasks = []

            # final constraint set
            constraint_tasks = []
            seen = set()
            for task in (excess_tasks + sens_tasks):
                if task not in seen:
                    seen.add(task)
                    constraint_tasks.append(task)

            # directional derivative bounds
            with torch.no_grad():
                inc_rate = allowed_inc_per_step / max(h, eps)
                b = {}
                for task in constraint_tasks:
                    b_raw = (L_max_next[task] - L_cur[task]) / max(h, eps)
                    b[task] = float(min(float(b_raw), float(inc_rate)))

            merged_model.zero_grad()
            with self.profile("compute macro gradient"):
                g_macro = {k: torch.zeros_like(v) for k, v in params.items()}
                total_w = 0.0
                for task in macro_tasks:
                    merged_model.zero_grad()
                    _backward_task_avg_loss(task, max_batches=macro_batches, denom_scale=1.0)
                    g_t = get_param_gradients(merged_model, param_names)

                    gtn2 = max(dot_dict(g_t, g_t), eps)
                    gtn = math.sqrt(gtn2)
                    scale = (gtn + eps) ** macro_norm_power

                    wt = w_new if (new_task is not None and task == new_task) else w_old
                    wt = float(wt)
                    total_w += wt

                    for k in g_macro.keys():
                        g_macro[k].add_(g_t[k], alpha=(wt / scale))

                if total_w > 0:
                    for k in g_macro.keys():
                        g_macro[k].mul_(1.0 / total_w)

            # eta from dv_target
            with torch.no_grad():
                gM_norm2 = max(dot_dict(g_macro, g_macro), eps)
                gM_norm = math.sqrt(gM_norm2)

                dv_t = dv_target * (1.0 + dv_ramp * (t / max(T_end, eps)))
                eta = dv_t * u_norm / (gM_norm + eps)
                eta = float(max(eta_min, min(eta_max, eta)))

            # constraint gradients
            g_list = []
            with self.profile("compute constraint gradients"):
                for task in constraint_tasks:
                    merged_model.zero_grad()
                    _backward_task_avg_loss(task, max_batches=constraint_batches, denom_scale=1.0)
                    g_j = get_param_gradients(merged_model, param_names)
                    g_list.append(g_j)

            with torch.no_grad():
                base_dict = {k: u_dict[k].clone() for k in u_dict.keys()}
                axpy_(base_dict, -eta, g_macro)

                m = len(constraint_tasks)
                if m == 0:
                    v_dict = base_dict
                    lam = []
                else:
                    K = torch.zeros((m, m), device=device, dtype=torch.float64)
                    c = torch.zeros((m,), device=device, dtype=torch.float64)
                    bvec = torch.zeros((m,), device=device, dtype=torch.float64)

                    for i_g in range(m):
                        c[i_g] = torch.tensor(dot_dict(g_list[i_g], base_dict), device=device, dtype=torch.float64)
                        bvec[i_g] = torch.tensor(float(b[constraint_tasks[i_g]]), device=device, dtype=torch.float64)
                        for j_g in range(m):
                            K[i_g, j_g] = torch.tensor(dot_dict(g_list[i_g], g_list[j_g]), device=device, dtype=torch.float64)

                    best_lam = None
                    best_obj = None
                    best_feas = False

                    for mask in range(1 << m):
                        idx = [k for k in range(m) if (mask >> k) & 1]
                        lam_try = torch.zeros((m,), device=device, dtype=torch.float64)

                        if len(idx) > 0:
                            Kss = K[idx][:, idx] + qp_reg * torch.eye(len(idx), device=device, dtype=torch.float64)
                            rhs = c[idx] - bvec[idx]
                            try:
                                sol = torch.linalg.solve(Kss, rhs)
                            except RuntimeError:
                                continue
                            if torch.any(sol < -qp_tol):
                                continue
                            lam_try[idx] = torch.clamp(sol, min=0.0)

                        gi_v = c - (K @ lam_try)
                        if torch.any(gi_v > bvec + qp_tol):
                            continue

                        obj = 0.5 * float((lam_try @ (K @ lam_try)).item())
                        if (best_obj is None) or (obj < best_obj):
                            best_obj = obj
                            best_lam = lam_try.clone()
                            best_feas = True

                    if not best_feas:
                        best_lam = torch.zeros((m,), device=device, dtype=torch.float64)

                    lam = [float(x) for x in best_lam.tolist()]

                    v_dict = {k: base_dict[k].clone() for k in base_dict.keys()}
                    for i_g in range(m):
                        if lam[i_g] != 0.0:
                            axpy_(v_dict, -lam[i_g], g_list[i_g])

            # step_scale
            with torch.no_grad():
                backup = {n: p.detach().clone() for n, p in params.items()}

            def _is_feasible(step_scale: float) -> tuple[bool, str, float]:
                with torch.no_grad():
                    copy_dict_(params, backup)
                    for name, p in params.items():
                        p.add_(v_dict[name], alpha=(h * step_scale))

                    worst_excess = -1e30
                    worst_task = None
                    violated = False
                    for task in protect_tasks:
                        Lj_next = probe_task_loss(merged_model, task, probe)
                        excess = Lj_next - L_max_next[task]
                        if excess > worst_excess:
                            worst_excess = excess
                            worst_task = task
                        if excess > 0.0:
                            violated = True
                    return (not violated), (worst_task if worst_task is not None else ""), float(worst_excess)

            # optionally grow
            step_scale = 1.0
            accepted = False
            worst_task_now = None
            worst_excess_now = None

            for _ in range(max_bt):
                ok, wt, we = _is_feasible(step_scale)
                worst_task_now, worst_excess_now = wt, we
                if ok:
                    accepted = True
                    break
                step_scale *= shrink
                if step_scale < step_min:
                    break

            if not accepted:
                step_scale = 0.0
            else:
                # try to grow step if enabled
                if step_max > 1.0:
                    cur = step_scale
                    while True:
                        nxt = min(cur * step_grow, step_max)
                        if nxt <= cur + 1e-12:
                            break

                        ok, wt, we = _is_feasible(nxt)
                        if not ok:
                            break

                        cur = nxt
                        worst_task_now, worst_excess_now = wt, we

                    step_scale = cur

            # commit exactly the accepted step
            # failed trial point from the feasibility checks above
            with torch.no_grad():
                copy_dict_(params, backup)
                for name, p in params.items():
                    p.add_(v_dict[name], alpha=(h * step_scale))

            # Diagnostics
            with torch.no_grad():
                du_dict = {k: (v_dict[k] - u_dict[k]) for k in u_dict.keys()}
                du_norm2 = max(dot_dict(du_dict, du_dict), 0.0)
                dv_over_u = math.sqrt(du_norm2) / (u_norm + eps)

                gM_dot_u = dot_dict(g_macro, u_dict)
                cos_ugM = gM_dot_u / ((u_norm * gM_norm) + eps)


        return merged_model



    def _post_refine_model(
        self,
        merged_model: nn.Module,
        datasets: any,
        tasks: Sequence[str],
    ) -> nn.Module:
        """
        Perform a short trust-region calibration refinement around the ODE solution.

        This stage is intentionally constrained: it uses the same calibration subsets,
        anchors parameters to the ODE solution, projects updates into a relative
        trust region, and returns the best probe-loss snapshot that satisfies a
        per-task loss tolerance.
        """
        steps = int(getattr(self, "post_refine_steps", 30))
        if steps <= 0 or len(tasks) == 0:
            return merged_model

        lr = float(getattr(self, "post_refine_lr", 1e-6))
        anchor_weight = float(getattr(self, "post_refine_anchor", 1e-3))
        radius = float(getattr(self, "post_refine_radius", 1e-3))
        task_tol = float(getattr(self, "post_refine_task_tol", 0.02))
        grad_clip = float(getattr(self, "post_refine_grad_clip", 1.0))
        probe_batches = int(getattr(self, "post_refine_probe_batches", 4))

        param_names = get_param_names_to_merge(
            [n for n, _ in merged_model.named_parameters()],
            self.exclude_param_names_regex,
        )
        params = get_param(merged_model, param_names)
        if len(params) == 0:
            return merged_model

        @torch.no_grad()
        def make_probe_batches(max_batches_per_task: int):
            probe = {}
            for task in tasks:
                dl = DataLoader(datasets[task], shuffle=False, **self.dataloader_kwargs)
                batches = []
                for b_idx, batch in enumerate(dl):
                    if b_idx >= max_batches_per_task:
                        break
                    batches.append(batch)
                probe[task] = batches
            return probe

        @torch.no_grad()
        def probe_task_loss(task: str, probe) -> float:
            batches = probe[task]
            if len(batches) == 0:
                return 0.0
            loss_sum = 0.0
            for batch in batches:
                loss_sum += float(self.compute_loss(merged_model, batch, task))
            return loss_sum / len(batches)

        @torch.no_grad()
        def probe_losses(probe):
            return {task: probe_task_loss(task, probe) for task in tasks}

        @torch.no_grad()
        def average_probe_loss(losses: Dict[str, float]) -> float:
            return sum(float(losses[task]) for task in tasks) / max(len(tasks), 1)

        @torch.no_grad()
        def copy_params(src: Dict[str, Tensor]):
            for name, p in params.items():
                p.copy_(src[name])

        @torch.no_grad()
        def clone_params() -> Dict[str, Tensor]:
            return {name: p.detach().clone() for name, p in params.items()}

        @torch.no_grad()
        def sq_norm(param_dict: Dict[str, Tensor]) -> float:
            value = 0.0
            for p in param_dict.values():
                value += float((p.detach().double() * p.detach().double()).sum())
            return value

        @torch.no_grad()
        def project_to_trust_region(anchor: Dict[str, Tensor], max_delta_norm: float):
            delta_norm_sq = 0.0
            for name, p in params.items():
                delta = p.detach().double() - anchor[name].detach().double()
                delta_norm_sq += float((delta * delta).sum())
            delta_norm = math.sqrt(max(delta_norm_sq, 0.0))
            if delta_norm <= max_delta_norm or delta_norm <= 1e-12:
                return
            scale = max_delta_norm / delta_norm
            for name, p in params.items():
                p.copy_(anchor[name] + (p - anchor[name]) * scale)

        merged_model.train()

        probe = make_probe_batches(probe_batches)
        baseline_losses = probe_losses(probe)
        best_loss = average_probe_loss(baseline_losses)

        anchor = clone_params()
        best_params = clone_params()
        anchor_norm = math.sqrt(max(sq_norm(anchor), 1e-12))
        max_delta_norm = radius * anchor_norm

        loaders = {}
        for task in tasks:
            loaders[task] = iter(DataLoader(datasets[task], shuffle=True, **self.dataloader_kwargs))

        def next_batch(task: str):
            try:
                return next(loaders[task])
            except StopIteration:
                loaders[task] = iter(DataLoader(datasets[task], shuffle=True, **self.dataloader_kwargs))
                return next(loaders[task])

        optimizer = torch.optim.AdamW(list(params.values()), lr=lr, weight_decay=0.0)

        for _ in range(steps):
            optimizer.zero_grad()

            loss = 0.0
            for task in tasks:
                loss = loss + self.compute_loss(merged_model, next_batch(task), task)
            loss = loss / max(len(tasks), 1)

            anchor_sq = 0.0
            for name, p in params.items():
                delta = p - anchor[name]
                anchor_sq = anchor_sq + (delta * delta).sum()
            anchor_loss = anchor_sq / max(anchor_norm * anchor_norm, 1e-12)

            total_loss = loss + anchor_weight * anchor_loss
            total_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(list(params.values()), grad_clip)
            optimizer.step()

            with torch.no_grad():
                project_to_trust_region(anchor, max_delta_norm)
                current_losses = probe_losses(probe)
                current_avg_loss = average_probe_loss(current_losses)
                task_safe = all(
                    current_losses[task] <= baseline_losses[task] + task_tol
                    for task in tasks
                )
                if task_safe and current_avg_loss < best_loss:
                    best_loss = current_avg_loss
                    best_params = clone_params()

        with torch.no_grad():
            copy_params(best_params)

        merged_model.eval()
        return merged_model



    def compute_loss(self, module, batch, task: str) -> Tensor:
        """
        Compute the loss for the given batch and task.

        Args:
            module (nn.Module): The model module.
            batch (tuple): The input batch.
            task (str): The name of the task.

        Returns:
            Tensor: The computed loss.
        """
        images, labels = batch 
        images = images.to(self.accelerator)
        labels = labels.to(self.accelerator)

        text_embeds = self.zeroshot_weights[task].to(self.accelerator)

        pooled = module(pixel_values=images).pooler_output  # (B,768)
        image_embeds = self.visual_projection(pooled)
        
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        
        # cosine similarity
        logits = torch.matmul(image_embeds, text_embeds.t()) * self.logit_scale_exp

        loss = F.cross_entropy(logits, labels)

        return loss
    

    @torch.no_grad()
    def compute_average_loss(
        self,
        module,
        datasets: any,
        tasks: Sequence[str],
    ) -> float:
        """
        Compute average loss over a dataloader.

        Args:
            module (nn.Module): CLIP vision module (or wrapper).
            datasets (Dict[str: Dataset]): yields (task, dataset).
            task (str): task name for zeroshot weights.

        Returns:
            float: average loss over all samples.
        """
        module.eval()

        total_loss = 0.0
        total_samples = 0

        for task in tasks:
            loader = DataLoader(datasets[task], **self.dataloader_kwargs)
            for batch in loader:
                bs = len(batch)

                loss = self.compute_loss(module, batch, task)
                total_loss += loss.item() * bs
                total_samples += bs

        return total_loss / max(total_samples, 1)
    
