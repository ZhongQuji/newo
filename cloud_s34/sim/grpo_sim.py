#!/usr/bin/env python3
"""Decision-level GRPO simulator for the "verification paradox".

Independent re-implementation (cloud session S34) of the toy model used in the
project to study why outcome-only RLVR rewards turn a verifier into a
rubber-stamper, and to compare reward/audit designs under a *token (oracle
call) budget*.  Pure numpy; vectorised over (cells x seeds) so whole phase
diagrams run in seconds.

Model
-----
A prompt is a record from a layer with base rate pi of being physically true.
The policy picks one of up to four behaviours (softmax over logits):

  S0  stamp, no ledger      assert "consistent" without any derivation
  S1  stamp, fake ledger    assert "consistent" and write a fabricated derivation
  C   check                 run the verification (cost c), genuine ledger,
                            right with prob a
  A   abstain               "cannot verify" (reward r_a), never audited

Outcome reward: 1 if the final verdict is right (S0/S1 right iff record true,
C right w.p. a), minus c for C.  Audit designs add a penalty kappa when an
audited ledger is wrong/missing (S0/S1 always fail an audit, C fails w.p. 1-a).

Reward schemes (``scheme`` argument)
  outcome      plain RLVR
  balanced     negatives up-sampled so the effective base rate is 0.5
  tapo         twin-paired: the sample must be right on the record *and* its
               negative twin; stamping gets exactly one of the two
  ra           random audit of every response w.p. q (TAPO-RA)
  ra_ledger    S0 (no ledger) penalised for free by a format check;
               only responses *with* a ledger (S1, C) are oracle-audited w.p. q
  adaptive     ra_ledger whose q is a controller on the observed audit-failure
               rate (q_{t+1} = clip(q_t + eta*(fail_t - target), q_min, q_max))

Estimators (``estimator`` argument)
  grpo         A = (r - mean)/ (std + eps) within the group      (default)
  drgrpo       A = r - mean (no std normalisation)
  reinforce    A = r - b, b an EMA baseline over all samples (no group term)

Oracle-call accounting (per sample, *including* the base label that every
RLVR scheme needs):  outcome 1; balanced 1 (+ twin generation amortised: 2);
tapo 3 (label + twin label + twin construction check); ra 1 + q;
ra_ledger 1 + q*P(ledger present); adaptive likewise with its running q.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict, field

import numpy as np

ACTIONS = ["S0", "S1", "C", "A"]
S0, S1, C, A = range(4)


@dataclass
class Config:
    pi: float = 0.9          # base rate of true records
    a: float = 0.9           # accuracy of a genuine check
    c: float = 0.05          # cost of checking (reward units)
    kappa: float = 1.0       # audit penalty
    q: float = 0.3           # audit probability (ra / ra_ledger / initial for adaptive)
    r_abstain: float = 0.0   # reward for abstaining
    kappa0: float = 1.0      # free penalty for a missing ledger (ra_ledger / adaptive)
    scheme: str = "outcome"
    estimator: str = "grpo"
    G: int = 8               # group size
    n: int = 16              # prompts per step
    lr: float = 0.3
    eps_explore: float = 0.0 # epsilon-mixing with uniform at sampling time (TAPO-eps)
    allow_abstain: bool = False
    allow_s1: bool = True    # if False the policy has no "fake ledger" action
    # adaptive controller
    eta: float = 0.5
    target_fail: float = 0.05
    q_min: float = 0.0
    q_max: float = 1.0
    # initial policy: P(C) at step 0 (rest split between stamping actions)
    p_check0: float = 0.5
    steps: int = 400
    seeds: int = 20


# ----------------------------------------------------------------------------- theory
def c_star(pi: float, a: float) -> float:
    """Outcome-only: checking beats stamping iff c < a - pi."""
    return a - pi


def q_star(pi: float, a: float, c: float, kappa: float) -> float:
    """Random audit: checking beats fake-ledger stamping iff q > (pi + c - a)/(kappa a)."""
    return (pi + c - a) / (kappa * a)


def expected_rewards(cfg: Config, q: float | None = None) -> dict:
    """Closed-form expected reward of each pure behaviour under the scheme."""
    pi, a, c, k = cfg.pi, cfg.a, cfg.c, cfg.kappa
    q = cfg.q if q is None else q
    if cfg.scheme == "outcome":
        return {"S0": pi, "S1": pi, "C": a - c, "A": cfg.r_abstain}
    if cfg.scheme == "balanced":
        return {"S0": 0.5, "S1": 0.5, "C": a - c, "A": cfg.r_abstain}
    if cfg.scheme == "tapo":
        # must be right on both members of the pair (true record + negative twin)
        return {"S0": 0.0, "S1": 0.0, "C": a * a - c, "A": cfg.r_abstain}
    if cfg.scheme == "ra":
        return {"S0": pi - q * k, "S1": pi - q * k, "C": a - c - q * k * (1 - a), "A": cfg.r_abstain}
    if cfg.scheme in ("ra_ledger", "adaptive"):
        return {"S0": pi - cfg.kappa0, "S1": pi - q * k, "C": a - c - q * k * (1 - a), "A": cfg.r_abstain}
    raise ValueError(cfg.scheme)


# ----------------------------------------------------------------------------- simulator
class Sim:
    """Vectorised over `cells` (parameter settings) and `seeds`."""

    def __init__(self, cfgs: list[Config], seed: int = 0):
        self.cfgs = cfgs
        self.K = len(cfgs)
        self.S = cfgs[0].seeds
        self.rng = np.random.default_rng(seed)
        f = lambda name: np.array([getattr(c, name) for c in cfgs], dtype=float)[:, None]  # noqa: E731
        self.pi, self.a, self.c, self.kappa, self.kappa0 = f("pi"), f("a"), f("c"), f("kappa"), f("kappa0")
        self.r_abs, self.lr, self.eps = f("r_abstain"), f("lr"), f("eps_explore")
        self.q = np.repeat(f("q"), self.S, axis=1)  # per (cell, seed) so adaptive can drift
        self.G, self.n = cfgs[0].G, cfgs[0].n
        self.scheme = np.array([c.scheme for c in cfgs])
        self.estimator = cfgs[0].estimator
        self.allow_abstain = np.array([c.allow_abstain for c in cfgs])[:, None]
        self.allow_s1 = np.array([c.allow_s1 for c in cfgs])[:, None]
        # logits [K, S, 4]; disallowed actions get -inf
        z = np.zeros((self.K, self.S, 4))
        p0 = f("p_check0")
        z[:, :, C] = np.log(p0 / (1 - p0 + 1e-12))  # vs stamping mass
        # split stamping mass between S0 and S1 equally (log 0.5 each) when both allowed
        z[:, :, S0] = np.log(0.5)
        z[:, :, S1] = np.where(self.allow_s1, np.log(0.5), -np.inf)
        z[:, :, A] = np.where(self.allow_abstain, np.log(0.1), -np.inf)
        self.z = z
        self.baseline = np.full((self.K, self.S), 0.5)
        self.hist_pc: list[np.ndarray] = []
        self.hist_q: list[np.ndarray] = []
        self.hist_cost: list[np.ndarray] = []
        self.hist_fail: list[np.ndarray] = []
        self.cum_cost = np.zeros((self.K, self.S))

    # -- helpers
    def probs(self) -> np.ndarray:
        z = self.z - np.max(np.where(np.isfinite(self.z), self.z, -1e9), axis=-1, keepdims=True)
        e = np.where(np.isfinite(z), np.exp(z), 0.0)
        return e / e.sum(-1, keepdims=True)

    def _sample_actions(self, p: np.ndarray) -> np.ndarray:
        """p [K,S,4] -> actions [K,S,n,G] with eps-mixing over *allowed* actions."""
        allowed = np.isfinite(self.z)
        u = allowed / allowed.sum(-1, keepdims=True)
        pm = (1 - self.eps[..., None]) * p + self.eps[..., None] * u
        cdf = np.cumsum(pm, axis=-1)
        r = self.rng.random((self.K, self.S, self.n, self.G, 1))
        return (r > cdf[:, :, None, None, :]).sum(-1).clip(0, 3)

    def _rewards(self, act: np.ndarray):
        """Return rewards [K,S,n,G], oracle cost per sample [K,S], audit fail rate [K,S]."""
        K, S, n, G = act.shape
        true = self.rng.random(act.shape) < self.pi[..., None, None]
        chk_ok = self.rng.random(act.shape) < self.a[..., None, None]
        qx = self.q[..., None, None]
        audited = self.rng.random(act.shape) < qx
        is_s0, is_s1 = act == S0, act == S1
        is_s = is_s0 | is_s1
        is_c = act == C
        is_a = act == A
        c4, k4, k04 = self.c[..., None, None], self.kappa[..., None, None], self.kappa0[..., None, None]
        sch = self.scheme[:, None, None, None]
        # outcome / balanced / tapo
        t_bal = self.rng.random(act.shape) < 0.5
        ok2 = self.rng.random(act.shape) < self.a[..., None, None]
        r_out = np.where(is_s, true * 1.0, 0.0) + np.where(is_c, chk_ok - c4, 0.0)
        r_bal = np.where(is_s, t_bal * 1.0, 0.0) + np.where(is_c, chk_ok - c4, 0.0)
        r_tapo = np.where(is_c, (chk_ok & ok2) * 1.0 - c4, 0.0)
        # random audit of everything
        pen = audited * k4
        r_ra = np.where(is_s, true * 1.0 - pen, 0.0) + np.where(is_c, chk_ok - c4 - pen * (~chk_ok), 0.0)
        # ledger-aware audit: S0 penalised for free, S1/C audited
        r_led = (np.where(is_s0, true * 1.0 - k04, 0.0) + np.where(is_s1, true * 1.0 - pen, 0.0)
                 + np.where(is_c, chk_ok - c4 - pen * (~chk_ok), 0.0))
        is_led = (sch == "ra_ledger") | (sch == "adaptive")
        r = np.where(sch == "outcome", r_out,
            np.where(sch == "balanced", r_bal,
            np.where(sch == "tapo", r_tapo,
            np.where(sch == "ra", r_ra, r_led))))
        r = r + np.where(is_a, self.r_abs[..., None, None], 0.0)
        # oracle-call accounting per sample (incl. the base label)
        ledger = is_s1 | is_c
        cost = np.where(self.scheme[:, None] == "outcome", 1.0,
               np.where(self.scheme[:, None] == "balanced", 2.0,
               np.where(self.scheme[:, None] == "tapo", 3.0,
               np.where(self.scheme[:, None] == "ra", 1.0 + self.q, 1.0 + self.q * ledger.mean((2, 3))))))
        # observed audit failure rate (what a trainer can see without knowing the policy)
        au_scope = np.where(is_led, audited & ledger, audited)
        na = au_scope.sum((2, 3))
        nf = (au_scope & (np.where(is_led, is_s1, is_s) | (is_c & ~chk_ok))).sum((2, 3))
        fail = np.where(na > 0, nf / np.maximum(na, 1), 0.0)
        return r, cost, fail

    def _advantages(self, r: np.ndarray) -> np.ndarray:
        if self.estimator == "grpo":
            m = r.mean(-1, keepdims=True)
            s = r.std(-1, keepdims=True)
            return (r - m) / (s + 1e-6)
        if self.estimator == "drgrpo":
            return r - r.mean(-1, keepdims=True)
        if self.estimator == "reinforce":
            adv = r - self.baseline[..., None, None]
            self.baseline = 0.9 * self.baseline + 0.1 * r.mean((-1, -2))
            return adv
        raise ValueError(self.estimator)

    def step(self):
        p = self.probs()
        act = self._sample_actions(p)
        r, cost, fail = self._rewards(act)
        adv = self._advantages(r)
        onehot = np.eye(4)[act]                                  # [K,S,n,G,4]
        grad = (adv[..., None] * (onehot - p[:, :, None, None, :])).mean((2, 3))
        grad = np.where(np.isfinite(self.z), grad, 0.0)
        self.z = self.z + self.lr[..., None] * grad
        self.z = np.where(np.isfinite(self.z), np.clip(self.z, -30, 30), self.z)  # keep disallowed at -inf
        # adaptive controller on observed audit failure rate
        adp = self.scheme == "adaptive"
        if adp.any():
            cfg0 = next(c for c in self.cfgs if c.scheme == "adaptive")
            newq = np.clip(self.q + cfg0.eta * (fail - cfg0.target_fail), cfg0.q_min, cfg0.q_max)
            self.q = np.where(adp[:, None], newq, self.q)
        self.cum_cost += cost
        pc = self.probs()[:, :, C]
        self.hist_pc.append(pc.copy())
        self.hist_q.append(self.q.copy())
        self.hist_cost.append(cost.copy())
        self.hist_fail.append(fail.copy())

    def run(self, steps: int | None = None):
        for _ in range(steps or self.cfgs[0].steps):
            self.step()
        return self

    def set_scheme(self, scheme: str):
        """Switch every cell to a new scheme (used for the inoculation experiment)."""
        self.scheme = np.array([scheme] * self.K)
        return self

    # -- summaries
    def pc(self) -> np.ndarray:
        return np.stack(self.hist_pc)  # [T,K,S]

    def final_pc(self) -> np.ndarray:
        return self.pc()[-1]

    def steps_to(self, thr: float = 0.95) -> np.ndarray:
        """First step at which P(C) >= thr, per (cell, seed); NaN if never."""
        pc = self.pc()
        hit = pc >= thr
        first = np.where(hit.any(0), hit.argmax(0), -1).astype(float)
        first[first < 0] = np.nan
        return first

    def lockin_step(self, thr: float = 0.01) -> np.ndarray:
        pc = self.pc()
        hit = pc <= thr
        first = np.where(hit.any(0), hit.argmax(0), -1).astype(float)
        first[first < 0] = np.nan
        return first


# ----------------------------------------------------------------------------- selftest
def selftest() -> int:
    """Check the simulator against closed-form predictions."""
    fails = []

    def check(name, cond):
        print(("PASS " if cond else "FAIL ") + name)
        if not cond:
            fails.append(name)

    # 1. outcome-only: lock-in to stamping when c > c*, to checking when c < c*
    hi = Config(pi=0.9, a=0.9, c=0.3, scheme="outcome", steps=300, seeds=16)   # c* = 0 -> stamp
    lo = Config(pi=0.5, a=0.9, c=0.05, scheme="outcome", steps=300, seeds=16)  # c* = 0.4 -> check
    s = Sim([hi, lo], seed=1).run()
    f = s.final_pc()
    check(f"outcome lock-in to stamping when c>c*: mean P(C)={f[0].mean():.3f} < 0.05", f[0].mean() < 0.05)
    check(f"outcome lock-in to checking when c<c*: mean P(C)={f[1].mean():.3f} > 0.95", f[1].mean() > 0.95)
    # 2. random audit: two sides of q*
    base = dict(pi=0.96, a=0.8, c=0.05, kappa=1.0)
    qs = q_star(**base)
    check(f"q*(VD-like pi=.96,a=.8,c=.05)={qs:.3f} matches record 0.26", abs(qs - 0.2625) < 1e-3)
    above = Config(**base, q=qs + 0.2, scheme="ra", steps=400, seeds=16)
    below = Config(**base, q=max(qs - 0.2, 0.0), scheme="ra", steps=400, seeds=16)
    near = Config(**base, q=qs + 0.12, scheme="ra", steps=400, seeds=16)
    s = Sim([above, below, near], seed=2).run()
    f = s.final_pc()
    check(f"RA well above q* (+0.2): P(C)={f[0].mean():.3f} > 0.9", f[0].mean() > 0.9)
    check(f"RA well below q* (-0.2): P(C)={f[1].mean():.3f} < 0.1", f[1].mean() < 0.1)
    frac_locked_wrong = (f[2] < 0.5).mean()
    print(f"INFO near boundary (q*+0.12): median P(C)={np.median(f[2]):.3f}, "
          f"{frac_locked_wrong:.0%} of seeds locked into stamping (stochastic lock-in near the phase edge)")
    check("RA near boundary: median seed drifts toward checking (slow, small reward gap)", np.median(f[2]) > 0.6)
    # 3. zero-variance absorbing state: a homogeneous group gives exactly zero gradient under grpo
    cfg = Config(pi=0.9, scheme="outcome", seeds=2, G=8, n=4, allow_s1=False)
    sim = Sim([cfg], seed=3)
    sim.z[:, :, C] = -30  # policy ~deterministic stamping (all samples S0)
    z0 = sim.z.copy()
    sim.step()
    check("homogeneous groups -> zero logit change (absorbing state)", np.allclose(sim.z, z0, atol=1e-9))
    # 3b. the same homogeneous group moves under a non-group (EMA) baseline
    cfg = Config(pi=0.9, scheme="outcome", seeds=2, G=8, n=4, allow_s1=False, estimator="reinforce")
    sim = Sim([cfg], seed=3)
    sim.z[:, :, C] = -5  # strongly but not perfectly stamping
    z0 = sim.z.copy()
    for _ in range(5):
        sim.step()
    check("REINFORCE+EMA baseline keeps moving where GRPO would not", not np.allclose(sim.z, z0, atol=1e-6))
    # 3c. GRPO cost-invariance: in an all-correct mixed group the normalised advantage of the k
    #     checking samples is -sqrt((G-k)/k) whatever the cost c > 0.
    G, k = 8, 1
    for cc in (0.001, 0.1):
        r = np.array([1.0] * (G - k) + [1.0 - cc] * k)
        adv = (r - r.mean()) / (r.std() + 1e-6)
        check(f"GRPO advantage of a correct-but-costly check in an all-correct group: c={cc} -> {adv[-1]:.3f} ≈ -sqrt(7)={-np.sqrt(7):.3f}",
              abs(adv[-1] + np.sqrt((G - k) / k)) < 1e-2)
    # 4. expected rewards closed form sanity
    er = expected_rewards(Config(pi=0.9, a=0.9, c=0.05, kappa=1.0, q=0.3, scheme="ra"))
    check(f"RA expected rewards S1={er['S1']:.3f} C={er['C']:.3f}", abs(er["S1"] - 0.6) < 1e-9 and abs(er["C"] - 0.82) < 1e-9)
    # 5. abstention never attractive when r_a = 0 and q moderate
    cfg = Config(pi=0.9, a=0.9, c=0.05, q=0.3, scheme="ra", allow_abstain=True, r_abstain=0.0, steps=300, seeds=16)
    s = Sim([cfg], seed=4).run()
    pa = s.probs()[:, :, A].mean()
    check(f"abstention stays rare when r_a=0: P(A)={pa:.3f} < 0.05", pa < 0.05)
    print("selftest", "OK" if not fails else f"FAILED: {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    ap.print_help()
