# -*- coding: utf-8 -*-
"""Adapter around the Python Anesthesia Simulator (PAS).

PAS simulates propofol / remifentanil / norepinephrine and outputs BIS, MAP, CO
and TOL. It is used here to answer one question: *given the window the model sees,
what does the patient look like five minutes after a candidate action?*

Four things about PAS are worth knowing, all of them measured rather than assumed:

- **TOL is an index of tolerance, and 1 is the good end.** ``TOL_model.compute_tol``
  documents it as "TOL = 1 mean very relaxed and will tolerate laryngoscopie while
  TOL = 0 mean fully awake and will not tolerate": it rises with the opioid. Measured
  with propofol fixed at 0.15 mg/s, TOL goes 0.065 (no remifentanil) -> 0.717 (0.05)
  -> 0.950 (0.10) -> 1.000 (0.50 ug/s). Anything judging TOL rewards *more* analgesia
  when it wants TOL **close to 1**, never close to 0.
- **The horizon is real only if the patient starts from the calibrated rates.**
  ``initialized_at_given_input`` places the patient *at the equilibrium of the rates it
  is given*, so initialising with the post-action rates and integrating makes the patient
  sit at the new equilibrium for ever: the ``one_step`` at t=300 s returns the same state
  as t=1 s, which is the infinity-time answer, not the five-minute one. The patient is
  therefore initialised with the **calibrated** rates and the action is applied during
  the integration. Measured difference for a x1.5 propofol step: the steady state gives
  BIS 29.8 where the true five-minute transient gives 36.7.
- ``Patient.find_equilibrium`` needs the CasADi ``ipopt`` plugin, which the pip
  wheel does not ship. It is **not** used: the calibration is a plain bisection.
- ``c_blood_remi_eq`` reads 0 whatever the rate (a defect in the wheel); the
  equilibrium concentrations are read from the PK models instead (``propo_pk.x[0]``,
  ``remi_pk.x[0]``).

**Calibration.** The propofol:remifentanil *mix* comes from the effect-site
concentrations the window carries (``ce_propofol``, ``ce_remi``), which is what the mix
should come from: a fixed ratio is arbitrary and was measurably wrong, producing a
model concentration ratio of 2.9 against an observed one of 0.7. The *level* is then set
so the model reproduces the BIS the window shows, because matching the concentrations in
absolute terms leaves the BIS up to 19 points off: the dataset's PK/PD model and PAS's do
not agree on concentration -> BIS. MAP is matched with norepinephrine, which must be read
as a modelling compromise: the windows carry no vasopressor information at all (no field
in the corpus) and the model's MAP runs ~28 mmHg below the observed one, so that rate
absorbs model error as much as real drug. ``match_map=False`` drops it.

The calibration needs a baseline BIS and MAP. Windows that do not carry them (BIS is
absent in ~5.6% of the corpus, MAP in ~1.8%, and most of ``dev``) are calibrated
against the ideal values instead (``DEFAULT_BIS`` / ``DEFAULT_MAP``); each axis
records whether it was observed or imputed in ``Calibration.bis_source`` and
``Calibration.map_source``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

HORIZON_SECONDS = 300  # the user's "window five minutes after the action"
TS = 1  # integration step, in seconds
BISECTIONS = 40

#: Imputation targets, used only for the baselines a window does not carry:
#: the middle of the BIS target band and a MAP safely above the 65 mmHg threshold.
DEFAULT_BIS = 50.0
DEFAULT_MAP = 70.0

#: Multiplicative step applied to the calibrated rate of the target drug.
ACTION_FACTORS = {
    "increase_hypnotic": ("propo", 1.5),
    "reduce_hypnotic": ("propo", 0.5),
    "increase_opioid": ("remi", 1.5),
    "reduce_opioid": ("remi", 0.5),
    "no_action": (None, 1.0),
}

#: Search ranges for the calibration bisections.
U_REMI_MAX = 3.0  # ug/s
U_NORE_MAX = 10.0  # ug/s
U_PROPO_MAX = 1.0  # mg/s, only used to read the window's concentrations back

#: Fallback propofol:remifentanil rate ratio, for the few windows that carry no
#: effect-site concentrations. It used to be the only source of the mix.
DEFAULT_RATIO = 2.0


def _is_positive(value):
    """A usable positive number (the windows carry nulls and strings)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


@dataclass(frozen=True)
class Calibration:
    """Baseline rates that reproduce the BIS and MAP the window shows."""

    u_propo: float
    u_remi: float
    u_nore: float
    bis: float
    map: float
    bis_residual: float
    map_residual: float
    bis_source: str = "observed"
    map_source: str = "observed"
    #: propofol:remifentanil rate ratio, and where it came from.
    ratio: float = 0.0
    ratio_source: str = "default"
    #: Whether norepinephrine was used to reach the observed MAP.
    map_matched_with_nore: bool = True

    @property
    def matched(self):
        return abs(self.bis_residual) < 0.5 and abs(self.map_residual) < 0.5

    @property
    def imputed(self):
        """The baselines that were imputed rather than read from the window."""
        return tuple(
            name
            for name, source in (("bis", self.bis_source), ("map", self.map_source))
            if source == "imputed"
        )

    def as_dict(self):
        return {
            "u_propo": round(self.u_propo, 6),
            "u_remi": round(self.u_remi, 6),
            "u_nore": round(self.u_nore, 6),
            "bis": round(self.bis, 3),
            "map": round(self.map, 3),
            "bis_residual": round(self.bis_residual, 3),
            "map_residual": round(self.map_residual, 3),
            "bis_source": self.bis_source,
            "map_source": self.map_source,
            "ratio": round(self.ratio, 4),
            "ratio_source": self.ratio_source,
            "map_matched_with_nore": self.map_matched_with_nore,
        }


@dataclass(frozen=True)
class SimulatedOutcome:
    """What the simulator shows five minutes after an action."""

    action: str
    bis: float
    map: float
    co: float
    tol: float
    u_propo: float
    u_remi: float
    u_nore: float

    def as_dict(self):
        return {
            "action": self.action,
            "bis": round(self.bis, 2),
            "map": round(self.map, 2),
            "co": round(self.co, 3),
            "tol": round(self.tol, 4),
        }

    def signals(self):
        return {
            "bis": round(self.bis, 1),
            "map": round(self.map, 1),
            "co": round(self.co, 2),
            "tol": round(self.tol, 3),
        }


@dataclass
class _Signals:
    bis: float = 0.0
    co: float = 0.0
    map: float = 0.0
    tol: float = 0.0
    ce_propo: float = 0.0
    ce_remi: float = 0.0
    extra: dict = field(default_factory=dict)


class AnesthesiaSimulator:
    """Calibrate a patient from a window and simulate the candidate actions."""

    def __init__(self, horizon_seconds=HORIZON_SECONDS, ts=TS, bisections=BISECTIONS,
                 action_factors=None, default_bis=DEFAULT_BIS, default_map=DEFAULT_MAP,
                 match_map=True):
        self.horizon_seconds = horizon_seconds
        self.ts = ts
        self.steps = max(1, int(round(horizon_seconds / ts)))
        self.bisections = bisections
        self.action_factors = dict(action_factors or ACTION_FACTORS)
        self.default_bis = float(default_bis)
        self.default_map = float(default_map)
        #: Match the observed MAP with norepinephrine (a modelling compromise; see the
        #: module docstring). ``False`` keeps the patient off vasopressors.
        self.match_map = bool(match_map)
        self._pas = None

    # -- PAS plumbing ----------------------------------------------------- #

    def _module(self):
        """Import PAS lazily so loading the harness never depends on it."""
        if self._pas is None:
            try:
                import python_anesthesia_simulator as pas
            except ImportError as exc:  # pragma: no cover - dependency present
                raise ImportError(
                    "the simulator needs 'python-anesthesia-simulator': "
                    "pip install python-anesthesia-simulator"
                ) from exc
            self._pas = pas
        return self._pas

    def patient_characteristic(self, inp):
        """PAS wants ``[age (yr), height (cm), weight (kg), gender (0: F, 1: M)]``."""
        return [
            float(inp["pt_age"]),
            float(inp["pt_height"]),
            float(inp["pt_weight"]),
            1 if inp.get("pt_sex") == "M" else 0,
        ]

    def _new_patient(self, inp):
        return self._module().Patient(
            self.patient_characteristic(inp), ts=self.ts, save_data_bool=False
        )

    @staticmethod
    def _read(patient, values):
        """The signals of one step, plus the effect-site concentrations of the state."""
        import numpy as np

        flat = [float(np.ravel(value)[0]) for value in values]
        return _Signals(
            bis=flat[0],
            co=flat[1],
            map=flat[2],
            tol=flat[3],
            ce_propo=float(np.ravel(patient.propo_pk.x[0])[0]),
            ce_remi=float(np.ravel(patient.remi_pk.x[0])[0]),
        )

    def _steady(self, patient, u_propo, u_remi, u_nore):
        """Steady-state response: the init lands on the equilibrium of the rates."""
        patient.initialized_at_given_input(u_propo, u_remi, u_nore)
        return self._read(patient, patient.one_step(u_propo, u_remi, u_nore, noise=False))

    # -- Calibration ------------------------------------------------------ #

    def _bisect(self, function, low, high, target, increasing):
        """Bisection on a monotone ``function`` returning the matching argument."""
        for _ in range(self.bisections):
            middle = (low + high) / 2
            value = function(middle)
            on_low_side = value < target if increasing else value > target
            if on_low_side:
                low = middle
            else:
                high = middle
        return (low + high) / 2

    @staticmethod
    def _baseline(inp, field, default):
        """The baseline the window carries, or the imputation target when it has none."""
        value = inp.get(field)
        if value is None:
            return float(default), "imputed"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return float(default), "imputed"
        if not math.isfinite(number):
            return float(default), "imputed"
        return number, "observed"

    def observed_ratio(self, inp, patient):
        """The propofol:remifentanil rate ratio the window's concentrations imply.

        Each rate is solved independently so that the model's equilibrium concentration
        equals the one the window shows, and the mix is the quotient. Returns
        ``(None, "default")`` when the window carries no concentrations.
        """
        ce_propo = inp.get("ce_propofol")
        ce_remi = inp.get("ce_remi")
        if not _is_positive(ce_propo) or not _is_positive(ce_remi):
            return None, "default"
        u_propo = self._bisect(
            lambda rate: self._steady(patient, rate, 0.0, 0.0).ce_propo,
            0.0, U_PROPO_MAX, float(ce_propo), increasing=True,
        )
        u_remi = self._bisect(
            lambda rate: self._steady(patient, 0.0, rate, 0.0).ce_remi,
            0.0, U_REMI_MAX, float(ce_remi), increasing=True,
        )
        if u_remi <= 0.0:
            return None, "default"
        return u_propo / u_remi, "observed_concentrations"

    def calibrate(self, inp, patient=None):
        """Rates reproducing the BIS and MAP the window shows.

        The *mix* of the two hypnotics comes from the window's effect-site
        concentrations; the *level* is set so the model reproduces its BIS (see the
        module docstring for why the concentrations cannot set the level too).
        """
        patient = patient or self._new_patient(inp)
        target_bis, bis_source = self._baseline(inp, "bis_current", self.default_bis)
        target_map, map_source = self._baseline(inp, "map_current", self.default_map)

        ratio, ratio_source = self.observed_ratio(inp, patient)
        if ratio is None:
            ratio = DEFAULT_RATIO

        # One scalar is enough once the mix is fixed: both drugs move together.
        u_remi = self._bisect(
            lambda rate: self._steady(patient, rate * ratio, rate, 0.0).bis,
            0.0, U_REMI_MAX, target_bis, increasing=False,
        )
        u_propo = u_remi * ratio

        # The haemodynamics: norepinephrine absorbs what the model gets wrong about the
        # baseline MAP as much as any vasopressor the patient may be on.
        u_nore = 0.0
        if self.match_map:
            u_nore = self._bisect(
                lambda rate: self._steady(patient, u_propo, u_remi, rate).map,
                0.0, U_NORE_MAX, target_map, increasing=True,
            )

        signals = self._steady(patient, u_propo, u_remi, u_nore)
        return Calibration(
            u_propo=u_propo,
            u_remi=u_remi,
            u_nore=u_nore,
            bis=signals.bis,
            map=signals.map,
            bis_residual=signals.bis - target_bis,
            map_residual=signals.map - target_map,
            bis_source=bis_source,
            map_source=map_source,
            ratio=ratio,
            ratio_source=ratio_source,
            map_matched_with_nore=self.match_map,
        )

    # -- Simulation ------------------------------------------------------- #

    def _apply(self, calibration, action):
        """The rates resulting from an action (multiplicative on the calibrated one)."""
        if action not in self.action_factors:
            raise ValueError(f"unknown action: {action!r}")
        drug, factor = self.action_factors[action]
        u_propo, u_remi, u_nore = calibration.u_propo, calibration.u_remi, calibration.u_nore
        if drug == "propo":
            u_propo *= factor
        elif drug == "remi":
            u_remi *= factor
        return u_propo, u_remi, u_nore

    def simulate_action(self, patient, calibration, action):
        """Simulate ``action`` for the horizon and read the resulting signals.

        The patient starts at the *calibrated* state and the action is applied during the
        integration, so what comes out is the state five minutes later -- not the
        equilibrium of the new rates, which is what initialising at them would return.
        """
        u_propo, u_remi, u_nore = self._apply(calibration, action)
        baseline = (calibration.u_propo, calibration.u_remi, calibration.u_nore)
        patient.initialized_at_given_input(*baseline)
        signals = self._read(patient, patient.one_step(*baseline, noise=False))
        for _ in range(self.steps):
            signals = self._read(patient, patient.one_step(u_propo, u_remi, u_nore, noise=False))
        return SimulatedOutcome(
            action=action,
            bis=signals.bis,
            map=signals.map,
            co=signals.co,
            tol=signals.tol,
            u_propo=u_propo,
            u_remi=u_remi,
            u_nore=u_nore,
        )

    def simulate(self, inp, actions):
        """Calibrate once and simulate every action. Returns ``(calibration, outcomes)``."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            patient = self._new_patient(inp)
            calibration = self.calibrate(inp, patient=patient)
            outcomes = [
                self.simulate_action(patient, calibration, action) for action in actions
            ]
        return calibration, outcomes
