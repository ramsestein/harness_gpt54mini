# -*- coding: utf-8 -*-
"""``zero_shot`` with the effect-site fields cut at -30 s: does the t0 Ce sample matter?

View 4 shows the effect-site concentration at the decision point, and for remifentanil the
t0 sample already moves with the action (README, "Effect-site leak at t0"). The tree controls
read those fields through the control view; the language models stay on view 4. This
strategy measures whether that matters to them: it is ``zero_shot`` (same template, same
prose, same call) with every effect-site field of the window -- ``ce_*``, ``propofol_ce_*``,
``remi_ce_*``, and the ``*_ce_trend`` words of the prose -- recomputed by
``src.control.control_view`` from the Ce samples at ``t <= CE_CUT_SEC`` only. Nothing else
changes, so the paired difference against the ``zero_shot`` run of the same model and effort
is the effect of the cut.

Two differences come with the cut rather than on top of it: the dataset's ``ce_*`` scalars
are the native-resolution value at t0 (after the cut only the 30 s series is left), and the
dataset's trend codes agree with the control view's rule (|short slope| > 0.05 per minute)
on ~98.6 % of the training windows.
"""

from harness.zero_shot.strategy import ZeroShotStrategy
from src import prompting
from src.control import control_view

#: The dataset's trend words, by trend code; no code (no slope) is ``unknown``, as in view 4.
TREND_WORDS = {-1.0: "falling", 0.0: "stable", 1.0: "rising"}


def cut_input(window, cut=control_view.CE_CUT_SEC):
    """The window's input with its effect-site fields recomputed up to ``cut`` seconds.

    A field the cut leaves without samples becomes ``None`` (missing, as in view 4: the prose
    cannot format a NaN). A window without ``raw_input`` (a hand-built test double) keeps
    its input.
    """
    inp = dict(getattr(window, "input", None) or {})
    raw = getattr(window, "raw_input", None)
    if raw is None:
        return inp
    values = control_view.ce_features(raw, cut)
    inp.update({name: None if value != value else value for name, value in values.items()})
    for prefix in ("propofol_ce_", "remi_ce_"):
        inp[prefix + "trend"] = TREND_WORDS.get(values[prefix + "trend_code"], "unknown")
    return inp


class ZeroShotCeCutStrategy(ZeroShotStrategy):
    """``zero_shot`` on the window with its effect-site fields cut at ``CE_CUT_SEC``."""

    name = "zero_shot_ce_cut"
    description = (
        "zero_shot ablation: the same call, with every effect-site field (ce_*, *_ce_*, the "
        "Ce trend words) recomputed from the Ce samples up to -30 s, the cut the tree "
        "controls read."
    )

    def build_messages(self, window):
        """Chat messages for one window, rendered from the cut input."""
        template = prompting.load_template(self.prompt_file)
        return prompting.build_messages_from_case(prompting.render_case(cut_input(window)),
                                                  template)

    def predict(self, window):
        prediction = super().predict(window)
        prediction.meta.update(ce_cut_sec=control_view.CE_CUT_SEC,
                               control_view_version=control_view.CONTROL_VIEW_VERSION)
        return prediction
