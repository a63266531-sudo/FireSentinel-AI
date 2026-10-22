"""Executes app.py end-to-end against a stub of Streamlit/Plotly. This checks OUR Python logic
(state handling, pandas operations, function signatures) - it does not validate real Streamlit/Plotly rendering."""
import runpy
import sys
import types
from datetime import date
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent


class Rerun(Exception):
    pass


class Stop(Exception):
    pass


class SS(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


class Ctx:
    def __init__(self, st):
        self._st = st

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        return getattr(self._st, name)


class FakeSt:
    def __init__(self, press=(), arabic=False, overrides=None):
        self.session_state, self.press, self.arabic = SS(), set(press), arabic
        self.overrides = overrides or {}
        self.sidebar = Ctx(self)
        self.calls, self.errors = {}, []

    def __getattr__(self, name):  # any display call (markdown, metric, plotly_chart, ...) is a no-op
        def f(*a, **k):
            self.calls[name] = self.calls.get(name, 0) + 1
            return Ctx(self)
        return f

    def set_page_config(self, **k): pass
    def rerun(self): raise Rerun()
    def stop(self): raise Stop()
    def spinner(self, *a, **k): return Ctx(self)
    def expander(self, *a, **k): return Ctx(self)
    def progress(self, *a, **k): return Ctx(self)
    def columns(self, spec): return [Ctx(self) for _ in range(spec if isinstance(spec, int) else len(spec))]
    def tabs(self, names): return [Ctx(self) for _ in names]
    def button(self, label, **k):  # like Streamlit: True only for the single run after the click
        if label in self.press:
            self.press.discard(label)
            return True
        return False
    def download_button(self, *a, **k): return False
    def radio(self, label, options, index=0, **k): return ("العربية" if self.arabic else "English") if label.startswith("Language") else options[index]
    def selectbox(self, label, options, index=0, **k): return list(options)[index]
    def multiselect(self, label, options, default=None, **k): return list(default) if default is not None else []
    def slider(self, label, lo=0, hi=1, value=None, **k): return self.overrides.get(label, lo if value is None else value)
    def number_input(self, label, value=0, **k): return value
    def text_input(self, label, value="", **k): return value
    def checkbox(self, label, value=False, **k): return value
    def date_input(self, label, value=None, **k): return value
    def file_uploader(self, *a, **k): return None
    def data_editor(self, df, **k): return df
    def cache_data(self, *a, **k): return a[0] if a and callable(a[0]) else (lambda f: f)
    def success(self, *a, **k): self.calls["success"] = self.calls.get("success", 0) + 1; return None
    def error(self, *a, **k): self.errors.append(str(a[0]) if a else ""); return None


def run_app(st, tmp, max_reruns=3):
    stubs = {"streamlit": st, "plotly": mock.MagicMock(), "plotly.express": mock.MagicMock(),
             "plotly.graph_objects": mock.MagicMock(), "plotly.subplots": mock.MagicMock()}
    with mock.patch.dict(sys.modules, stubs), mock.patch("firesentinel.config.MODELS", tmp), \
            mock.patch("firesentinel.experiments.EXPERIMENTS", tmp / "exp"):
        for _ in range(max_reruns):
            try:
                runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
                return "finished"
            except Rerun:
                continue
            except Stop:
                return "stopped"
    return "rerun-limit"


def test_app_landing_then_one_click_demo_renders_every_tab(tmp_path):
    st = FakeSt(press={"▶ One-click demo (offline)"})
    assert run_app(st, tmp_path) == "finished"
    S = st.session_state
    assert S["origin"] == "demo" and S["res"] is not None and len(S["contract"]) > 0 and S["info"] is not None
    assert [e for e in st.errors if not e.startswith("Synthetic demo data")] == []
    assert st.calls["plotly_chart"] >= 8 and st.calls["dataframe"] >= 8


def test_app_buttons_pdf_save_exposure_send_experiment_and_arabic(tmp_path):
    st = FakeSt(press={"▶ One-click demo (offline)"}, overrides={"Use last N days": 60})
    run_app(st, tmp_path)
    ok_errors = lambda: [e for e in st.errors if not e.startswith(("Synthetic demo data", "بيانات تجريبية اصطناعية"))] == []
    st.press = {"Estimate exposure", "Send", "Build PDF with charts", "Save model and model card", "Run experiment"}
    assert run_app(st, tmp_path, 4) == "finished" and ok_errors()
    S = st.session_state
    assert S["pdf"][:4] == b"%PDF" and S["exposure"] is not None and S["meta"]["sha256"] and S["exp_view"]["origin"] == "demo"
    assert any(tmp_path.glob("fs-*.joblib")) and any((tmp_path / "exp").glob("exp-*/experiment.json"))
    st.press = {"Verify reproduction"}
    assert run_app(st, tmp_path, 4) == "finished" and ok_errors()
    st.press = {"Build dataset and train"}
    assert run_app(st, tmp_path, 4) == "finished" and ok_errors() and st.session_state["res"] is not None
    st.press = {"Run / refresh full pipeline"}
    assert run_app(st, tmp_path, 4) == "finished" and ok_errors()
    st.arabic = True
    st.press = set()
    assert run_app(st, tmp_path) == "finished" and ok_errors()


def test_app_demo_generate_button_path(tmp_path):
    st = FakeSt(press={"Generate demo data"})
    assert run_app(st, tmp_path) == "finished" and st.session_state["origin"] == "demo"
