"""
Unit tests for diag_evo — runs without metview/earthkit (mocked).
"""

import sys
import types
import os
import tempfile
import shutil
from datetime import datetime

import pytest

# ---------------------------------------------------------------------------
# Mock heavy dependencies so tests run offline / without Metview
# ---------------------------------------------------------------------------

_mv_mock = types.ModuleType("metview")
_mv_mock.read = lambda *a, **k: None
sys.modules.setdefault("metview", _mv_mock)

_ek = types.ModuleType("earthkit")
_ek_data = types.ModuleType("earthkit.data")
_ek_data.from_source = lambda *a, **k: None
sys.modules.setdefault("earthkit", _ek)
sys.modules.setdefault("earthkit.data", _ek_data)

for _mod in [
    "ipywidgets", "ipyleaflet", "plotly", "plotly.graph_objs",
    "matplotlib", "matplotlib.pyplot", "matplotlib.patches",
    "cartopy", "cartopy.crs", "cartopy.feature",
    "IPython", "IPython.display",
]:
    sys.modules.setdefault(_mod, types.ModuleType(_mod))

sys.modules["ipyleaflet"].Map = type("Map", (), {"__init__": lambda *a, **k: None})
sys.modules["ipyleaflet"].DrawControl = type("DC", (), {"__init__": lambda *a, **k: None})
sys.modules["ipyleaflet"].Rectangle = type("Rect", (), {"__init__": lambda *a, **k: None})
sys.modules["ipyleaflet"].basemaps = types.SimpleNamespace(
    CartoDB=types.SimpleNamespace(Positron={})
)
sys.modules["ipyleaflet"].basemap_to_tiles = lambda *a: None
sys.modules["plotly.graph_objs"].Figure = type("Figure", (), {})
sys.modules["IPython.display"].display = lambda *a, **k: None

# ---------------------------------------------------------------------------
# Imports (after mocks are in place)
# ---------------------------------------------------------------------------

from diag_evo.variables import (
    _split_var_level,
    get_base_var,
    get_level,
    get_variable_settings,
    get_retrieval_settings,
    apply_conversion,
    _resolve_param_id,
    _convert,
    convert_to_display,
    convert_from_display,
    load_variable_settings,
    get_variable_display_name,
)
from diag_evo.settings import (
    load_settings,
    _hex_to_rgb,
    _generate_default_plot_settings,
    get_model_retrieval_settings,
    get_analysis_settings,
    get_climatology_settings,
    register_custom_model,
    unregister_custom_model,
    clear_custom_models,
)
from diag_evo.core import (
    _safe_label,
    _mars_file_tag,
    _model_grib_filename,
    _reference_grib_filename,
    _parse_area_from_dirname,
    get_area_string,
    sanitize_mars_request,
    _should_keep_grid,
    setup_data_directories,
    _find_existing_directory_for_point,
)


# ===================================================================
# variables.py
# ===================================================================

class TestSplitVarLevel:
    def test_surface_vars(self):
        assert _split_var_level("2t") == ("2t", None)
        assert _split_var_level("tp") == ("tp", None)
        assert _split_var_level("msl") == ("msl", None)

    def test_pl_vars_with_level(self):
        assert _split_var_level("z500") == ("z", 500)
        assert _split_var_level("t850") == ("t", 850)

    def test_bare_pl_var(self):
        assert _split_var_level("t") == ("t", None)
        assert _split_var_level("z") == ("z", None)

    def test_empty(self):
        assert _split_var_level("") == ("", None)

    def test_case_insensitive(self):
        assert _split_var_level("T850") == ("t", 850)
        assert _split_var_level("Z500") == ("z", 500)


class TestGetBaseVarLevel:
    def test_base(self):
        assert get_base_var("z500") == "z"
        assert get_base_var("2t") == "2t"

    def test_level(self):
        assert get_level("z500") == 500
        assert get_level("t850") == 850
        assert get_level("2t") is None
        assert get_level("t") is None


class TestRetrievalSettings:
    def test_bare_pl_no_levelist(self):
        """Bug 3 fix: bare 't' should NOT produce levelist=None."""
        req = get_retrieval_settings("t", datetime(2025, 1, 1), 12, [50, -5, 40, 10])
        assert "levelist" not in req

    def test_pl_with_level(self):
        req = get_retrieval_settings("t850", datetime(2025, 1, 1), 12, [50, -5, 40, 10])
        assert req["levelist"] == 850

    def test_sfc_no_levelist(self):
        req = get_retrieval_settings("2t", datetime(2025, 1, 1), 12, [50, -5, 40, 10])
        assert "levelist" not in req

    def test_accumulated_step(self):
        req = get_retrieval_settings("tp", datetime(2025, 1, 1), 12, [50, -5, 40, 10],
                                      step=24, step_start=0)
        assert req["step"] == "0/24"


class TestConversions:
    def test_subtract_273(self):
        assert abs(apply_conversion(300, "subtract_273.15") - 26.85) < 0.01

    def test_multiply_1000(self):
        assert apply_conversion(0.5, "multiply_1000") == 500.0

    def test_none_passthrough(self):
        assert apply_conversion(42, "none") == 42

    def test_convert_to_display_kelvin(self):
        val = _convert(300.0, "2t", "to_display", grib_units="K")
        assert abs(val - 26.85) < 0.01

    def test_convert_from_display_kelvin(self):
        val = _convert(26.85, "2t", "from_display", grib_units="K")
        assert abs(val - 300.0) < 0.01

    def test_convert_roundtrip(self):
        """to_display then from_display should return ~original."""
        orig = 285.0
        display = _convert(orig, "2t", "to_display", grib_units="K")
        back = _convert(display, "2t", "from_display", grib_units="K")
        assert abs(back - orig) < 0.001


class TestResolveParamId:
    def test_known(self):
        settings = load_variable_settings()
        name, vs = _resolve_param_id("167", settings)
        assert name == "2t"

    def test_unknown(self):
        settings = load_variable_settings()
        name, vs = _resolve_param_id("99999", settings)
        assert name is None


class TestDisplayName:
    def test_pl_with_level(self):
        dn = get_variable_display_name("z500")
        assert "500" in dn and "hPa" in dn

    def test_sfc(self):
        dn = get_variable_display_name("2t")
        assert "Temperature" in dn


# ===================================================================
# settings.py
# ===================================================================

class TestHexToRgb:
    def test_valid_hex(self):
        assert _hex_to_rgb("#1f77b4") == (31, 119, 180)
        assert _hex_to_rgb("#FFFFFF") == (255, 255, 255)
        assert _hex_to_rgb("#000000") == (0, 0, 0)

    def test_named_color(self):
        assert _hex_to_rgb("red") is None

    def test_invalid_hex(self):
        assert _hex_to_rgb("#GG1234") is None
        assert _hex_to_rgb("#1f77b") is None  # too short

    def test_rgba_hex(self):
        # 8-char hex not supported by _hex_to_rgb
        assert _hex_to_rgb("#1f77b4ff") is None


class TestGeneratePlotSettings:
    def test_ensemble_hex(self):
        ps = _generate_default_plot_settings("test", True, color="#1f77b4")
        assert "rgba(31" in ps["box"]["fillcolor"]

    def test_ensemble_named_color(self):
        ps = _generate_default_plot_settings("test", True, color="red")
        assert ps["box"]["fillcolor"] == "rgba(128, 128, 128, 0.3)"

    def test_deterministic(self):
        ps = _generate_default_plot_settings("test", False, color="#aabbcc")
        assert ps["marker"]["symbol"] == "diamond"
        assert ps["line"] is None


class TestExpverStandardised:
    def test_ifs_string(self):
        ms = load_settings("model_settings.json")
        assert isinstance(ms["models"]["IFS Control"]["expver"], str)

    def test_clim_string(self):
        ms = load_settings("model_settings.json")
        assert isinstance(ms["climatology"]["expver"], str)


class TestDuplicate3073Removed:
    def test_no_3073_key(self):
        vs = load_settings("variable_settings.json")
        assert "3073" not in vs["variable_settings"]

    def test_lcc_exists(self):
        vs = load_settings("variable_settings.json")
        assert "lcc" in vs["variable_settings"]


class TestJsonCaching:
    def test_variable_settings_cached(self):
        s1 = load_variable_settings()
        s2 = load_variable_settings()
        assert s1 is s2


# ===================================================================
# core.py — naming, sanitisation, grid logic
# ===================================================================

class TestSafeLabel:
    def test_spaces(self):
        assert _safe_label("IFS Control") == "IFS_Control"
        assert _safe_label("AIFS ENS") == "AIFS_ENS"

    def test_no_spaces(self):
        assert _safe_label("DE-ATOS") == "DE-ATOS"


class TestMarsFileTag:
    def test_known_model(self):
        tag = _mars_file_tag("IFS Control")
        assert "od" in tag and "oper" in tag and "fc" in tag


class TestModelGribFilename:
    def test_sfc(self):
        fc = datetime(2025, 3, 27, 0, 0)
        fn = _model_grib_filename("IFS Control", "2t", "sfc", None, fc, 12)
        assert "IFS_Control" in fn
        assert "2t" in fn
        assert "sfc" in fn
        assert "step12" in fn
        assert fn.endswith(".grib")

    def test_with_accumulation(self):
        fc = datetime(2025, 3, 27, 0, 0)
        fn = _model_grib_filename("IFS ENS", "tp", "sfc", None, fc, 24, acc_period=24)
        assert "acc24h" in fn

    def test_with_level(self):
        fc = datetime(2025, 3, 27, 0, 0)
        fn = _model_grib_filename("IFS Control", "t", "pl", 850, fc, 12)
        assert "L850" in fn
        assert "pl" in fn

    def test_no_spaces_in_filename(self):
        fc = datetime(2025, 3, 27, 0, 0)
        fn = _model_grib_filename("AIFS ENS Control", "2t", "sfc", None, fc, 6)
        assert " " not in fn


class TestReferenceGribFilename:
    def test_analysis(self):
        an = get_analysis_settings()
        fn = _reference_grib_filename("Analysis", "2t", an, "sfc", None, "20250327", "1200")
        assert "Analysis" in fn and "od" in fn and "2t" in fn

    def test_climatology_with_level(self):
        cl = get_climatology_settings()
        fn = _reference_grib_filename("Climatology", "t", cl, "pl", 850, "20250327", "1200")
        assert "L850" in fn and "Climatology" in fn


class TestParseAreaFromDirname:
    def test_new_format(self):
        result = _parse_area_from_dirname("2t_N50.0_W8.0_S48.0_E13.0_20250327_1200")
        assert result is not None
        var, area, date, time = result
        assert var == "2t"
        assert date == "20250327"
        assert time == "1200"

    def test_old_format(self):
        result = _parse_area_from_dirname("2t_N50.0_W8.0_S48.0_E13.0_20250327")
        assert result is not None
        var, area, date, time = result
        assert var == "2t"
        assert date == "20250327"
        assert time == "0000"

    def test_invalid(self):
        assert _parse_area_from_dirname("not_a_valid_dirname") is None

    def test_tp_complex_area(self):
        result = _parse_area_from_dirname("tp_N29.88265_W53.02097_S23.88265_E59.02097_20260327_1200")
        assert result is not None
        var, area, date, time = result
        assert var == "tp"
        assert abs(area[0] - 29.88265) < 1e-5


class TestGetAreaString:
    def test_format(self):
        s = get_area_string([50.0, 8.0, 48.0, 13.0])
        assert s == "N50.0_W8.0_S48.0_E13.0"


class TestSanitizeMarsRequest:
    def test_number_removed_non_pf(self):
        req = sanitize_mars_request({"type": "fc", "number": [1, "TO", 50]})
        assert "number" not in req

    def test_number_kept_pf(self):
        req = sanitize_mars_request({"type": "pf", "number": [1, "TO", 50]})
        assert "number" in req

    def test_levelist_removed_non_pl(self):
        req = sanitize_mars_request({"levtype": "sfc", "levelist": 500})
        assert "levelist" not in req

    def test_levelist_kept_pl(self):
        req = sanitize_mars_request({"levtype": "pl", "levelist": 500})
        assert "levelist" in req

    def test_dataset_removed_non_d1(self):
        req = sanitize_mars_request({"class": "od", "dataset": "test"})
        assert "dataset" not in req

    def test_ensemble_key_removed(self):
        req = sanitize_mars_request({"ensemble": True, "type": "fc"})
        assert "ensemble" not in req


class TestShouldKeepGrid:
    def test_pl(self):
        assert _should_keep_grid("pl") is True

    def test_sfc(self):
        assert _should_keep_grid("sfc") is False

    def test_destine(self):
        assert _should_keep_grid("sfc", "DE-ATOS") is True
        assert _should_keep_grid("sfc", "DE-LUMI") is True

    def test_non_destine(self):
        assert _should_keep_grid("sfc", "IFS Control") is False


class TestSetupDataDirectories:
    def test_creates_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grib, obs, plot = setup_data_directories(tmpdir, "2t", "N50_W8_S48_E13", "20250327", "1200")
            assert os.path.isdir(grib)
            assert os.path.isdir(obs)
            assert os.path.isdir(plot)
            assert "1200" in grib  # time included in path

    def test_default_time(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grib, _, _ = setup_data_directories(tmpdir, "tp", "N50_W8_S48_E13", "20250327")
            assert "0000" in grib


class TestFindExistingDirectory:
    def test_finds_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a matching directory
            dirname = "2t_N50.0_W8.0_S48.0_E13.0_20250327_1200"
            grib_dir = os.path.join(tmpdir, dirname, "grib_files")
            os.makedirs(grib_dir)
            # Create a dummy file so directory is non-empty
            open(os.path.join(grib_dir, "dummy.grib"), "w").close()

            area_str, area = _find_existing_directory_for_point(
                tmpdir, "2t", "20250327", [49.0, 10.0], "1200"
            )
            assert area_str is not None
            assert area is not None

    def test_no_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            area_str, area = _find_existing_directory_for_point(
                tmpdir, "2t", "20250327", [49.0, 10.0], "1200"
            )
            assert area_str is None
