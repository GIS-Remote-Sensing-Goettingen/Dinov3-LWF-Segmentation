"""Metadata-helper tests for the DOP20 downloader."""

from __future__ import annotations

import importlib.util
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_get_data_api_module():
    """Load the downloader module from disk without running it as a script.

    Returns:
        object: Imported module object.

    Examples:
        >>> callable(_load_get_data_api_module)
        True
    """

    module_path = REPO_ROOT / "utility" / "get_data_api.py"
    spec = importlib.util.spec_from_file_location("get_data_api", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    """Minimal fake HTTP response for metadata unit tests."""

    def __init__(
        self,
        *,
        text: str,
        status_code: int = 200,
        content_type: str = "application/json",
    ) -> None:
        """Store fake response fields.

        Args:
            text (str): Response body text.
            status_code (int): HTTP status code.
            content_type (str): Response content type.
        """

        self.text = text
        self.status_code = status_code
        self.headers = {"Content-Type": content_type}
        self.content = text.encode("utf-8")


class _FakeSession:
    """Minimal fake session that records the last metadata request."""

    def __init__(self, response: _FakeResponse) -> None:
        """Store the fake response returned by ``get``.

        Args:
            response (_FakeResponse): Response returned for all requests.
        """

        self.response = response
        self.calls: list[dict[str, object]] = []

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any],
        timeout: int,
    ) -> _FakeResponse:
        """Record one fake GET invocation and return the canned response.

        Args:
            url (str): Request URL.
            params: Request query parameters.
            timeout (int): Request timeout in seconds.

        Returns:
            _FakeResponse: The configured fake response.
        """

        self.calls.append({"url": url, "params": dict(params), "timeout": timeout})
        return self.response


def test_build_getfeatureinfo_params_targets_tile_center() -> None:
    """GetFeatureInfo params should query the tile center pixel.

    This checks the request mode, target layer, and center-pixel query
    coordinates used for one metadata request.

    Examples:
        >>> True
        True
    """

    module = _load_get_data_api_module()
    params = module.build_getfeatureinfo_params(453000.0, 6074000.0)

    assert params["REQUEST"] == "GetFeatureInfo"
    assert params["QUERY_LAYERS"] == module.METADATA_LAYER
    assert params["INFO_FORMAT"] == "text/plain"
    assert params["WIDTH"] == module.METADATA_QUERY_WIDTH
    assert params["HEIGHT"] == module.METADATA_QUERY_HEIGHT
    assert params["I"] == module.METADATA_QUERY_WIDTH // 2
    assert params["J"] == module.METADATA_QUERY_HEIGHT // 2
    assert params["BBOX"] == "453000.0,6074000.0,454000.0,6075000.0"


def test_build_getmap_params_uses_full_image_resolution() -> None:
    """GetMap params should keep the original 20 cm image raster size.

    This guards against leaking the smaller metadata query canvas into the
    imagery download path.

    Examples:
        >>> True
        True
    """

    module = _load_get_data_api_module()
    params = module.build_getmap_params(453000.0, 6074000.0)

    assert params["REQUEST"] == "GetMap"
    assert params["LAYERS"] == module.IMAGE_LAYER
    assert params["WIDTH"] == module.WIDTH
    assert params["HEIGHT"] == module.HEIGHT


def test_parse_semicolon_featureinfo_record_extracts_md_dop_columns() -> None:
    """MD DOP text payloads should parse into column/value mappings.

    This captures the live semicolon `text/plain` response shape from the
    official metadata service.

    Examples:
        >>> True
        True
    """

    module = _load_get_data_api_module()
    payload = (
        "@DOP20 FID;A_DATUM;A_DATUM2;E_DATUM; 308;2024-05-15;15.05.2024;" "2024-12-02;"
    )

    record = module.parse_semicolon_featureinfo_record(payload)

    assert record["FID"] == "308"
    assert record["A_DATUM"] == "2024-05-15"
    assert record["A_DATUM2"] == "15.05.2024"


def test_parse_tile_origin_parses_known_tile_coordinate() -> None:
    """Tile-origin parser should accept one known folder tile coordinate.

    This allows smoke tests to request a specific existing tile instead of
    relying on the first tile in the AOI grid.

    Examples:
        >>> True
        True
    """

    module = _load_get_data_api_module()

    assert module.parse_tile_origin("453000,6066000") == (453000.0, 6066000.0)


def test_build_tile_origins_honors_tile_limit() -> None:
    """Tile-origin builder should support small smoke-test batches.

    This keeps the smoke-test wrapper bounded to a predictable 1-2 tile run.

    Examples:
        >>> True
        True
    """

    module = _load_get_data_api_module()
    origins = module.build_tile_origins(tile_limit=2)

    assert len(origins) == 2
    assert all(int(x0) % 1000 == 0 and int(y0) % 1000 == 0 for x0, y0 in origins)


def test_is_blank_tile_detects_uniform_white_tiles() -> None:
    """Blank-tile helper should reject uniform all-white imagery.

    This protects the downloader from writing no-coverage WMS responses as
    valid TIFF tiles.

    Examples:
        >>> True
        True
    """

    module = _load_get_data_api_module()

    assert module.is_blank_tile(np.full((2, 2, 3), 255, dtype=np.uint8)) is True
    assert module.is_blank_tile(np.full((2, 2, 3), 254, dtype=np.uint8)) is False


def test_extract_acquisition_date_supports_nested_json_and_german_text() -> None:
    """Date extraction should handle nested JSON and plain-text metadata.

    The parser should accept both nested JSON payloads and text blobs returned
    by WMS metadata endpoints.

    Examples:
        >>> True
        True
    """

    module = _load_get_data_api_module()

    nested_payload = {
        "feature": {
            "properties": {
                "aufnahme_datum": "2024-06-15",
            }
        }
    }
    assert module.extract_acquisition_date(nested_payload).isoformat() == "2024-06-15"
    assert (
        module.extract_acquisition_date("Befliegungsdatum=15.10.2023").isoformat()
        == "2023-10-15"
    )
    md_payload = (
        "@DOP20 FID;A_DATUM;A_DATUM2;E_DATUM; 308;2024-05-15;15.05.2024;" "2024-12-02;"
    )
    assert module.extract_acquisition_date(md_payload).isoformat() == "2024-05-15"


def test_detect_wms_service_exception_maps_not_queryable_layer() -> None:
    """WMS exception detection should normalize non-queryable-layer errors.

    This distinguishes a metadata-service failure from a real no-date payload.

    Examples:
        >>> True
        True
    """

    module = _load_get_data_api_module()
    payload = (
        '<?xml version="1.0"?><ServiceExceptionReport>'
        "<ServiceException>layer sh_dop20_rgb is not queryable</ServiceException>"
        "</ServiceExceptionReport>"
    )

    assert module.detect_wms_service_exception(payload) == "layer_not_queryable"


def test_is_preferred_acquisition_date_rejects_october() -> None:
    """Preferred-date helper should accept summer dates and reject October.

    The season gate must allow spring/summer months and reject October.

    Examples:
        >>> True
        True
    """

    module = _load_get_data_api_module()

    assert module.is_preferred_acquisition_date(module.date(2024, 4, 19)) is False
    assert module.is_preferred_acquisition_date(module.date(2024, 4, 20)) is True
    assert module.is_preferred_acquisition_date(module.date(2024, 6, 15)) is True
    assert module.is_preferred_acquisition_date(module.date(2024, 10, 1)) is False
    assert module.is_preferred_acquisition_date(None) is False


def test_fetch_tile_metadata_extracts_date_and_season_flag() -> None:
    """Metadata fetch should parse acquisition date and season from response.

    This verifies the fake metadata request returns a parsed date plus the
    derived season-acceptance flag.

    Examples:
        >>> True
        True
    """

    module = _load_get_data_api_module()
    session = _FakeSession(
        _FakeResponse(
            text='{"properties": {"flugdatum": "2024-07-01"}}',
            content_type="application/json; charset=utf-8",
        )
    )

    result = module.fetch_tile_metadata(session, 453000.0, 6074000.0, timeout_s=12)

    assert result["status_code"] == 200
    assert result["acquisition_date"] == "2024-07-01"
    assert result["season_ok"] is True
    assert result["content_type"] == "application/json; charset=utf-8"
    assert session.calls[0]["timeout"] == 12
    assert session.calls[0]["url"] == module.METADATA_WMS_URL
    assert session.calls[0]["params"]["LAYERS"] == module.METADATA_LAYER
    assert session.calls[0]["params"]["INFO_FORMAT"] == "text/plain"
    assert session.calls[0]["params"]["REQUEST"] == "GetFeatureInfo"


def test_fetch_tile_metadata_parses_live_md_dop_text_payload() -> None:
    """Metadata fetch should parse the verified MD DOP text/plain format.

    This matches the live `MD DOP` response structure that exposes `A_DATUM`
    for season filtering.

    Examples:
        >>> True
        True
    """

    module = _load_get_data_api_module()
    session = _FakeSession(
        _FakeResponse(
            text=(
                "@DOP20 FID;Shape;ID;BILDFLUG;A_DATUM;A_DATUM2;E_DATUM;E_DATUM2;"
                " 308;Polygon;324536066;2024;2024-05-15;15.05.2024;2024-12-02;"
                "02.12.2024;"
            ),
            content_type="text/plain",
        )
    )

    result = module.fetch_tile_metadata(session, 453000.0, 6066000.0, timeout_s=10)

    assert result["status_code"] == 200
    assert result["content_type"] == "text/plain"
    assert result["metadata_error"] is None
    assert result["acquisition_date"] == "2024-05-15"
    assert result["season_ok"] is True


def test_fetch_tile_metadata_reports_wms_service_exception() -> None:
    """Metadata fetch should expose WMS exception payloads explicitly.

    This avoids treating a non-queryable layer error as a normal metadata
    response with a missing date.

    Examples:
        >>> True
        True
    """

    module = _load_get_data_api_module()
    session = _FakeSession(
        _FakeResponse(
            text=(
                '<?xml version="1.0"?><ServiceExceptionReport>'
                "<ServiceException>layer sh_dop20_rgb is not queryable</ServiceException>"
                "</ServiceExceptionReport>"
            ),
            content_type="text/xml; charset=utf-8",
        )
    )

    result = module.fetch_tile_metadata(session, 453000.0, 6066000.0, timeout_s=10)

    assert result["status_code"] == 200
    assert result["content_type"] == "text/xml; charset=utf-8"
    assert result["metadata_error"] == "layer_not_queryable"
    assert result["acquisition_date"] is None
    assert result["season_ok"] is False


def test_evaluate_tile_metadata_for_download_accepts_summer_dates() -> None:
    """Download decisions should accept metadata-approved summer imagery.

    This keeps the old image WMS download path gated by the metadata service.

    Examples:
        >>> True
        True
    """

    module = _load_get_data_api_module()

    decision = module.evaluate_tile_metadata_for_download(
        {
            "metadata_error": None,
            "acquisition_date": "2024-05-15",
            "season_ok": True,
        }
    )

    assert decision == (True, "season_ok")


def test_evaluate_tile_metadata_for_download_rejects_october() -> None:
    """Download decisions should reject non-summer imagery by metadata date.

    This is the main protection against October tiles entering the dataset.

    Examples:
        >>> True
        True
    """

    module = _load_get_data_api_module()

    decision = module.evaluate_tile_metadata_for_download(
        {
            "metadata_error": None,
            "acquisition_date": "2024-10-15",
            "season_ok": False,
        }
    )

    assert decision == (False, "season_rejected")


def test_fetch_and_write_tile_rejects_uniform_white_response(
    tmp_path: Path,
) -> None:
    """Tile writer should reject all-white WMS images as blank responses.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.

    Examples:
        >>> True
        True
    """

    module = _load_get_data_api_module()
    image = Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8))
    image_buffer = BytesIO()
    image.save(image_buffer, format="PNG")
    response = _FakeResponse(text="")
    response.headers = {"Content-Type": "image/png"}
    response.content = image_buffer.getvalue()
    image_session = _FakeSession(response)
    metadata_session = _FakeSession(
        _FakeResponse(
            text=(
                "@DOP20 FID;A_DATUM;A_DATUM2;E_DATUM; "
                "308;2024-05-15;15.05.2024;2024-12-02;"
            ),
            content_type="text/plain",
        )
    )

    result = module.fetch_and_write_tile(
        image_session,
        metadata_session,
        453000.0,
        6066000.0,
        out_dir=tmp_path,
        timeout_s=5,
        metadata_timeout_s=5,
    )

    assert result.startswith("FAIL_BLANK")
    assert not any(tmp_path.glob("*.tif"))


def test_fetch_and_write_tile_skips_october_tile_before_image_download(
    tmp_path: Path,
) -> None:
    """Tile writer should skip October imagery before downloading image bytes.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.

    Examples:
        >>> True
        True
    """

    module = _load_get_data_api_module()
    metadata_session = _FakeSession(
        _FakeResponse(
            text=(
                "@DOP20 FID;A_DATUM;A_DATUM2;E_DATUM; "
                "308;2024-10-15;15.10.2024;2024-12-02;"
            ),
            content_type="text/plain",
        )
    )
    image_session = _FakeSession(_FakeResponse(text="", content_type="image/png"))

    result = module.fetch_and_write_tile(
        image_session,
        metadata_session,
        453000.0,
        6066000.0,
        out_dir=tmp_path,
        timeout_s=5,
        metadata_timeout_s=5,
    )

    assert result.startswith("SKIP_SEASON")
    assert "date=2024-10-15" in result
    assert image_session.calls == []
    assert not any(tmp_path.glob("*.tif"))


def test_fetch_and_write_tile_fails_when_metadata_service_errors(
    tmp_path: Path,
) -> None:
    """Tile writer should fail fast when metadata lookup cannot validate date.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.

    Examples:
        >>> True
        True
    """

    module = _load_get_data_api_module()
    metadata_session = _FakeSession(
        _FakeResponse(
            text=(
                '<?xml version="1.0"?><ServiceExceptionReport>'
                "<ServiceException>layer DOP20 is not queryable</ServiceException>"
                "</ServiceExceptionReport>"
            ),
            content_type="text/xml",
        )
    )
    image_session = _FakeSession(_FakeResponse(text="", content_type="image/png"))

    result = module.fetch_and_write_tile(
        image_session,
        metadata_session,
        453000.0,
        6066000.0,
        out_dir=tmp_path,
        timeout_s=5,
        metadata_timeout_s=5,
    )

    assert result.startswith("FAIL_METADATA")
    assert "reason=layer_not_queryable" in result
    assert image_session.calls == []
    assert not any(tmp_path.glob("*.tif"))


def test_main_honors_smoke_test_cli_flags(tmp_path: Path, monkeypatch) -> None:
    """CLI should forward smoke-test flags into the downloader runtime.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
        monkeypatch: Pytest fixture used to replace runtime helpers.

    Examples:
        >>> True
        True
    """

    module = _load_get_data_api_module()
    captured: dict[str, object] = {}

    def _fake_configure_logging(log_file: Path) -> None:
        """Capture the configured smoke-test log file path.

        Args:
            log_file (Path): Requested log file path.
        """

        captured["log_file"] = log_file

    def _fake_run_download(**kwargs):
        """Capture the forwarded downloader kwargs.

        Args:
            **kwargs: Forwarded downloader keyword arguments.

        Returns:
            tuple[int, int]: Fake success and failure counts.
        """

        captured.update(kwargs)
        return 0, 0

    monkeypatch.setattr(module, "configure_logging", _fake_configure_logging)
    monkeypatch.setattr(module, "run_download", _fake_run_download)

    out_dir = tmp_path / "download_smoke"
    module.main(
        [
            "--out-dir",
            str(out_dir),
            "--max-workers",
            "2",
            "--max-tiles",
            "2",
            "--tile-origin",
            "453000,6066000",
        ]
    )

    assert captured["log_file"] == out_dir / "download.log"
    assert captured["out_dir"] == out_dir
    assert captured["max_workers"] == 2
    assert captured["tile_limit"] == 2
    assert captured["tile_origins"] == [(453000.0, 6066000.0)]
