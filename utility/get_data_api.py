"""Download 1 km DOP20 GeoTIFF tiles and expose metadata-query helpers.

Examples:
    >>> WIDTH
    5000
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pyproj
import rasterio
import requests
from PIL import Image
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from requests.adapters import HTTPAdapter, Retry

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

IMAGE_WMS_URL = "https://dienste.gdi-sh.de/WMS_SH_DOP20col_OpenGBD"
IMAGE_LAYER = "sh_dop20_rgb"
METADATA_WMS_URL = "https://service.gdi-sh.de/WMS_SH_MD_DOP"
METADATA_LAYER = "DOP20"
CRS_EPSG = 25832
TILE_M = 1000
GSD_M = 0.2
WIDTH = HEIGHT = int(round(TILE_M / GSD_M))
METADATA_QUERY_WIDTH = METADATA_QUERY_HEIGHT = 1000
MAX_WORKERS = 12
TIMEOUT_S = 90
OUT_DIR = Path("/mnt/ceph-hdd/projects/mthesis_davide_mattioli/patches_mt")
LOG_FILE = OUT_DIR / "download.log"
BBOX_LL = (8.1, 53.278, 11.56, 55.36)
ACCEPTED_MONTHS = frozenset({4, 5, 6, 7, 8, 9})
APRIL_MIN_DAY = 20
DATE_KEY_HINTS = (
    "datum",
    "date",
    "beflieg",
    "flug",
    "aufnahme",
    "acquisition",
)
ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
SLASH_DATE_RE = re.compile(r"\b(\d{4})/(\d{2})/(\d{2})\b")
EURO_DATE_RE = re.compile(r"\b(\d{2})\.(\d{2})\.(\d{4})\b")
LOGGER = logging.getLogger(__name__)


def parse_tile_origin(spec: str) -> tuple[float, float]:
    """Parse one ``x,y`` tile-origin string.

    Args:
        spec (str): Tile-origin string in ``x,y`` form.

    Returns:
        tuple[float, float]: Parsed tile origin coordinates.

    Examples:
        >>> parse_tile_origin("453000,6066000")
        (453000.0, 6066000.0)
    """

    parts = [part.strip() for part in spec.split(",")]
    if len(parts) != 2:
        raise ValueError(f"invalid tile origin: {spec!r}")
    return float(parts[0]), float(parts[1])


def configure_logging(log_file: Path = LOG_FILE) -> None:
    """Configure console and file logging for the downloader script.

    Args:
        log_file (Path): Destination log file path.

    Examples:
        >>> callable(configure_logging)
        True
    """

    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
        force=True,
    )


def make_session() -> requests.Session:
    """Build one pooled HTTP session with retry behavior.

    Returns:
        requests.Session: Configured session.

    Examples:
        >>> isinstance(make_session(), requests.Session)
        True
    """

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "dop20-tiler/1.0",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
    )
    retry = Retry(
        total=10,
        backoff_factor=0.7,
        status_forcelist=(502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def project_and_snap_bbox(
    bbox_ll: tuple[float, float, float, float] = BBOX_LL,
    tile_size_m: int = TILE_M,
    crs_epsg: int = CRS_EPSG,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Project one lon/lat AOI and snap it outward to the tile grid.

    Args:
        bbox_ll (tuple[float, float, float, float]): AOI in lon/lat order.
        tile_size_m (int): Tile edge length in meters.
        crs_epsg (int): Projected CRS EPSG code.

    Returns:
        tuple[float, float, float, float, float, float, float, float]:
            Projected min/max bounds followed by snapped min/max bounds.

    Examples:
        >>> bounds = project_and_snap_bbox()
        >>> len(bounds)
        8
    """

    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{crs_epsg}",
        always_xy=True,
    )
    min_x, min_y = transformer.transform(bbox_ll[0], bbox_ll[1])
    max_x, max_y = transformer.transform(bbox_ll[2], bbox_ll[3])
    gx0 = math.floor(min_x / tile_size_m) * tile_size_m
    gy0 = math.floor(min_y / tile_size_m) * tile_size_m
    gx1 = math.ceil(max_x / tile_size_m) * tile_size_m
    gy1 = math.ceil(max_y / tile_size_m) * tile_size_m
    return min_x, min_y, max_x, max_y, gx0, gy0, gx1, gy1


def build_grid_coordinates(
    bbox_ll: tuple[float, float, float, float] = BBOX_LL,
    tile_size_m: int = TILE_M,
    crs_epsg: int = CRS_EPSG,
) -> tuple[np.ndarray, np.ndarray]:
    """Build snapped x/y tile origins for the configured AOI.

    Args:
        bbox_ll (tuple[float, float, float, float]): AOI in lon/lat order.
        tile_size_m (int): Tile edge length in meters.
        crs_epsg (int): Projected CRS EPSG code.

    Returns:
        tuple[np.ndarray, np.ndarray]: X and Y tile-origin arrays.

    Examples:
        >>> xs, ys = build_grid_coordinates()
        >>> int(xs[0]) % 1000 == 0 and int(ys[0]) % 1000 == 0
        True
    """

    _, _, _, _, gx0, gy0, gx1, gy1 = project_and_snap_bbox(
        bbox_ll=bbox_ll,
        tile_size_m=tile_size_m,
        crs_epsg=crs_epsg,
    )
    xs = np.arange(gx0, gx1, tile_size_m)
    ys = np.arange(gy0, gy1, tile_size_m)
    return xs, ys


def build_tile_origins(
    bbox_ll: tuple[float, float, float, float] = BBOX_LL,
    *,
    tile_limit: int | None = None,
) -> list[tuple[float, float]]:
    """Build ordered tile origins for the configured AOI.

    Args:
        bbox_ll (tuple[float, float, float, float]): AOI in lon/lat order.
        tile_limit (int | None): Optional limit on returned tile count.

    Returns:
        list[tuple[float, float]]: Ordered `(x0, y0)` tile origins.

    Examples:
        >>> build_tile_origins(
        ...     (8.1, 53.278, 8.11, 53.279),
        ...     tile_limit=1,
        ... )[0][0] % 1000 == 0
        True
    """

    if tile_limit is not None and tile_limit <= 0:
        raise ValueError("tile_limit must be > 0 when provided")
    xs, ys = build_grid_coordinates(bbox_ll=bbox_ll)
    origins = [(float(x0), float(y0)) for x0 in xs for y0 in ys]
    return origins if tile_limit is None else origins[:tile_limit]


def is_blank_tile(arr: np.ndarray, *, blank_value: int = 255) -> bool:
    """Return whether one downloaded tile is a uniform blank image.

    Args:
        arr (np.ndarray): Downloaded image array.
        blank_value (int): Pixel value treated as blank/no-coverage.

    Returns:
        bool: True when the tile is uniformly blank.

    Examples:
        >>> is_blank_tile(np.full((3, 2, 2), 255, dtype=np.uint8))
        True
        >>> is_blank_tile(np.array([[[255, 1]]], dtype=np.uint8))
        False
    """

    return bool(arr.size > 0 and np.all(arr == blank_value))


def build_tile_bbox(
    x0: float,
    y0: float,
    tile_size_m: int = TILE_M,
) -> tuple[float, float, float, float]:
    """Build one projected tile bounding box.

    Args:
        x0 (float): Tile origin x coordinate.
        y0 (float): Tile origin y coordinate.
        tile_size_m (int): Tile edge length in meters.

    Returns:
        tuple[float, float, float, float]: Tile bbox in projected meters.

    Examples:
        >>> build_tile_bbox(0.0, 0.0)
        (0.0, 0.0, 1000.0, 1000.0)
    """

    return x0, y0, x0 + tile_size_m, y0 + tile_size_m


def build_getmap_params(
    x0: float,
    y0: float,
    *,
    tile_size_m: int = TILE_M,
    width: int = WIDTH,
    height: int = HEIGHT,
    crs_epsg: int = CRS_EPSG,
    layer: str = IMAGE_LAYER,
) -> dict[str, Any]:
    """Build WMS GetMap parameters for one tile.

    Args:
        x0 (float): Tile origin x coordinate.
        y0 (float): Tile origin y coordinate.
        tile_size_m (int): Tile edge length in meters.
        width (int): Output image width in pixels.
        height (int): Output image height in pixels.
        crs_epsg (int): Projected CRS EPSG code.
        layer (str): WMS layer name.

    Returns:
        dict[str, Any]: GetMap parameter mapping.

    Examples:
        >>> params = build_getmap_params(1000.0, 2000.0)
        >>> params["REQUEST"]
        'GetMap'
    """

    bbox = build_tile_bbox(x0, y0, tile_size_m=tile_size_m)
    return {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "STYLES": "",
        "CRS": f"EPSG:{crs_epsg}",
        "BBOX": ",".join(map(str, bbox)),
        "WIDTH": width,
        "HEIGHT": height,
        "FORMAT": "image/png",
    }


def build_getfeatureinfo_params(
    x0: float,
    y0: float,
    *,
    tile_size_m: int = TILE_M,
    width: int = METADATA_QUERY_WIDTH,
    height: int = METADATA_QUERY_HEIGHT,
    crs_epsg: int = CRS_EPSG,
    layer: str = METADATA_LAYER,
    info_format: str = "text/plain",
) -> dict[str, Any]:
    """Build WMS GetFeatureInfo parameters for the tile center pixel.

    Args:
        x0 (float): Tile origin x coordinate.
        y0 (float): Tile origin y coordinate.
        tile_size_m (int): Tile edge length in meters.
        width (int): Virtual query image width in pixels.
        height (int): Virtual query image height in pixels.
        crs_epsg (int): Projected CRS EPSG code.
        layer (str): WMS layer name.
        info_format (str): Requested feature-info payload format.

    Returns:
        dict[str, Any]: GetFeatureInfo parameter mapping.

    Examples:
        >>> params = build_getfeatureinfo_params(0.0, 0.0)
        >>> (params["I"], params["J"])
        (500, 500)
    """

    params = build_getmap_params(
        x0,
        y0,
        tile_size_m=tile_size_m,
        width=width,
        height=height,
        crs_epsg=crs_epsg,
        layer=layer,
    )
    params.update(
        {
            "REQUEST": "GetFeatureInfo",
            "QUERY_LAYERS": layer,
            "INFO_FORMAT": info_format,
            "FEATURE_COUNT": 10,
            "I": width // 2,
            "J": height // 2,
        }
    )
    return params


def _candidate_date_strings(payload: Any) -> list[str]:
    """Collect date-like strings recursively from a response payload.

    Args:
        payload (Any): Nested JSON/text payload.

    Returns:
        list[str]: Candidate strings that may contain acquisition dates.
    """

    candidates: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_lower = str(key).lower()
            if any(hint in key_lower for hint in DATE_KEY_HINTS):
                candidates.append(str(value))
            candidates.extend(_candidate_date_strings(value))
        return candidates
    if isinstance(payload, list):
        for item in payload:
            candidates.extend(_candidate_date_strings(item))
        return candidates
    if isinstance(payload, bytes):
        try:
            candidates.append(payload.decode("utf-8", errors="ignore"))
        except Exception:
            return candidates
        return candidates
    if payload is not None:
        candidates.append(str(payload))
    return candidates


def _parse_date_string(value: str) -> date | None:
    """Parse one candidate string into a date when possible.

    Args:
        value (str): Candidate value string.

    Returns:
        date | None: Parsed acquisition date.
    """

    for pattern, fmt in (
        (ISO_DATE_RE, "%Y-%m-%d"),
        (SLASH_DATE_RE, "%Y/%m/%d"),
        (EURO_DATE_RE, "%d.%m.%Y"),
    ):
        match = pattern.search(value)
        if match is None:
            continue
        return datetime.strptime(match.group(0), fmt).date()
    return None


def parse_semicolon_featureinfo_record(payload: str) -> dict[str, str]:
    """Parse the `text/plain` semicolon table returned by `MD DOP`.

    Args:
        payload (str): Raw `GetFeatureInfo` text payload.

    Returns:
        dict[str, str]: Mapping from column names to row values.

    Examples:
        >>> parse_semicolon_featureinfo_record('@DOP20 A_DATUM;A_DATUM2; 2024-05-15;15.05.2024;')
        {'A_DATUM': '2024-05-15', 'A_DATUM2': '15.05.2024'}
    """

    lines = [line.strip() for line in payload.splitlines() if line.strip()]
    target_line = next((line for line in lines if line.startswith("@DOP20 ")), "")
    if not target_line:
        return {}
    row_text = target_line[len("@DOP20 ") :]
    tokens = [token.strip() for token in row_text.split(";")]
    while tokens and tokens[-1] == "":
        tokens.pop()
    if len(tokens) < 2:
        return {}
    if len(tokens) % 2 != 0:
        return {}
    split_index = len(tokens) // 2
    headers = tokens[:split_index]
    values = tokens[split_index:]
    return {
        header: value for header, value in zip(headers, values, strict=False) if header
    }


def extract_acquisition_date(payload: Any) -> date | None:
    """Extract one acquisition date from JSON, XML-like text, or plain text.

    Args:
        payload (Any): Response payload.

    Returns:
        date | None: First parsed acquisition date.

    Examples:
        >>> extract_acquisition_date({'flugdatum': '2024-06-15'}).isoformat()
        '2024-06-15'
        >>> extract_acquisition_date('Befliegungsdatum=15.10.2023').isoformat()
        '2023-10-15'
    """

    if isinstance(payload, (str, bytes)):
        text = (
            payload.decode("utf-8", errors="ignore")
            if isinstance(payload, bytes)
            else payload
        )
        featureinfo_record = parse_semicolon_featureinfo_record(text)
        for key in ("A_DATUM", "A_DATUM2", "E_DATUM", "E_DATUM2"):
            parsed_date = _parse_date_string(featureinfo_record.get(key, ""))
            if parsed_date is not None:
                return parsed_date
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = text
    else:
        parsed = payload
    for candidate in _candidate_date_strings(parsed):
        parsed_date = _parse_date_string(candidate)
        if parsed_date is not None:
            return parsed_date
    return None


def detect_wms_service_exception(payload: str) -> str | None:
    """Detect a WMS service-exception payload and return a reason code.

    Args:
        payload (str): Response text payload.

    Returns:
        str | None: Normalized metadata error code when the payload is a WMS
            exception report.

    Examples:
        >>> detect_wms_service_exception(
        ...     '<ServiceExceptionReport><ServiceException>'
        ...     'layer sh_dop20_rgb is not queryable'
        ...     '</ServiceException></ServiceExceptionReport>'
        ... )
        'layer_not_queryable'
        >>> detect_wms_service_exception(
        ...     '{"properties": {"flugdatum": "2024-07-01"}}'
        ... ) is None
        True
    """

    payload_lower = payload.lower()
    if "serviceexceptionreport" not in payload_lower:
        return None
    if "not queryable" in payload_lower:
        return "layer_not_queryable"
    return "service_exception"


def is_preferred_acquisition_date(
    acquisition_date: date | None,
    *,
    accepted_months: frozenset[int] = ACCEPTED_MONTHS,
    april_min_day: int = APRIL_MIN_DAY,
) -> bool:
    """Return whether an acquisition date falls in the accepted season.

    Args:
        acquisition_date (date | None): Candidate acquisition date.
        accepted_months (frozenset[int]): Allowed calendar months.
        april_min_day (int): First accepted April day.

    Returns:
        bool: True when the date is accepted.

    Examples:
        >>> is_preferred_acquisition_date(date(2024, 6, 15))
        True
        >>> is_preferred_acquisition_date(date(2024, 4, 19))
        False
        >>> is_preferred_acquisition_date(date(2024, 4, 20))
        True
        >>> is_preferred_acquisition_date(date(2024, 10, 1))
        False
    """

    if acquisition_date is None:
        return False
    month = int(acquisition_date.month)
    if month not in accepted_months:
        return False
    if month == 4:
        return int(acquisition_date.day) >= april_min_day
    return True


def fetch_tile_metadata(
    session: requests.Session,
    x0: float,
    y0: float,
    *,
    timeout_s: int = TIMEOUT_S,
    info_format: str = "text/plain",
    metadata_url: str = METADATA_WMS_URL,
    metadata_layer: str = METADATA_LAYER,
) -> dict[str, Any]:
    """Fetch feature-info metadata for one tile and extract acquisition date.

    Args:
        session (requests.Session): HTTP session.
        x0 (float): Tile origin x coordinate.
        y0 (float): Tile origin y coordinate.
        timeout_s (int): Request timeout in seconds.
        info_format (str): Requested feature-info payload format.
        metadata_url (str): Metadata WMS service URL.
        metadata_layer (str): Queryable metadata layer name.

    Returns:
        dict[str, Any]: Metadata fetch summary.

    Examples:
        >>> callable(fetch_tile_metadata)
        True
    """

    params = build_getfeatureinfo_params(
        x0,
        y0,
        layer=metadata_layer,
        info_format=info_format,
    )
    response = session.get(metadata_url, params=params, timeout=timeout_s)
    payload_text = response.text
    metadata_error = detect_wms_service_exception(payload_text)
    acquisition_date = extract_acquisition_date(payload_text)
    return {
        "status_code": int(response.status_code),
        "content_type": str(response.headers.get("Content-Type", "")),
        "metadata_error": metadata_error,
        "acquisition_date": (
            acquisition_date.isoformat() if acquisition_date is not None else None
        ),
        "season_ok": (
            False
            if metadata_error is not None
            else is_preferred_acquisition_date(acquisition_date)
        ),
        "body_excerpt": payload_text[:500],
    }


def evaluate_tile_metadata_for_download(metadata: dict[str, Any]) -> tuple[bool, str]:
    """Return whether one tile should be downloaded from its metadata.

    Args:
        metadata (dict[str, Any]): Metadata lookup summary.

    Returns:
        tuple[bool, str]: Download decision and normalized reason code.

    Examples:
        >>> evaluate_tile_metadata_for_download(
        ...     {"metadata_error": None, "acquisition_date": "2024-05-15", "season_ok": True}
        ... )
        (True, 'season_ok')
        >>> evaluate_tile_metadata_for_download(
        ...     {"metadata_error": None, "acquisition_date": "2024-10-15", "season_ok": False}
        ... )
        (False, 'season_rejected')
    """

    metadata_error = metadata.get("metadata_error")
    if metadata_error:
        return False, str(metadata_error)
    if metadata.get("acquisition_date") in (None, ""):
        return False, "missing_acquisition_date"
    if bool(metadata.get("season_ok")):
        return True, "season_ok"
    return False, "season_rejected"


def fetch_and_write_tile(
    image_session: requests.Session,
    metadata_session: requests.Session,
    x0: float,
    y0: float,
    *,
    out_dir: Path = OUT_DIR,
    timeout_s: int = TIMEOUT_S,
    metadata_timeout_s: int = TIMEOUT_S,
) -> str:
    """Download and write one tile GeoTIFF.

    Args:
        image_session (requests.Session): Imagery WMS HTTP session.
        metadata_session (requests.Session): Metadata WMS HTTP session.
        x0 (float): Tile origin x coordinate.
        y0 (float): Tile origin y coordinate.
        out_dir (Path): Destination directory.
        timeout_s (int): Request timeout in seconds.
        metadata_timeout_s (int): Metadata request timeout in seconds.

    Returns:
        str: One short status line.

    Examples:
        >>> callable(fetch_and_write_tile)
        True
    """

    params = build_getmap_params(x0, y0)
    bbox = build_tile_bbox(x0, y0)
    out_path = out_dir / f"dop20_{int(x0)}_{int(y0)}_1km_20cm.tif"
    if out_path.exists():
        return f"SKIP {out_path.name}"

    start_time = time.time()
    metadata = fetch_tile_metadata(
        metadata_session,
        x0,
        y0,
        timeout_s=metadata_timeout_s,
    )
    should_download, reason = evaluate_tile_metadata_for_download(metadata)
    acquisition_date = metadata.get("acquisition_date")
    if not should_download:
        elapsed_s = time.time() - start_time
        status_prefix = (
            "SKIP_SEASON" if reason == "season_rejected" else "FAIL_METADATA"
        )
        date_suffix = (
            f" date={acquisition_date}" if acquisition_date not in (None, "") else ""
        )
        return f"{status_prefix} {out_path.name} reason={reason}{date_suffix} ({elapsed_s:.1f}s)"

    response = image_session.get(IMAGE_WMS_URL, params=params, timeout=timeout_s)
    if response.status_code != 200 or not response.headers.get(
        "Content-Type", ""
    ).startswith("image/"):
        elapsed_s = time.time() - start_time
        detail = response.text[:220].replace("\n", " ")
        return "FAIL %s,%s [%s] ct=%s (%.1fs) %s" % (
            int(x0),
            int(y0),
            response.status_code,
            response.headers.get("Content-Type"),
            elapsed_s,
            detail,
        )

    img = Image.open(BytesIO(response.content))
    arr = np.array(img)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    if is_blank_tile(arr):
        elapsed_s = time.time() - start_time
        return f"FAIL_BLANK {out_path.name} ({elapsed_s:.1f}s)"
    transform = from_bounds(*bbox, width=WIDTH, height=HEIGHT)
    crs = CRS.from_epsg(CRS_EPSG)
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=int(arr.shape[2]),
        dtype=arr.dtype,
        crs=crs,
        transform=transform,
        tiled=True,
        compress="DEFLATE",
        predictor=2,
    ) as dst:
        for band_idx in range(int(arr.shape[2])):
            dst.write(arr[:, :, band_idx], band_idx + 1)
    elapsed_s = time.time() - start_time
    return f"DONE {out_path.name} ({elapsed_s:.1f}s)"


def run_download(
    *,
    out_dir: Path = OUT_DIR,
    max_workers: int = MAX_WORKERS,
    bbox_ll: tuple[float, float, float, float] = BBOX_LL,
    tile_limit: int | None = None,
    tile_origins: list[tuple[float, float]] | None = None,
) -> tuple[int, int]:
    """Run the tiled downloader over the configured AOI.

    Args:
        out_dir (Path): Destination tile directory.
        max_workers (int): Maximum concurrent worker count.
        bbox_ll (tuple[float, float, float, float]): AOI in lon/lat order.
        tile_limit (int | None): Optional limit on number of downloaded tiles.
        tile_origins (list[tuple[float, float]] | None): Optional explicit tile
            origins overriding the AOI grid order.

    Returns:
        tuple[int, int]: Success/skip count and failure count.

    Examples:
        >>> callable(run_download)
        True
    """

    projected = project_and_snap_bbox(bbox_ll=bbox_ll)
    min_x, min_y, max_x, max_y, gx0, gy0, gx1, gy1 = projected
    xs, ys = build_grid_coordinates(bbox_ll=bbox_ll)
    selected_tile_origins = (
        tile_origins
        if tile_origins is not None
        else build_tile_origins(bbox_ll=bbox_ll, tile_limit=tile_limit)
    )
    LOGGER.info("=== DOP20 WMS 1 km tiler (≈0.20 m/px) ===")
    LOGGER.info("Output dir: %s  Max workers: %s", out_dir.resolve(), max_workers)
    LOGGER.info(
        "AOI 25832: (%.1f,%.1f)–(%.1f,%.1f); snapped=(%s,%s)–(%s,%s); grid: %s × %s = %s tiles; px=%sx%s",
        min_x,
        min_y,
        max_x,
        max_y,
        int(gx0),
        int(gy0),
        int(gx1),
        int(gy1),
        len(xs),
        len(ys),
        len(xs) * len(ys),
        WIDTH,
        HEIGHT,
    )
    if tile_origins is not None:
        LOGGER.info(
            "Explicit tile list enabled: downloading %s requested tile(s)",
            len(selected_tile_origins),
        )
    elif tile_limit is not None:
        LOGGER.info(
            "Tile limit enabled: downloading first %s tile(s)",
            len(selected_tile_origins),
        )
    image_session = make_session()
    metadata_session = make_session()
    start_time = time.time()
    tasks = []
    ok = 0
    bad = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for x0, y0 in selected_tile_origins:
            tasks.append(
                executor.submit(
                    fetch_and_write_tile,
                    image_session,
                    metadata_session,
                    x0,
                    y0,
                    out_dir=out_dir,
                )
            )
        iterator = (
            tqdm(as_completed(tasks), total=len(tasks), desc="Downloading", ncols=100)
            if tqdm
            else as_completed(tasks)
        )
        for future in iterator:
            message = future.result()
            if message.startswith(("DONE", "SKIP", "SKIP_SEASON")):
                ok += 1
                LOGGER.info(message)
            else:
                bad += 1
                LOGGER.warning(message)
    elapsed = timedelta(seconds=int(time.time() - start_time))
    LOGGER.info(
        "=== Finished: %s success/skip, %s failed, elapsed %s ===", ok, bad, elapsed
    )
    return ok, bad


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the downloader script.

    Returns:
        argparse.ArgumentParser: Configured parser.

    Examples:
        >>> isinstance(_build_arg_parser().prog, str)
        True
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        help="Destination directory for downloaded GeoTIFF tiles.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS,
        help="Maximum concurrent download workers.",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Optional limit for smoke-test downloads (for example 1 or 2).",
    )
    parser.add_argument(
        "--tile-origin",
        action="append",
        default=[],
        help="Explicit tile origin in x,y form. Repeat to download specific tiles.",
    )
    parser.add_argument(
        "--log-file",
        default="",
        help="Optional log file path. Defaults to <out-dir>/download.log.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the configured downloader script.

    Args:
        argv (list[str] | None): Optional CLI argument list for tests or
            wrapper scripts.

    Examples:
        >>> callable(main)
        True
    """

    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    log_file = Path(args.log_file) if args.log_file else out_dir / "download.log"
    tile_origins = [parse_tile_origin(spec) for spec in args.tile_origin]
    configure_logging(log_file)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_download(
        out_dir=out_dir,
        max_workers=int(args.max_workers),
        bbox_ll=BBOX_LL,
        tile_limit=args.max_tiles,
        tile_origins=tile_origins or None,
    )


if __name__ == "__main__":
    main()
