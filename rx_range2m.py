import os
import sys
import time
import math
from typing import Optional

import cv2
import numpy as np


# ─────────────────────────── CONFIGURACIÓN GENERAL ───────────────────────────
# Modos:
#   AUTO
#   OOK_MANCHESTER
#   ASK4_GRAY
#   CSK_RGB
#
# Uso:
#   python rx.py OOK_MANCHESTER
#   python rx.py ASK4_GRAY
#   python rx.py CSK_RGB
#   python rx.py AUTO
RX_MODULATION = "AUTO"

SYMBOL_SIZE = 40
FID_SIZE = 9
QUIET_INNER = 1
BORDER = 2
FQ = FID_SIZE + QUIET_INNER

CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30

WARP_SIZE = 800
PROCESS_SCALE = 1

# OOK funciona bien con 1.
# ASK4 es más delicado. La robustez la da ASK4_REPEAT, pero dejamos estabilidad en 1
# para no volver demasiado lento el sistema.
REQUIRED_STABLE = 1

DEBUG_VERBOSE = False
PRINT_INVALID_FRAMES = False
PRINT_CRC_ERRORS = False
SAVE_DEBUG_IMAGE = False
PRINT_FPS_EVERY_SECOND = False

# Muestreo robusto del centro de cada celda.
# Mantiene la version estable 40x40, pero reduce ruido de bordes/reflejos.
CELL_CENTER_MARGIN_FRACTION = 0.22
CELL_STATISTIC = "TRIMMED_MEAN"  # opciones: MEAN, MEDIAN, TRIMMED_MEAN
COLOR_STATISTIC = "MEDIAN"      # opciones: MEAN, MEDIAN

ENABLE_BER_EVALUATION = True

ASK4_REPEAT = 3

# Repetición espacial para CSK/RGB. Debe coincidir con tx_final.py.
CSK_REPEAT = 1

EXPECTED_TEXT = (
    "La vision artificial permite interpretar imagenes mediante algoritmos "
    "que detectan patrones, formas y relaciones espaciales. Los sistemas "
    "modernos utilizan transformaciones geometricas, segmentacion y analisis "
    "de contornos para identificar objetos incluso bajo rotacion o perspectiva. "
    "Los fiduciales son referencias visuales usadas para calcular orientacion, "
    "escala y posicion, facilitando la reconstruccion proyectiva y la extraccion "
    "robusta de informacion contenida dentro de una region determinada."
)

VALID_RX_MODULATIONS = {
    "AUTO",
    "OOK_MANCHESTER",
    "ASK4_GRAY",
    "CSK_RGB",
}

# Pilotos de 4 niveles. Deben coincidir con tx_final.py.
PILOT_LEVEL_POSITIONS = {
    0: [
        (11, 11), (28, 28),
        (13, 20), (20, 13),
    ],
    1: [
        (11, 12), (28, 27),
        (14, 20), (20, 14),
    ],
    2: [
        (11, 28), (28, 11),
        (26, 20), (20, 26),
    ],
    3: [
        (11, 29), (28, 10),
        (27, 20), (20, 27),
    ],
}

USE_PILOT_THRESHOLD = True
MIN_BINARY_PILOT_CONTRAST = 20.0

# Cabecera común
PREAMBLE = 0xDEAD
PREAMBLE_BITS = 16
NTOTAL_BITS = 16
NFRAME_BITS = 16
NPAYLOAD_BITS = 16
CHECKSUM_BITS = 16

HEADER_BITS = (
    PREAMBLE_BITS +
    NTOTAL_BITS +
    NFRAME_BITS +
    NPAYLOAD_BITS +
    CHECKSUM_BITS
)

MAX_TOTAL_FRAMES = 100


def get_rx_modulation_from_args(default: str) -> str:
    if len(sys.argv) >= 2:
        mode = sys.argv[1].strip().upper()
    else:
        mode = default

    if mode not in VALID_RX_MODULATIONS:
        raise ValueError(
            f"Modo RX inválido: {mode}. "
            f"Opciones válidas: {sorted(VALID_RX_MODULATIONS)}"
        )

    return mode


def load_expected_text_from_args(default_text: str) -> str:
    """
    Permite usar el mismo archivo .txt transmitido por el TX para calcular BER:
        python rx.py OOK_MANCHESTER mensaje.txt
        python rx.py ASK4_GRAY mensaje.txt

    Si no se pasa archivo, usa EXPECTED_TEXT.
    """
    if len(sys.argv) >= 3:
        path = sys.argv[2]
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    return default_text


# ─────────────────────────── MÁSCARAS ESPACIALES ─────────────────────────────
def all_pilot_positions() -> list[tuple[int, int]]:
    positions = []

    for pts in PILOT_LEVEL_POSITIONS.values():
        positions.extend(pts)

    return positions


def build_reserved_mask(N=SYMBOL_SIZE, FQ=FQ):
    mask = np.zeros((N, N), dtype=bool)

    # Fiduciales + quiet zones
    for r0, c0 in [
        (0, 0),
        (0, N - FQ),
        (N - FQ, 0),
        (N - FQ, N - FQ),
    ]:
        mask[r0:r0 + FQ, c0:c0 + FQ] = True

    # Pilotos
    for r, c in all_pilot_positions():
        if not (0 <= r < N and 0 <= c < N):
            raise ValueError(f"Piloto fuera de la grilla: {(r, c)}")
        mask[r, c] = True

    return mask


RESERVED_MASK = build_reserved_mask()

DATA_POSITIONS = [
    (r, c)
    for r in range(SYMBOL_SIZE)
    for c in range(SYMBOL_SIZE)
    if not RESERVED_MASK[r, c]
]

DATA_CELLS = len(DATA_POSITIONS)
PAYLOAD_CELLS = DATA_CELLS - HEADER_BITS

OOK_MAX_PAYLOAD_BITS_PER_FRAME = PAYLOAD_CELLS // 2
ASK4_MAX_PAYLOAD_BITS_PER_FRAME = (PAYLOAD_CELLS // ASK4_REPEAT) * 2
CSK_MAX_PAYLOAD_BITS_PER_FRAME = (PAYLOAD_CELLS // CSK_REPEAT) * 2

BROAD_MAX_PAYLOAD_BITS_PER_FRAME = max(
    OOK_MAX_PAYLOAD_BITS_PER_FRAME,
    ASK4_MAX_PAYLOAD_BITS_PER_FRAME,
    CSK_MAX_PAYLOAD_BITS_PER_FRAME,
)


# ─────────────────────────── CRC-16 CCITT-FALSE ──────────────────────────────
def crc16(bits: list) -> int:
    crc = 0xFFFF

    for i in range(0, len(bits), 8):
        chunk = bits[i:i + 8]

        if len(chunk) < 8:
            chunk += [0] * (8 - len(chunk))

        byte_val = 0

        for b in chunk:
            byte_val = (byte_val << 1) | int(b)

        crc ^= byte_val << 8

        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc = crc << 1

        crc &= 0xFFFF

    return crc


# ─────────────────────────── DECODIFICACIÓN Y MÉTRICAS ───────────────────────
def bits_to_int(bits: list) -> int:
    val = 0

    for b in bits:
        val = (val << 1) | int(b)

    return val


def text_to_bits(text: str) -> list[int]:
    return [int(b) for char in text for b in format(ord(char), "08b")]


def bits_to_text(bits: list) -> str:
    chars = []

    for i in range(0, len(bits) - 7, 8):
        val = bits_to_int(bits[i:i + 8])

        if val == 0:
            break

        try:
            chars.append(chr(val))
        except ValueError:
            pass

    return "".join(chars)


def compute_ber(expected_bits: list[int], received_bits: list[int]) -> dict:
    max_len = max(len(expected_bits), len(received_bits))

    if max_len == 0:
        return {
            "expected_bits": len(expected_bits),
            "received_bits": len(received_bits),
            "compared_bits": 0,
            "bit_errors": 0,
            "ber": 0.0,
        }

    bit_errors = 0

    for i in range(max_len):
        expected = expected_bits[i] if i < len(expected_bits) else None
        received = received_bits[i] if i < len(received_bits) else None

        if expected != received:
            bit_errors += 1

    return {
        "expected_bits": len(expected_bits),
        "received_bits": len(received_bits),
        "compared_bits": max_len,
        "bit_errors": bit_errors,
        "ber": bit_errors / max_len,
    }


def manchester_decode(bits: list) -> Optional[list]:
    decoded = []

    if len(bits) % 2 != 0:
        return None

    for i in range(0, len(bits), 2):
        a = bits[i]
        b = bits[i + 1]

        if a == 1 and b == 0:
            decoded.append(1)
        elif a == 0 and b == 1:
            decoded.append(0)
        else:
            return None

    return decoded


def parse_common_header(binary_bits: list[int]):
    if len(binary_bits) < HEADER_BITS:
        return None

    ptr = 0

    preamble = bits_to_int(binary_bits[ptr:ptr + PREAMBLE_BITS])
    ptr += PREAMBLE_BITS

    n_total = bits_to_int(binary_bits[ptr:ptr + NTOTAL_BITS])
    ptr += NTOTAL_BITS

    n_frame = bits_to_int(binary_bits[ptr:ptr + NFRAME_BITS])
    ptr += NFRAME_BITS

    n_payload = bits_to_int(binary_bits[ptr:ptr + NPAYLOAD_BITS])
    ptr += NPAYLOAD_BITS

    crc_rx = bits_to_int(binary_bits[ptr:ptr + CHECKSUM_BITS])
    ptr += CHECKSUM_BITS

    if preamble != PREAMBLE:
        return None

    if n_total <= 0 or n_total > MAX_TOTAL_FRAMES:
        return None

    if n_frame < 0 or n_frame >= n_total:
        return None

    if n_payload <= 0 or n_payload > BROAD_MAX_PAYLOAD_BITS_PER_FRAME:
        return None

    return {
        "n_total": n_total,
        "n_frame": n_frame,
        "n_payload": n_payload,
        "crc_rx": crc_rx,
    }


def ask4_decode_from_means(payload_means: list[float], n_bits: int, calibration: dict) -> list[int]:
    """
    Decodifica ASK4 con repetición espacial.
    Cada símbolo 4ASK ocupa ASK4_REPEAT celdas.
    Se usa la mediana de esas celdas para reducir ruido.
    """
    level_means = calibration["level_means"]

    decoded = []

    level_to_bits = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0),
        3: (1, 1),
    }

    for i in range(0, len(payload_means), ASK4_REPEAT):
        group = payload_means[i:i + ASK4_REPEAT]

        if not group:
            break

        value = float(np.median(group))

        nearest_level = min(
            [0, 1, 2, 3],
            key=lambda level: abs(value - level_means[level])
        )

        decoded.extend(list(level_to_bits[nearest_level]))

        if len(decoded) >= n_bits:
            break

    return decoded[:n_bits]




def _chromaticity(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    denom = float(np.sum(vec))

    if denom <= 1e-6:
        return np.zeros_like(vec, dtype=float)

    return vec / denom


def csk_decode_from_colors(payload_colors: list[np.ndarray], n_bits: int, calibration: dict) -> list[int]:
    """
    Decodifica CSK/RGB con pilotos de color.
    Cada símbolo de color transporta 2 bits. La decisión se hace por cercanía
    a los colores piloto capturados por la propia cámara.
    """
    color_means = calibration.get("color_level_means")

    if color_means is None:
        return []

    decoded = []

    level_to_bits = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0),
        3: (1, 1),
    }

    level_chroma = {
        level: _chromaticity(color_means[level])
        for level in [0, 1, 2, 3]
    }

    for i in range(0, len(payload_colors), CSK_REPEAT):
        group = payload_colors[i:i + CSK_REPEAT]

        if not group:
            break

        value = np.median(np.array(group, dtype=float), axis=0)
        value_chroma = _chromaticity(value)

        nearest_level = min(
            [0, 1, 2, 3],
            key=lambda level: float(np.linalg.norm(value_chroma - level_chroma[level]))
        )

        decoded.extend(list(level_to_bits[nearest_level]))

        if len(decoded) >= n_bits:
            break

    return decoded[:n_bits]

def try_decode_payload_ook(binary_bits: list[int], header: dict):
    n_payload = header["n_payload"]

    if n_payload > OOK_MAX_PAYLOAD_BITS_PER_FRAME:
        return None

    needed_cells = n_payload * 2
    start = HEADER_BITS
    end = start + needed_cells

    if end > len(binary_bits):
        return None

    payload_manchester_cells = binary_bits[start:end]
    payload_bits = manchester_decode(payload_manchester_cells)

    if payload_bits is None:
        return None

    if len(payload_bits) != n_payload:
        return None

    crc_calc = crc16(payload_bits)
    crc_ok = crc_calc == header["crc_rx"]

    return {
        **header,
        "modulation": "OOK_MANCHESTER",
        "crc_ok": crc_ok,
        "payload": payload_bits,
        "crc_calc": crc_calc,
    }


def try_decode_payload_ask4(cell_means: list[float], header: dict, calibration: dict):
    n_payload = header["n_payload"]

    if n_payload > ASK4_MAX_PAYLOAD_BITS_PER_FRAME:
        return None

    ask4_symbols = math.ceil(n_payload / 2)
    needed_cells = ask4_symbols * ASK4_REPEAT

    start = HEADER_BITS
    end = start + needed_cells

    if end > len(cell_means):
        return None

    payload_means = cell_means[start:end]
    payload_bits = ask4_decode_from_means(payload_means, n_payload, calibration)

    if len(payload_bits) != n_payload:
        return None

    crc_calc = crc16(payload_bits)
    crc_ok = crc_calc == header["crc_rx"]

    return {
        **header,
        "modulation": "ASK4_GRAY",
        "crc_ok": crc_ok,
        "payload": payload_bits,
        "crc_calc": crc_calc,
    }




def try_decode_payload_csk(color_means: list[np.ndarray], header: dict, calibration: dict):
    n_payload = header["n_payload"]

    if n_payload > CSK_MAX_PAYLOAD_BITS_PER_FRAME:
        return None

    csk_symbols = math.ceil(n_payload / 2)
    needed_cells = csk_symbols * CSK_REPEAT

    start = HEADER_BITS
    end = start + needed_cells

    if end > len(color_means):
        return None

    payload_colors = color_means[start:end]
    payload_bits = csk_decode_from_colors(payload_colors, n_payload, calibration)

    if len(payload_bits) != n_payload:
        return None

    crc_calc = crc16(payload_bits)
    crc_ok = crc_calc == header["crc_rx"]

    return {
        **header,
        "modulation": "CSK_RGB",
        "crc_ok": crc_ok,
        "payload": payload_bits,
        "crc_calc": crc_calc,
    }

def parse_and_decode_frame(cell_means: list[float], color_means: list[np.ndarray], binary_bits: list[int], calibration: dict, rx_mode: str):
    header = parse_common_header(binary_bits)

    if header is None:
        return None

    candidates = []

    if rx_mode in ("AUTO", "OOK_MANCHESTER"):
        decoded_ook = try_decode_payload_ook(binary_bits, header)

        if decoded_ook is not None:
            candidates.append(decoded_ook)

    if rx_mode in ("AUTO", "ASK4_GRAY"):
        decoded_ask4 = try_decode_payload_ask4(cell_means, header, calibration)

        if decoded_ask4 is not None:
            candidates.append(decoded_ask4)

    if rx_mode in ("AUTO", "CSK_RGB"):
        decoded_csk = try_decode_payload_csk(color_means, header, calibration)

        if decoded_csk is not None:
            candidates.append(decoded_csk)

    if not candidates:
        return None

    # En AUTO se acepta el primero que pase CRC.
    for candidate in candidates:
        if candidate["crc_ok"]:
            return candidate

    # Si ninguno pasa, se devuelve un candidato para reportar CRC error.
    return candidates[0]


# ─────────────────────────── LECTURA DE GRILLA CON PILOTOS ───────────────────
_debug_saved = False


def _center_patch(img, row: int, col: int, cell_px: float, border: int = 0):
    r_cell = row + border
    c_cell = col + border

    r0 = int(r_cell * cell_px)
    r1 = int((r_cell + 1) * cell_px)
    c0 = int(c_cell * cell_px)
    c1 = int((c_cell + 1) * cell_px)

    margin_r = max(1, int((r1 - r0) * CELL_CENTER_MARGIN_FRACTION))
    margin_c = max(1, int((c1 - c0) * CELL_CENTER_MARGIN_FRACTION))

    patch = img[
        r0 + margin_r:r1 - margin_r,
        c0 + margin_c:c1 - margin_c
    ]

    return patch


def cell_mean(gray_img, row: int, col: int, cell_px: float, border: int = 0) -> float:
    patch = _center_patch(gray_img, row, col, cell_px, border)

    if patch.size == 0:
        return 0.0

    values = patch.astype(np.float32).ravel()

    if CELL_STATISTIC == "MEDIAN":
        return float(np.median(values))

    if CELL_STATISTIC == "TRIMMED_MEAN" and values.size >= 10:
        lo = np.percentile(values, 10)
        hi = np.percentile(values, 90)
        trimmed = values[(values >= lo) & (values <= hi)]
        if trimmed.size > 0:
            return float(trimmed.mean())

    return float(values.mean())


def cell_color_mean(bgr_img, row: int, col: int, cell_px: float, border: int = 0) -> np.ndarray:
    patch = _center_patch(bgr_img, row, col, cell_px, border)

    if patch.size == 0:
        return np.zeros(3, dtype=float)

    values = patch.reshape(-1, 3).astype(np.float32)

    if COLOR_STATISTIC == "MEDIAN":
        return np.median(values, axis=0).astype(float)

    return values.mean(axis=0).astype(float)


def estimate_calibration_with_pilots(warp_gray, cell_px: float, border: int = 0, warp_bgr=None):
    level_means = {}

    for level, positions in PILOT_LEVEL_POSITIONS.items():
        values = [
            cell_mean(warp_gray, r, c, cell_px, border)
            for r, c in positions
        ]
        level_means[level] = float(np.mean(values)) if values else 0.0

    binary_black = (level_means[0] + level_means[1]) / 2.0
    binary_white = (level_means[2] + level_means[3]) / 2.0

    binary_contrast = binary_white - binary_black

    if USE_PILOT_THRESHOLD and binary_contrast >= MIN_BINARY_PILOT_CONTRAST:
        binary_threshold = (binary_black + binary_white) / 2.0
        mode = "PILOT"
    else:
        otsu_threshold, _ = cv2.threshold(
            warp_gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        binary_threshold = float(otsu_threshold)
        mode = "OTSU_FALLBACK"

    ask4_thresholds = {
        "T01": (level_means[0] + level_means[1]) / 2.0,
        "T12": (level_means[1] + level_means[2]) / 2.0,
        "T23": (level_means[2] + level_means[3]) / 2.0,
    }

    color_level_means = None

    if warp_bgr is not None:
        color_level_means = {}

        for level, positions in PILOT_LEVEL_POSITIONS.items():
            values = [
                cell_color_mean(warp_bgr, r, c, cell_px, border)
                for r, c in positions
            ]
            color_level_means[level] = (
                np.mean(np.array(values, dtype=float), axis=0)
                if values else np.zeros(3, dtype=float)
            )

    return {
        "mode": mode,
        "binary_threshold": binary_threshold,
        "binary_black": binary_black,
        "binary_white": binary_white,
        "binary_contrast": binary_contrast,
        "level_means": level_means,
        "ask4_thresholds": ask4_thresholds,
        "color_level_means": color_level_means,
    }


def read_symbol_cells(warp_gray, warp_bgr=None, symbol_size=SYMBOL_SIZE, border=0):
    global _debug_saved

    H, W = warp_gray.shape
    cell_px = H / symbol_size

    calibration = estimate_calibration_with_pilots(warp_gray, cell_px, border, warp_bgr=warp_bgr)
    threshold = calibration["binary_threshold"]

    cell_means = []
    color_means = []
    binary_bits = []

    for row, col in DATA_POSITIONS:
        mean_val = cell_mean(warp_gray, row, col, cell_px, border)

        cell_means.append(mean_val)
        binary_bits.append(1 if mean_val >= threshold else 0)

        if warp_bgr is not None:
            color_means.append(cell_color_mean(warp_bgr, row, col, cell_px, border))
        else:
            color_means.append(np.zeros(3, dtype=float))

    if SAVE_DEBUG_IMAGE and not _debug_saved:
        os.makedirs("debug_rx", exist_ok=True)

        bw = np.where(warp_gray >= threshold, 255, 0).astype(np.uint8)
        debug_color = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

        # Primera celda de datos en azul
        first_row, first_col = DATA_POSITIONS[0]
        r0 = int(first_row * cell_px)
        r1 = int((first_row + 1) * cell_px)
        c0 = int(first_col * cell_px)
        c1 = int((first_col + 1) * cell_px)
        cv2.rectangle(debug_color, (c0, r0), (c1, r1), (255, 0, 0), 2)

        # Pilotos por color visual
        colors = {
            0: (0, 0, 255),
            1: (0, 140, 255),
            2: (0, 255, 255),
            3: (0, 255, 0),
        }

        for level, positions in PILOT_LEVEL_POSITIONS.items():
            color = colors[level]

            for row, col in positions:
                r0 = int(row * cell_px)
                r1 = int((row + 1) * cell_px)
                c0 = int(col * cell_px)
                c1 = int((col + 1) * cell_px)
                cv2.rectangle(debug_color, (c0, r0), (c1, r1), color, 2)

        cv2.imwrite(os.path.join("debug_rx", "debug_bw_pilots_4levels.png"), debug_color)
        _debug_saved = True

    if DEBUG_VERBOSE:
        lm = calibration["level_means"]
        print(
            f"Calibración {calibration['mode']} | "
            f"thr={calibration['binary_threshold']:.1f} | "
            f"L0={lm[0]:.1f} L1={lm[1]:.1f} L2={lm[2]:.1f} L3={lm[3]:.1f}"
        )

    return cell_means, color_means, binary_bits, calibration


# ─────────────────────────── DETECTOR DE FIDUCIALES ──────────────────────────
def is_quad(cnt, min_area=8):
    if cv2.contourArea(cnt) < min_area:
        return False

    eps = 0.03 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True)

    if len(approx) != 4:
        return False

    return cv2.isContourConvex(approx)


def detect_fiducials(binary, contours, hierarchy):
    fiducials = []

    if hierarchy is None:
        return fiducials

    hier = hierarchy[0]

    for i, cnt in enumerate(contours):
        if not is_quad(cnt):
            continue

        child_idx = hier[i][2]

        if child_idx == -1 or not is_quad(contours[child_idx]):
            continue

        gc_idx = hier[child_idx][2]

        if gc_idx == -1 or not is_quad(contours[gc_idx]):
            continue

        rect = cv2.minAreaRect(cnt)
        w, h = rect[1]

        if w == 0 or h == 0:
            continue

        aspect = w / h

        if not (0.4 < aspect < 2.5):
            continue

        area_rect = w * h

        if area_rect <= 0:
            continue

        if cv2.contourArea(cnt) / area_rect < 0.5:
            continue

        box = np.int32(cv2.boxPoints(rect))
        cx = float(np.mean(box[:, 0]))
        cy = float(np.mean(box[:, 1]))

        fiducials.append({
            "center": np.array([cx, cy], dtype=np.float32),
            "box": box
        })

    return fiducials


def order_corners(fiducials):
    centers = np.array([f["center"] for f in fiducials])

    s = centers.sum(axis=1)
    d = np.diff(centers, axis=1).flatten()

    tl_f = fiducials[np.argmin(s)]
    br_f = fiducials[np.argmax(s)]
    tr_f = fiducials[np.argmin(d)]
    bl_f = fiducials[np.argmax(d)]

    tl = tl_f["box"][np.argmin(tl_f["box"].sum(axis=1))].astype(float)
    tr = tr_f["box"][np.argmin(np.diff(tr_f["box"], axis=1).flatten())].astype(float)
    br = br_f["box"][np.argmax(br_f["box"].sum(axis=1))].astype(float)
    bl = bl_f["box"][np.argmax(np.diff(bl_f["box"], axis=1).flatten())].astype(float)

    return tl, tr, br, bl


# ─────────────────────────── CLASE RECEPTOR ──────────────────────────────────
class Rx:
    def __init__(self, scale=PROCESS_SCALE, warp_size=WARP_SIZE, modulation="AUTO"):
        self.scale = scale
        self.warp_size = warp_size
        self.modulation = modulation
        self.cap = None
        self.reset(announce=False)

    def reset(self, announce=True):
        self._frame_store = {}
        self._n_total_expected = None
        self._decoded_text = ""
        self._decoded_bits = []

        self._last_symbol_signature = None
        self._stable_count = 0
        self._required_stable = REQUIRED_STABLE
        self._processed_frame_keys = set()
        self._invalid_header_count = 0
        self._crc_error_counts = {
            "OOK_MANCHESTER": 0,
            "ASK4_GRAY": 0,
            "CSK_RGB": 0,
        }

        self._last_event_msg = ""
        self._last_calibration = None
        self._last_detected_modulation = None

        self._reset_time = time.time()
        self._first_ok_time = None
        self._complete_time = None

        self._ber_metrics = None
        self._text_match = None

        if announce:
            print("Decoder reiniciado.")
            print("Cronómetro RX iniciado desde reset.")

    # ── Cámara ────────────────────────────────────────────────────────────────
    def open_camera(self, cam_id=0):
        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_MSMF)

        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara id={cam_id}")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        for _ in range(10):
            self.cap.read()

        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 113)
        self.cap.set(cv2.CAP_PROP_CONTRAST, 128)
        self.cap.set(cv2.CAP_PROP_GAIN, 34)

        # Para 2 m conviene dejar que la cámara enfoque la pantalla real
        # al iniciar; luego se congela para que no varíe durante la transmisión.
        try:
            if ENABLE_AUTOFOCUS_STARTUP:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                for _ in range(AUTOFOCUS_WARMUP_FRAMES):
                    self.cap.read()
                if FREEZE_FOCUS_AFTER_STARTUP:
                    self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            else:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        except Exception:
            pass

        try:
            self.cap.set(cv2.CAP_PROP_SHARPNESS, 255)
        except Exception:
            pass

        # Congela balance de blancos si la cámara lo soporta.
        # Esto ayuda especialmente a CSK_RGB, porque evita que la cámara cambie
        # la mezcla de colores durante la transmisión.
        try:
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)
        except Exception:
            pass

        for _ in range(5):
            self.cap.read()

        print(f"FPS:        {self.cap.get(cv2.CAP_PROP_FPS)}")
        print(
            f"Resolución: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
            f"{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}"
        )
        print(f"Exposure:   {self.cap.get(cv2.CAP_PROP_EXPOSURE)}")
        print(f"Backend:    {self.cap.getBackendName()}")
        print(f"Warp size:  {self.warp_size}x{self.warp_size}")
        print(f"Scale:      {self.scale}")
        print(f"Modo RX:    {self.modulation}")
        print(f"Estabilidad requerida: {self._required_stable}")

        if self.modulation == "OOK_MANCHESTER":
            print("Pilotos:    binarios por referencia negro/blanco")
        elif self.modulation == "ASK4_GRAY":
            print("Pilotos:    4 niveles × 4 pilotos = 16 pilotos")
            print(f"ASK4_REPEAT: {ASK4_REPEAT}")
        elif self.modulation == "CSK_RGB":
            print("Pilotos:    4 colores calibrados por cámara")
            print(f"CSK_REPEAT: {CSK_REPEAT}")
        else:
            print("Pilotos:    compatibles con OOK, ASK4 y CSK/RGB")
            print(f"ASK4_REPEAT: {ASK4_REPEAT}")
            print(f"CSK_REPEAT: {CSK_REPEAT}")

        if ENABLE_BER_EVALUATION:
            print("BER:        habilitado contra texto de referencia local")
            print(f"Referencia: {len(EXPECTED_TEXT)} caracteres")

        cv2.namedWindow("Camara", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Símbolo", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camara", 960, 540)
        cv2.resizeWindow("Símbolo", 500, 500)

    def read_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def close_camera(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    # ── Métricas ──────────────────────────────────────────────────────────────
    def _time_metrics(self):
        if self._complete_time is None:
            return None, None

        tiempo_total = self._complete_time - self._reset_time

        if self._first_ok_time is not None:
            tiempo_desde_primer_ok = self._complete_time - self._first_ok_time
        else:
            tiempo_desde_primer_ok = None

        return tiempo_total, tiempo_desde_primer_ok

    def _update_ber_metrics(self):
        if not ENABLE_BER_EVALUATION:
            return

        expected_bits = text_to_bits(EXPECTED_TEXT)
        received_bits = self._decoded_bits[:len(expected_bits)]

        self._ber_metrics = compute_ber(expected_bits, received_bits)
        self._text_match = self._decoded_text == EXPECTED_TEXT

    def _print_final_metrics(self):
        tiempo_total, tiempo_desde_primer_ok = self._time_metrics()

        if self._last_detected_modulation is not None:
            print(f"\nModulación detectada/usada: {self._last_detected_modulation}")

        if tiempo_total is not None:
            print(f"Tiempo total desde reset RX: {tiempo_total:.2f} s")

        if tiempo_desde_primer_ok is not None:
            print(f"Tiempo desde primer FRAME OK: {tiempo_desde_primer_ok:.2f} s")

        if self._n_total_expected is not None:
            print(f"Tramas válidas recibidas: {len(self._frame_store)}/{self._n_total_expected}")

        total_crc_errors = sum(self._crc_error_counts.values())
        if self._invalid_header_count or total_crc_errors:
            print("\nTramas descartadas:")
            print(f"  Cabecera/preámbulo inválido: {self._invalid_header_count}")
            for mod_name, count in self._crc_error_counts.items():
                if count:
                    print(f"  CRC error {mod_name}: {count}")

        if self._last_calibration is not None:
            c = self._last_calibration
            lm = c["level_means"]
            th = c["ask4_thresholds"]

            # Elegimos qué mostrar según el demodulador realmente usado.
            active_mod = self._last_detected_modulation or self.modulation

            if active_mod == "OOK_MANCHESTER":
                print("\nCalibración binaria por pilotos:")
                print(f"  Modo umbral:          {c['mode']}")
                print(f"  Referencia negra:     {c['binary_black']:.2f}")
                print(f"  Referencia blanca:    {c['binary_white']:.2f}")
                print(f"  Contraste binario:    {c['binary_contrast']:.2f}")
                print(f"  Umbral usado:         {c['binary_threshold']:.2f}")

            elif active_mod == "ASK4_GRAY":
                print("\nCalibración ASK4 por pilotos:")
                print(f"  Modo umbral binario: {c['mode']}")
                print(f"  Nivel 0 media:       {lm[0]:.2f}")
                print(f"  Nivel 1 media:       {lm[1]:.2f}")
                print(f"  Nivel 2 media:       {lm[2]:.2f}")
                print(f"  Nivel 3 media:       {lm[3]:.2f}")
                print(f"  Contraste binario:   {c['binary_contrast']:.2f}")
                print(f"  Umbral binario:      {c['binary_threshold']:.2f}")
                print(
                    f"  Umbrales ASK4:       "
                    f"T01={th['T01']:.2f}, "
                    f"T12={th['T12']:.2f}, "
                    f"T23={th['T23']:.2f}"
                )

            elif active_mod == "CSK_RGB":
                print("\nCalibración CSK/RGB por pilotos de color:")
                cm = c.get("color_level_means")
                if cm is not None:
                    for level, name in [(0, "rojo"), (1, "verde"), (2, "azul"), (3, "amarillo")]:
                        b, g, r = cm[level]
                        print(f"  Nivel {level} ({name}): B={b:.1f}, G={g:.1f}, R={r:.1f}")
                print(f"  Umbral binario cabecera: {c['binary_threshold']:.2f}")

            else:
                print("\nCalibración por pilotos:")
                print(f"  Modo umbral binario: {c['mode']}")
                print(f"  Contraste binario:   {c['binary_contrast']:.2f}")
                print(f"  Umbral binario:      {c['binary_threshold']:.2f}")

        if ENABLE_BER_EVALUATION and self._ber_metrics is not None:
            m = self._ber_metrics

            print("\nMétricas BER:")
            print(f"  Caracteres esperados: {len(EXPECTED_TEXT)}")
            print(f"  Caracteres recibidos: {len(self._decoded_text)}")
            print(f"  Bits esperados:       {m['expected_bits']}")
            print(f"  Bits recibidos:       {m['received_bits']}")
            print(f"  Bits comparados:      {m['compared_bits']}")
            print(f"  Errores de bit:       {m['bit_errors']}")
            print(f"  BER:                  {m['ber']:.2e}")
            print(f"  Texto exacto:          {'SI' if self._text_match else 'NO'}")

        if tiempo_total is None:
            print("\nNo se completó la decodificación del texto.")

    # ── Reconstrucción ────────────────────────────────────────────────────────
    def _try_reconstruct(self):
        if self._n_total_expected is None:
            return

        if len(self._frame_store) < self._n_total_expected:
            return

        all_bits = []

        for idx in range(self._n_total_expected):
            if idx not in self._frame_store:
                return
            all_bits.extend(self._frame_store[idx])

        self._decoded_bits = all_bits
        self._decoded_text = bits_to_text(all_bits)

        if self._decoded_text and self._complete_time is None:
            self._complete_time = time.time()
            self._update_ber_metrics()

    # ── Pipeline principal ────────────────────────────────────────────────────
    def process_frame(self, frame):
        debug = frame.copy()

        scaled = cv2.resize(
            frame,
            None,
            fx=self.scale,
            fy=self.scale,
            interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)

        bw = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            5
        )

        kernel = np.ones((3, 3), np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, hierarchy = cv2.findContours(
            bw,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if hierarchy is None:
            self._draw_hud(debug, fid_count=0)
            return debug, None, self._decoded_text

        fiducials = detect_fiducials(bw, contours, hierarchy)
        offset = np.array([0, 0], dtype=np.float32)

        for f in fiducials:
            box_orig = (f["box"] / self.scale + offset).astype(np.int32)
            cv2.drawContours(debug, [box_orig], 0, (0, 255, 0), 2)

            c = ((f["center"] / self.scale) + offset).astype(int)
            cv2.circle(debug, tuple(c), 5, (0, 0, 255), -1)

        if len(fiducials) < 4:
            self._draw_hud(debug, fid_count=len(fiducials))
            return debug, None, self._decoded_text

        sel = fiducials[:4] if len(fiducials) == 4 else self._best_four(fiducials)
        tl, tr, br, bl = order_corners(sel)

        pts_orig = (np.array([tl, tr, br, bl]) / self.scale + offset).astype(np.float32)

        cv2.polylines(debug, [pts_orig.astype(np.int32)], True, (255, 0, 0), 2)

        for p, lbl in zip(pts_orig.astype(int), ["TL", "TR", "BR", "BL"]):
            cv2.circle(debug, tuple(p), 6, (255, 255, 0), -1)
            cv2.putText(
                debug,
                lbl,
                (p[0] + 5, p[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

        ws = self.warp_size

        dst = np.array(
            [[0, 0], [ws, 0], [ws, ws], [0, ws]],
            dtype=np.float32
        )

        M = cv2.getPerspectiveTransform(pts_orig, dst)
        warp = cv2.warpPerspective(frame, M, (ws, ws))

        warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)

        cell_means, color_means, binary_bits, calibration = read_symbol_cells(warp_gray, warp_bgr=warp, border=0)
        self._last_calibration = calibration

        # Firma simple para estabilidad.
        signature = tuple(int(v // 8) for v in cell_means[:200])

        if self._last_symbol_signature is None:
            self._last_symbol_signature = signature
            self._stable_count = 1
        elif signature == self._last_symbol_signature:
            self._stable_count += 1
        else:
            self._last_symbol_signature = signature
            self._stable_count = 1

        symbol_just_processed = False

        if self._stable_count >= self._required_stable:
            parsed = parse_and_decode_frame(
                cell_means,
                color_means,
                binary_bits,
                calibration,
                self.modulation
            )

            if parsed is None:
                self._invalid_header_count += 1
                msg = "[FRAME descartado] preámbulo/cabecera inválidos"
                self._last_event_msg = msg

                if PRINT_INVALID_FRAMES:
                    print(msg)

            elif not parsed["crc_ok"]:
                nf = parsed["n_frame"]
                nt = parsed["n_total"]
                mod = parsed["modulation"]
                self._crc_error_counts[mod] = self._crc_error_counts.get(mod, 0) + 1

                msg = f"[{mod}] [FRAME #{nf + 1}/{nt} DESCARTADO] CRC error"
                self._last_event_msg = msg

                if PRINT_CRC_ERRORS:
                    print(msg)

            else:
                nf = parsed["n_frame"]
                nt = parsed["n_total"]
                mod = parsed["modulation"]

                frame_key = (
                    mod,
                    nf,
                    nt,
                    parsed["n_payload"],
                    parsed["crc_rx"],
                )

                if frame_key not in self._processed_frame_keys:
                    self._processed_frame_keys.add(frame_key)

                    if self._first_ok_time is None:
                        self._first_ok_time = time.time()

                    self._last_detected_modulation = mod
                    self._n_total_expected = nt
                    self._frame_store[nf] = parsed["payload"]

                    self._try_reconstruct()

                    msg = (
                        f"[{mod}] [FRAME #{nf + 1}/{nt} OK] "
                        f"CRC ✓ | payload={parsed['n_payload']} bits | "
                        f"almacenados={len(self._frame_store)}/{nt} | "
                        f"texto={len(self._decoded_text)} chars"
                    )

                    if self._complete_time is not None:
                        tiempo_total, tiempo_desde_primer_ok = self._time_metrics()

                        if tiempo_total is not None:
                            msg += f" | tiempo_total_desde_reset={tiempo_total:.2f}s"

                        if tiempo_desde_primer_ok is not None:
                            msg += f" | tiempo_desde_primer_frame_ok={tiempo_desde_primer_ok:.2f}s"

                        if self._ber_metrics is not None:
                            msg += (
                                f" | BER={self._ber_metrics['ber']:.2e} "
                                f"| errores_bit={self._ber_metrics['bit_errors']}"
                            )

                    print(msg)

                    self._last_event_msg = msg
                    symbol_just_processed = True

        self._draw_hud(debug, fid_count=4, just_processed=symbol_just_processed)
        warp_vis = self._draw_grid(warp.copy(), ws)

        return debug, warp_vis, self._decoded_text

    # ── HUD ───────────────────────────────────────────────────────────────────
    def _draw_hud(self, debug, fid_count=0, just_processed=False):
        if fid_count < 4:
            cv2.putText(
                debug,
                f"Fiduciales: {fid_count}/4",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 80, 255),
                2
            )
            return

        est_color = (
            (0, 255, 80)
            if self._stable_count >= self._required_stable
            else (0, 180, 255)
        )

        n_total = self._n_total_expected or "?"
        mod = self._last_detected_modulation or self.modulation

        cv2.putText(
            debug,
            f"Modo={mod} | Fiduciales OK | estable: {self._stable_count}/{self._required_stable} "
            f"| frames: {len(self._frame_store)}/{n_total}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            est_color,
            2
        )

        if self._last_event_msg:
            color = (0, 255, 0) if just_processed else (180, 255, 180)

            cv2.putText(
                debug,
                self._last_event_msg[:120],
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2
            )

        if self._last_calibration is not None:
            c = self._last_calibration
            lm = c["level_means"]

            cv2.putText(
                debug,
                f"Cal={c['mode']} thr={c['binary_threshold']:.0f} "
                f"L0={lm[0]:.0f} L1={lm[1]:.0f} L2={lm[2]:.0f} L3={lm[3]:.0f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 0),
                2
            )

        if self._ber_metrics is not None:
            cv2.putText(
                debug,
                f"BER={self._ber_metrics['ber']:.2e} | errores={self._ber_metrics['bit_errors']}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

        if self._decoded_text:
            cv2.putText(
                debug,
                self._decoded_text[-90:],
                (10, debug.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _best_four(self, fiducials):
        centers = np.array([f["center"] for f in fiducials])

        s = centers.sum(axis=1)
        d = np.diff(centers, axis=1).flatten()

        candidate_indices = [
            int(np.argmin(s)),
            int(np.argmin(d)),
            int(np.argmax(s)),
            int(np.argmax(d)),
        ]

        unique = []

        for idx in candidate_indices:
            if idx not in unique:
                unique.append(idx)

        if len(unique) < 4:
            return fiducials[:4]

        return [fiducials[i] for i in unique[:4]]

    def _draw_grid(self, img, ws):
        total_cells = SYMBOL_SIZE
        cell_px = ws / total_cells

        for i in range(total_cells + 1):
            p = int(i * cell_px)
            cv2.line(img, (p, 0), (p, ws), (0, 180, 0), 1)
            cv2.line(img, (0, p), (ws, p), (0, 180, 0), 1)

        overlay = img.copy()

        # Datos
        for row, col in DATA_POSITIONS:
            r0 = int(row * cell_px)
            c0 = int(col * cell_px)
            r1 = int((row + 1) * cell_px)
            c1 = int((col + 1) * cell_px)
            cv2.rectangle(overlay, (c0, r0), (c1, r1), (255, 100, 0), -1)

        # Pilotos
        colors = {
            0: (0, 0, 255),
            1: (0, 140, 255),
            2: (0, 255, 255),
            3: (0, 255, 0),
        }

        for level, positions in PILOT_LEVEL_POSITIONS.items():
            color = colors[level]

            for row, col in positions:
                r0 = int(row * cell_px)
                c0 = int(col * cell_px)
                r1 = int((row + 1) * cell_px)
                c1 = int((col + 1) * cell_px)
                cv2.rectangle(overlay, (c0, r0), (c1, r1), color, -1)

        cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)

        return img


# ─────────────────────────── MAIN LOOP ───────────────────────────────────────
if __name__ == "__main__":
    rx_mode = get_rx_modulation_from_args(RX_MODULATION)
    EXPECTED_TEXT = load_expected_text_from_args(EXPECTED_TEXT)

    rx = Rx(scale=PROCESS_SCALE, warp_size=WARP_SIZE, modulation=rx_mode)
    rx.open_camera(cam_id=0)

    rx.reset()

    print("=" * 70)
    print("RX - MODEM OPTICO")
    print(f"Demodulador activo: {rx_mode}")
    print("Comandos: 'q' salir | 'r' reiniciar medición")
    print("Para pruebas formales usa modo forzado:")
    print("  python rx.py OOK_MANCHESTER")
    print("  python rx.py ASK4_GRAY")
    print("  python rx.py CSK_RGB")
    print("=" * 70)
    print("")

    last_time = time.time()
    fps_count = 0
    final_printed = False

    try:
        while True:
            frame = rx.read_frame()

            if frame is None:
                break

            fps_count += 1

            if time.time() - last_time >= 1:
                if PRINT_FPS_EVERY_SECOND:
                    print(f"FPS reales: {fps_count}")
                fps_count = 0
                last_time = time.time()

            debug, warp_vis, text = rx.process_frame(frame)

            cv2.imshow("Camara", debug)

            if warp_vis is not None:
                cv2.imshow("Símbolo", warp_vis)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                rx.reset()
                final_printed = False

            if text and not final_printed:
                print("\n" + "=" * 70)
                print("DECODIFICACIÓN COMPLETA")
                print("=" * 70)
                print("Texto completo decodificado:")
                print(text)

                rx._print_final_metrics()

                print("=" * 70)
                print("Puedes presionar 'q' para salir o 'r' para reiniciar.\n")
                final_printed = True

    finally:
        if not final_printed:
            print(f"\n\nTexto final decodificado:\n{rx._decoded_text}")
            rx._print_final_metrics()

        rx.close_camera()
