import os
import time
from typing import Optional

import cv2
import numpy as np


# ─────────────────────────── CONFIGURACIÓN GENERAL ───────────────────────────
# Debe coincidir con el transmisor tx_final.py
SYMBOL_SIZE = 40
FID_SIZE = 7
QUIET_INNER = 1
BORDER = 2
FQ = FID_SIZE + QUIET_INNER

# Cámara RX
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# Tamaño de la imagen rectificada.
# Antes estaba en 480. Con 800, cada celda de una grilla 40x40 queda de ~20 px.
WARP_SIZE = 800

# Estabilidad temporal:
# 1 = procesa una lectura apenas aparece.
# 2 = exige que la misma grilla se lea igual dos veces seguidas.
# Para la prueba que ya te funcionó, 1 suele ser mejor.
REQUIRED_STABLE = 1

# Debug / consola
DEBUG_VERBOSE = False
PRINT_INVALID_FRAMES = False
PRINT_CRC_ERRORS = True
SAVE_DEBUG_IMAGE = True

# Cabecera, igual que Tx
PREAMBLE = 0xDEAD
PREAMBLE_BITS = 16
NTOTAL_BITS = 16
NFRAME_BITS = 16
NPAYLOAD_BITS = 16
CHECKSUM_BITS = 16
HEADER_BITS = PREAMBLE_BITS + NTOTAL_BITS + NFRAME_BITS + NPAYLOAD_BITS + CHECKSUM_BITS

# Validaciones de protocolo
MAX_TOTAL_FRAMES = 100


def build_reserved_mask(N=SYMBOL_SIZE, FQ=FQ):
    mask = np.zeros((N, N), dtype=bool)
    for r0, c0 in [(0, 0), (0, N - FQ), (N - FQ, 0), (N - FQ, N - FQ)]:
        mask[r0:r0 + FQ, c0:c0 + FQ] = True
    return mask


RESERVED_MASK = build_reserved_mask()
DATA_POSITIONS = [
    (r, c)
    for r in range(SYMBOL_SIZE)
    for c in range(SYMBOL_SIZE)
    if not RESERVED_MASK[r, c]
]
DATA_CELLS = len(DATA_POSITIONS)
MAX_PAYLOAD_BITS_PER_FRAME = (DATA_CELLS - HEADER_BITS) // 2


# ─────────────────────────── CRC-16 (CCITT-FALSE) ────────────────────────────
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
            crc = ((crc << 1) ^ 0x1021) if (crc & 0x8000) else (crc << 1)
        crc &= 0xFFFF
    return crc


# ─────────────────────────── DECODIFICACIÓN ──────────────────────────────────
def bits_to_int(bits: list) -> int:
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def manchester_decode(bits: list) -> Optional[list]:
    """
    Manchester usado por Tx:
      1 -> 10
      0 -> 01

    Si aparece 00 o 11, la trama está corrupta y se descarta.
    Antes el código ignoraba esos pares, lo cual podía desalinear el payload.
    """
    decoded = []

    if len(bits) % 2 != 0:
        return None

    for i in range(0, len(bits), 2):
        a, b = bits[i], bits[i + 1]

        if a == 1 and b == 0:
            decoded.append(1)
        elif a == 0 and b == 1:
            decoded.append(0)
        else:
            return None

    return decoded


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


def parse_frame(all_bits: list):
    """
    Separa la trama en cabecera cruda y payload Manchester.

    Devuelve:
      dict con campos de trama si la estructura es válida.
      None si preámbulo/cabecera/payload son inválidos.
    """
    if len(all_bits) < HEADER_BITS:
        return None

    ptr = 0
    preamble = bits_to_int(all_bits[ptr:ptr + PREAMBLE_BITS])
    ptr += PREAMBLE_BITS

    n_total = bits_to_int(all_bits[ptr:ptr + NTOTAL_BITS])
    ptr += NTOTAL_BITS

    n_frame = bits_to_int(all_bits[ptr:ptr + NFRAME_BITS])
    ptr += NFRAME_BITS

    n_payload = bits_to_int(all_bits[ptr:ptr + NPAYLOAD_BITS])
    ptr += NPAYLOAD_BITS

    crc_rx = bits_to_int(all_bits[ptr:ptr + CHECKSUM_BITS])
    ptr += CHECKSUM_BITS

    # 1) Validar preámbulo
    if preamble != PREAMBLE:
        return None

    # 2) Validar cabecera para descartar casos absurdos:
    #    FRAME #8/7, FRAME #64/775, FRAME #7/32775, etc.
    if n_total <= 0 or n_total > MAX_TOTAL_FRAMES:
        return None

    if n_frame < 0 or n_frame >= n_total:
        return None

    if n_payload <= 0 or n_payload > MAX_PAYLOAD_BITS_PER_FRAME:
        return None

    # 3) Validar que el payload Manchester quepa en las celdas disponibles
    needed_cells = n_payload * 2
    if ptr + needed_cells > len(all_bits):
        return None

    payload_manchester_cells = all_bits[ptr:ptr + needed_cells]
    payload_bits = manchester_decode(payload_manchester_cells)

    if payload_bits is None:
        return None

    if len(payload_bits) != n_payload:
        return None

    # 4) Verificar CRC
    crc_calc = crc16(payload_bits)
    crc_ok = (crc_calc == crc_rx)

    return {
        "n_total": n_total,
        "n_frame": n_frame,
        "n_payload": n_payload,
        "crc_ok": crc_ok,
        "payload": payload_bits,
        "crc_rx": crc_rx,
        "crc_calc": crc_calc,
    }


# ─────────────────────────── LECTURA DE GRILLA ───────────────────────────────
_debug_saved = False


def read_symbol_bits(warp_gray, symbol_size=SYMBOL_SIZE, border=0):
    """
    Lee la grilla rectificada y devuelve los bits de DATA_POSITIONS.

    border=0 porque la homografía actual rectifica usando los fiduciales,
    no el borde blanco externo completo del transmisor.
    """
    global _debug_saved

    H, W = warp_gray.shape
    total_cells = symbol_size
    cell_px = H / total_cells

    # Umbral global adaptativo por Otsu sobre la imagen ya rectificada.
    _, bw = cv2.threshold(
        warp_gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    if SAVE_DEBUG_IMAGE and not _debug_saved:
        os.makedirs("debug_rx", exist_ok=True)

        debug_color = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

        # Marca la primera celda de datos realmente usada.
        first_row, first_col = DATA_POSITIONS[0]
        rr = first_row + border
        cc = first_col + border
        r0 = int(rr * cell_px)
        r1 = int((rr + 1) * cell_px)
        c0 = int(cc * cell_px)
        c1 = int((cc + 1) * cell_px)

        cv2.rectangle(debug_color, (c0, r0), (c1, r1), (0, 0, 255), 2)
        cv2.imwrite(os.path.join("debug_rx", "debug_bw.png"), debug_color)

        _debug_saved = True

    if DEBUG_VERBOSE:
        print(f"Primera posición de datos: {DATA_POSITIONS[0]}")
        print(f"cell_px: {cell_px:.2f}")
        print(f"total_cells: {total_cells}")

    bits = []

    for i, (row, col) in enumerate(DATA_POSITIONS):
        r_cell = row + border
        c_cell = col + border

        r0 = int(r_cell * cell_px)
        r1 = int((r_cell + 1) * cell_px)
        c0 = int(c_cell * cell_px)
        c1 = int((c_cell + 1) * cell_px)

        margin_r = max(1, (r1 - r0) // 5)
        margin_c = max(1, (c1 - c0) // 5)

        patch = bw[
            r0 + margin_r:r1 - margin_r,
            c0 + margin_c:c1 - margin_c
        ]

        mean_val = float(patch.mean()) if patch.size else 0.0

        if DEBUG_VERBOSE and i == 0:
            print(
                f"Celda ({row},{col}) -> r0={r0} r1={r1} "
                f"c0={c0} c1={c1} -> mean={mean_val:.0f}"
            )

        bits.append(0 if mean_val < 128 else 1)

    return bits


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
    """
    Ordena las esquinas aproximadas de los cuatro fiduciales:
      TL, TR, BR, BL
    """
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
    def __init__(self, scale=1, warp_size=WARP_SIZE):
        self.scale = scale
        self.warp_size = warp_size
        self.cap = None
        self.reset()

    # ── Estado ────────────────────────────────────────────────────────────────
    def reset(self):
        self._frame_store = {}
        self._n_total_expected = None
        self._decoded_text = ""

        self._last_symbol_bits = None
        self._stable_count = 0
        self._required_stable = REQUIRED_STABLE
        self._processed_symbol_ids = set()

        self._last_event_msg = ""
        self._first_ok_time = None
        self._complete_time = None
        self._final_reported = False

        print("Decoder reiniciado.")

    # ── Cámara ────────────────────────────────────────────────────────────────
    def open_camera(self, cam_id=0):
        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_MSMF)

        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara id={cam_id}")

        # Configuración sugerida para bajar carga frente a 1920x1080.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        for _ in range(10):
            self.cap.read()

        # Estas propiedades dependen de la cámara/driver.
        # Si el driver las ignora, no pasa nada.
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 113)
        self.cap.set(cv2.CAP_PROP_CONTRAST, 128)
        self.cap.set(cv2.CAP_PROP_GAIN, 34)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_FOCUS, 0)

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
        print(f"Estabilidad requerida: {self._required_stable}")

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

    # ── Reconstrucción de texto ───────────────────────────────────────────────
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

        self._decoded_text = bits_to_text(all_bits)

        if self._decoded_text and self._complete_time is None:
            self._complete_time = time.time()

    # ── Pipeline principal ────────────────────────────────────────────────────
    def process_frame(self, frame):
        debug = frame.copy()

        # Para la etapa actual buscamos en toda la imagen.
        # Más adelante se puede volver a ROI central para mejorar FPS.
        roi = frame
        scaled = cv2.resize(
            roi,
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
        bits = read_symbol_bits(warp_gray, border=0)

        # ── Estabilidad temporal ─────────────────────────────────────────────
        if self._last_symbol_bits is None:
            self._last_symbol_bits = bits.copy()
            self._stable_count = 1
        elif bits == self._last_symbol_bits:
            self._stable_count += 1
        else:
            self._last_symbol_bits = bits.copy()
            self._stable_count = 1

        symbol_just_processed = False

        if self._stable_count >= self._required_stable:
            symbol_id = "".join(map(str, bits))

            if symbol_id not in self._processed_symbol_ids:
                self._processed_symbol_ids.add(symbol_id)

                if DEBUG_VERBOSE:
                    preamble_bits = bits[:16]
                    preamble_val = bits_to_int(preamble_bits)
                    print(f"Preámbulo leído: {hex(preamble_val)} (esperado: {hex(PREAMBLE)})")
                    print(f"Primeros 80 bits: {bits[:80]}")

                parsed = parse_frame(bits)

                if parsed is None:
                    msg = "[FRAME descartado] preámbulo/cabecera inválidos"
                    self._last_event_msg = msg
                    if PRINT_INVALID_FRAMES:
                        print(msg)

                elif not parsed["crc_ok"]:
                    nf = parsed["n_frame"]
                    nt = parsed["n_total"]
                    msg = f"[FRAME #{nf + 1}/{nt} DESCARTADO] CRC error"
                    self._last_event_msg = msg
                    if PRINT_CRC_ERRORS:
                        print(msg)

                else:
                    nf = parsed["n_frame"]
                    nt = parsed["n_total"]

                    if self._first_ok_time is None:
                        self._first_ok_time = time.time()

                    self._n_total_expected = nt
                    self._frame_store[nf] = parsed["payload"]
                    self._try_reconstruct()

                    msg = (
                        f"[FRAME #{nf + 1}/{nt} OK] "
                        f"CRC ✓ | payload={parsed['n_payload']} bits | "
                        f"almacenados={len(self._frame_store)}/{nt} | "
                        f"texto={len(self._decoded_text)} chars"
                    )

                    if self._complete_time is not None and self._first_ok_time is not None:
                        elapsed = self._complete_time - self._first_ok_time
                        msg += f" | tiempo_rx={elapsed:.2f}s"

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

        cv2.putText(
            debug,
            f"Fiduciales OK | estable: {self._stable_count}/{self._required_stable} "
            f"| frames: {len(self._frame_store)}/{n_total}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            est_color,
            2
        )

        if self._last_event_msg:
            color = (0, 255, 0) if just_processed else (180, 255, 180)
            cv2.putText(
                debug,
                self._last_event_msg[:110],
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
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
        """
        Selecciona cuatro fiduciales extremos.
        """
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
        """
        Dibuja grilla visual sobre la imagen rectificada.
        Ahora no suma BORDER porque la lectura real usa border=0.
        """
        total_cells = SYMBOL_SIZE
        cell_px = ws / total_cells

        for i in range(total_cells + 1):
            p = int(i * cell_px)
            cv2.line(img, (p, 0), (p, ws), (0, 180, 0), 1)
            cv2.line(img, (0, p), (ws, p), (0, 180, 0), 1)

        overlay = img.copy()

        for row, col in DATA_POSITIONS:
            r0 = int(row * cell_px)
            c0 = int(col * cell_px)
            r1 = int((row + 1) * cell_px)
            c1 = int((col + 1) * cell_px)
            cv2.rectangle(overlay, (c0, r0), (c1, r1), (255, 100, 0), -1)

        cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)

        return img


# ─────────────────────────── MAIN LOOP ───────────────────────────────────────
if __name__ == "__main__":
    rx = Rx(scale=1, warp_size=WARP_SIZE)
    rx.open_camera(cam_id=0)

    print("Leyendo... 'q' para salir, 'r' para reiniciar decoder.")
    print("Tip: si no detecta cámara, cambia rx.open_camera(cam_id=0) por 1 o 2.")
    print("Tip: si hay muchos CRC error, prueba subir delay en TX o mejorar enfoque/brillo.\n")

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
                print("\n\nTexto completo decodificado:")
                print(text)
                print()
                final_printed = True

    finally:
        print(f"\n\nTexto final decodificado:\n{rx._decoded_text}")
        rx.close_camera()