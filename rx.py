import numpy as np
import cv2
import time


# ─────────────────────────── PARÁMETROS (deben coincidir con Tx) ──────────────
SYMBOL_SIZE   = 40
FID_SIZE      = 7
QUIET_INNER   = 1
BORDER        = 2
FQ            = FID_SIZE + QUIET_INNER

# Cabecera (igual que Tx)
PREAMBLE      = 0xDEAD
PREAMBLE_BITS = 16
NTOTAL_BITS   = 16
NFRAME_BITS   = 16
NPAYLOAD_BITS = 16
CHECKSUM_BITS = 16
HEADER_BITS   = PREAMBLE_BITS + NTOTAL_BITS + NFRAME_BITS + NPAYLOAD_BITS + CHECKSUM_BITS  # 80

def build_reserved_mask(N=SYMBOL_SIZE, FQ=FQ):
    mask = np.zeros((N, N), dtype=bool)
    for r0, c0 in [(0,0),(0,N-FQ),(N-FQ,0),(N-FQ,N-FQ)]:
        mask[r0:r0+FQ, c0:c0+FQ] = True
    return mask

RESERVED_MASK  = build_reserved_mask()
DATA_POSITIONS = [(r, c) for r in range(SYMBOL_SIZE)
                          for c in range(SYMBOL_SIZE)
                          if not RESERVED_MASK[r, c]]
DATA_CELLS     = len(DATA_POSITIONS)


# ─────────────────────────── CRC-16 (CCITT-FALSE) ────────────────────────────
def crc16(bits: list) -> int:
    crc = 0xFFFF
    for i in range(0, len(bits), 8):
        chunk = bits[i:i+8]
        if len(chunk) < 8:
            chunk += [0] * (8 - len(chunk))
        byte_val = 0
        for b in chunk:
            byte_val = (byte_val << 1) | b
        crc ^= byte_val << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if (crc & 0x8000) else (crc << 1)
        crc &= 0xFFFF
    return crc


# ─────────────────────────── DECODIFICACIÓN ───────────────────────────────────
def bits_to_int(bits: list) -> int:
    val = 0
    for b in bits:
        val = (val << 1) | b
    return val

def manchester_decode(bits: list) -> list:
    decoded = []
    for i in range(0, len(bits) - 1, 2):
        a, b = bits[i], bits[i+1]
        if   a == 1 and b == 0: decoded.append(1)
        elif a == 0 and b == 1: decoded.append(0)
        # par inválido → ruido, se descarta
    return decoded

def bits_to_text(bits: list) -> str:
    chars = []
    for i in range(0, len(bits) - 7, 8):
        val = bits_to_int(bits[i:i+8])
        if val == 0:
            break
        try:
            chars.append(chr(val))
        except ValueError:
            pass
    return ''.join(chars)

def parse_frame(all_bits: list):
    """
    Separa la trama en cabecera cruda y payload Manchester,
    exactamente como la construye Tx.

    Devuelve dict con los campos o None si el preámbulo no coincide.
    """
    if len(all_bits) < HEADER_BITS:
        return None

    # ── Cabecera cruda (80 bits, 1 celda = 1 bit) ────────────────────────────
    ptr = 0
    preamble  = bits_to_int(all_bits[ptr: ptr + PREAMBLE_BITS]);  ptr += PREAMBLE_BITS
    n_total   = bits_to_int(all_bits[ptr: ptr + NTOTAL_BITS]);    ptr += NTOTAL_BITS
    n_frame   = bits_to_int(all_bits[ptr: ptr + NFRAME_BITS]);    ptr += NFRAME_BITS
    n_payload = bits_to_int(all_bits[ptr: ptr + NPAYLOAD_BITS]);  ptr += NPAYLOAD_BITS
    crc_rx    = bits_to_int(all_bits[ptr: ptr + CHECKSUM_BITS]);  ptr += CHECKSUM_BITS

    if preamble != PREAMBLE:
        return None   # no es un símbolo válido

    # ── Payload Manchester (n_payload bits reales → n_payload*2 celdas) ───────
    payload_manchester_cells = all_bits[ptr: ptr + n_payload * 2]
    payload_bits = manchester_decode(payload_manchester_cells)

    # ── Verificar CRC ─────────────────────────────────────────────────────────
    crc_calc = crc16(payload_bits)
    crc_ok   = (crc_calc == crc_rx)

    return {
        "n_total":   n_total,
        "n_frame":   n_frame,
        "n_payload": n_payload,
        "crc_ok":    crc_ok,
        "payload":   payload_bits,
    }

_debug_saved = False
def read_symbol_bits(warp_gray, symbol_size=SYMBOL_SIZE, border=BORDER):
    global _debug_saved
    H, W        = warp_gray.shape
    total_cells = symbol_size  
    cell_px     = H / total_cells

    _, bw = cv2.threshold(warp_gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if not _debug_saved:
        import os
        os.makedirs("debug_rx", exist_ok=True)

        # Dibujar punto en la primera celda de datos
        debug_color = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(debug_color, (109, 21), (119, 32), (0,0,255), 2)

        cv2.imwrite("debug/debug_bw.png", debug_color)

        _debug_saved = True
    # en read_symbol_bits, antes del loop
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
        patch = bw[r0 + margin_r : r1 - margin_r,
                   c0 + margin_c : c1 - margin_c]
        if i == 0:
            print(f"Celda (0,8) → r0={r0} r1={r1} c0={c0} c1={c1} → mean={patch.mean():.0f}")
        bits.append(0 if patch.mean() < 128 else 1)
    return bits


# ─────────────────────────── DETECTOR DE FIDUCIALES ──────────────────────────
def is_quad(cnt, min_area=8):
    if cv2.contourArea(cnt) < min_area:
        return False
    eps    = 0.03 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True)
    if len(approx) != 4:
        return False
    return cv2.isContourConvex(approx)

def detect_fiducials(binary, contours, hierarchy):
    fiducials = []
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
        if not (0.4 < w/h < 2.5):
            continue
        if cv2.contourArea(cnt) / (w * h) < 0.5:
            continue
        box = np.int32(cv2.boxPoints(rect))
        cx  = float(np.mean(box[:, 0]))
        cy  = float(np.mean(box[:, 1]))
        fiducials.append({"center": np.array([cx, cy]), "box": box})
    return fiducials

def order_corners(fiducials):
    centers = np.array([f["center"] for f in fiducials])
    s = centers.sum(axis=1)
    d = np.diff(centers, axis=1).flatten()
    tl = fiducials[np.argmin(s)]["box"]
    br = fiducials[np.argmax(s)]["box"]
    tr = fiducials[np.argmin(d)]["box"]
    bl = fiducials[np.argmax(d)]["box"]
    return (
        tl[np.argmin(tl.sum(axis=1))].astype(float),
        tr[np.argmin(np.diff(tr, axis=1).flatten())].astype(float),
        br[np.argmax(br.sum(axis=1))].astype(float),
        bl[np.argmax(np.diff(bl, axis=1).flatten())].astype(float),
    )


# ─────────────────────────── CLASE RECEPTOR ──────────────────────────────────
class Rx:
    def __init__(self, scale=1, warp_size=480):
        self.scale     = scale
        self.warp_size = warp_size
        self.cap       = None
        self.reset()

    # ── Estado ────────────────────────────────────────────────────────────────
    def reset(self):
        # frame_store: dict { n_frame → payload_bits }
        self._frame_store           = {}
        self._n_total_expected      = None
        self._decoded_text          = ""
        self._last_symbol_bits      = None
        self._stable_count          = 0
        self._required_stable       = 1
        self._processed_symbol_ids  = set()
        self._last_event_msg        = ""
        print("Decoder reiniciado.")

    # ── Cámara ────────────────────────────────────────────────────────────────
    def open_camera(self, cam_id=0):
        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_MSMF)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara id={cam_id}")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS,          30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        for _ in range(10): self.cap.read()
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_EXPOSURE,      -6)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS,   113)
        self.cap.set(cv2.CAP_PROP_CONTRAST,     128)
        self.cap.set(cv2.CAP_PROP_GAIN,          34)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS,      0)
        self.cap.set(cv2.CAP_PROP_FOCUS,          0)
        for _ in range(5): self.cap.read()

        print(f"FPS:        {self.cap.get(cv2.CAP_PROP_FPS)}")
        print(f"Resolución: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
              f"{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")
        print(f"Exposure:   {self.cap.get(cv2.CAP_PROP_EXPOSURE)}")
        print(f"Backend:    {self.cap.getBackendName()}")

        cv2.namedWindow("Camara",  cv2.WINDOW_NORMAL)
        cv2.namedWindow("Símbolo", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camara",  960, 540)
        cv2.resizeWindow("Símbolo", 400, 400)

    def read_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def close_camera(self):
        if self.cap: self.cap.release()
        cv2.destroyAllWindows()

    # ── Reconstrucción de texto ───────────────────────────────────────────────
    def _try_reconstruct(self):
        """Reconstruye el texto cuando hay suficientes frames almacenados."""
        if self._n_total_expected is None:
            return
        # Verificar que tenemos todos los frames
        if len(self._frame_store) < self._n_total_expected:
            return
        # Ordenar por n_frame y concatenar payloads
        all_bits = []
        for idx in range(self._n_total_expected):
            if idx not in self._frame_store:
                return   # falta alguno
            all_bits.extend(self._frame_store[idx])
        self._decoded_text = bits_to_text(all_bits)

    # ── Pipeline principal ────────────────────────────────────────────────────
    def process_frame(self, frame):
        debug = frame.copy()
        h, w  = frame.shape[:2]
        roi    = frame[h//4:3*h//4, w//4:3*w//4]
        scaled = cv2.resize(roi, None, fx=self.scale, fy=self.scale,
                            interpolation=cv2.INTER_CUBIC)

        gray  = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        gray  = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        gray  = clahe.apply(gray)
        bw    = cv2.adaptiveThreshold(gray, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 21, 5)
        kernel = np.ones((3, 3), np.uint8)
        bw     = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, hierarchy = cv2.findContours(
            bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if hierarchy is None:
            self._draw_hud(debug, fid_count=0)
            return debug, None, self._decoded_text

        fiducials = detect_fiducials(bw, contours, hierarchy)

        offset = np.array([w//4, h//4], dtype=np.float32)
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
        for p, lbl in zip(pts_orig.astype(int), ["TL","TR","BR","BL"]):
            cv2.circle(debug, tuple(p), 6, (255, 255, 0), -1)
            cv2.putText(debug, lbl, (p[0]+5, p[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

        ws  = self.warp_size
        dst = np.array([[0,0],[ws,0],[ws,ws],[0,ws]], dtype=np.float32)
        M   = cv2.getPerspectiveTransform(pts_orig, dst)
        warp = cv2.warpPerspective(frame, M, (ws, ws))

        warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
        bits = read_symbol_bits(warp_gray, border=0)
        # DEBUG TEMPORAL - borrar después
        if self._stable_count >= self._required_stable:
            symbol_id = ''.join(map(str, bits))
            if symbol_id not in self._processed_symbol_ids:
                preamble_bits = bits[:16]
                preamble_val  = bits_to_int(preamble_bits)
                print(f"Preámbulo leído: {hex(preamble_val)} (esperado: {hex(PREAMBLE)})")
                print(f"Primeros 80 bits: {bits[:80]}")

        # ── Estabilidad ───────────────────────────────────────────────────────
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
            symbol_id = ''.join(map(str, bits))

            if symbol_id not in self._processed_symbol_ids:
                self._processed_symbol_ids.add(symbol_id)

                # ── Parsear cabecera + payload ─────────────────────────────
                parsed = parse_frame(bits)

                if parsed is None:
                    msg = f"[FRAME descartado] preámbulo inválido"
                    print(msg)
                    self._last_event_msg = msg
                elif not parsed["crc_ok"]:
                    msg = (f"[FRAME #{parsed['n_frame']+1}/{parsed['n_total']} "
                           f"DESCARTADO] CRC error")
                    print(msg)
                    self._last_event_msg = msg
                else:
                    nf = parsed["n_frame"]
                    nt = parsed["n_total"]
                    self._n_total_expected = nt
                    self._frame_store[nf]  = parsed["payload"]
                    self._try_reconstruct()

                    msg = (f"[FRAME #{nf+1}/{nt} OK] "
                           f"CRC ✓ | payload={parsed['n_payload']} bits | "
                           f"almacenados={len(self._frame_store)}/{nt} | "
                           f"texto={len(self._decoded_text)} chars")
                    print(msg)
                    self._last_event_msg = msg
                    symbol_just_processed = True

        self._draw_hud(debug, fid_count=4, just_processed=symbol_just_processed)
        warp_vis = self._draw_grid(warp.copy(), ws)
        return debug, warp_vis, self._decoded_text

    # ── HUD ───────────────────────────────────────────────────────────────────
    def _draw_hud(self, debug, fid_count=0, just_processed=False):
        if fid_count < 4:
            cv2.putText(debug, f"Fiduciales: {fid_count}/4",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 80, 255), 2)
            return

        est_color = (0,255,80) if self._stable_count >= self._required_stable else (0,180,255)
        n_total   = self._n_total_expected or "?"
        cv2.putText(debug,
                    f"Fiduciales OK | estable: {self._stable_count}/{self._required_stable} "
                    f"| frames: {len(self._frame_store)}/{n_total}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, est_color, 2)

        if self._last_event_msg:
            color = (0,255,0) if just_processed else (180,255,180)
            cv2.putText(debug, self._last_event_msg[:90],
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(debug, self._decoded_text[-80:],
                    (10, debug.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _best_four(self, fiducials):
        centers = np.array([f["center"] for f in fiducials])
        s = centers.sum(axis=1)
        d = np.diff(centers, axis=1).flatten()
        idx = [np.argmin(s), np.argmax(s), np.argmin(d), np.argmax(d)]
        return [fiducials[i] for i in idx]

    def _draw_grid(self, img, ws):
        total_cells = SYMBOL_SIZE
        cell_px     = ws / total_cells
        for i in range(total_cells + 1):
            p = int(i * cell_px)
            cv2.line(img, (p, 0), (p, ws), (0,180,0), 1)
            cv2.line(img, (0, p), (ws, p), (0,180,0), 1)
        overlay = img.copy()
        for (row, col) in DATA_POSITIONS:
            r0 = int((row + BORDER) * cell_px)
            c0 = int((col + BORDER) * cell_px)
            r1 = int((row + BORDER + 1) * cell_px)
            c1 = int((col + BORDER + 1) * cell_px)
            cv2.rectangle(overlay, (c0,r0), (c1,r1), (255,100,0), -1)
        cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
        return img


# ─────────────────────────── MAIN LOOP ───────────────────────────────────────
if __name__ == "__main__":
    rx = Rx(scale=1, warp_size=480)
    rx.open_camera(cam_id=0)
    print("Leyendo... 'q' para salir, 'r' para reiniciar decoder.")

    last_time = time.time()
    fps_count = 0

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
        if key == ord('q'):
            break
        if key == ord('r'):
            rx.reset()
        if text:
            print(f"\r>> {text[-120:]}", end="", flush=True)

    print(f"\n\nTexto final decodificado:\n{rx._decoded_text}")
    rx.close_camera()