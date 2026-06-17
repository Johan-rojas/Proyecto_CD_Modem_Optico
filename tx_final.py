import os
import sys
import math
import matplotlib

matplotlib.rcParams["toolbar"] = "None"

import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────── CONFIGURACIÓN DE TX ─────────────────────────────
# Modos disponibles:
#   OOK_MANCHESTER  → modo principal estable
#   ASK4_GRAY       → segunda modulación física con repetición espacial
#   CSK_RGB         → modulación por color RGB/CSK experimental
#
# Uso:
#   python tx_final.py OOK_MANCHESTER
#   python tx_final.py ASK4_GRAY
#   python tx_final.py CSK_RGB
MODULATION = "OOK_MANCHESTER"

SYMBOL_SIZE = 40

# Delay por modulación
TX_DELAY_OOK = 0.20
TX_DELAY_ASK4 = 0.20
TX_DELAY_CSK = 0.20

FULLSCREEN = True

# Modo limpio para sustentación:
# False evita salidas extra antes de abrir la ventana de transmisión.
SAVE_FIRST_FRAME_PNG = False
SAVE_MODULATION_EXAMPLES = False
RUN_DIGITAL_LOOPBACK_TEST = False

# Repetición espacial para 4ASK:
# cada símbolo 4ASK, que representa 2 bits, se dibuja en 3 celdas consecutivas.
ASK4_REPEAT = 3

# Repetición espacial para CSK/RGB.
# 1 = máxima velocidad; si la cámara confunde colores, se puede subir a 2.
CSK_REPEAT = 1


# ─────────────────────────── TEXTO DE PRUEBA ─────────────────────────────────
DEMO_TEXT = (
    "La vision artificial permite interpretar imagenes mediante algoritmos "
    "que detectan patrones, formas y relaciones espaciales. Los sistemas "
    "modernos utilizan transformaciones geometricas, segmentacion y analisis "
    "de contornos para identificar objetos incluso bajo rotacion o perspectiva. "
    "Los fiduciales son referencias visuales usadas para calcular orientacion, "
    "escala y posicion, facilitando la reconstruccion proyectiva y la extraccion "
    "robusta de informacion contenida dentro de una region determinada."
)


# ─────────────────────────── CRC-16 CCITT-FALSE ──────────────────────────────
def crc16(bits: list[int]) -> int:
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


# ─────────────────────────── UTILIDADES ──────────────────────────────────────
def bits_to_int(bits: list[int]) -> int:
    val = 0

    for b in bits:
        val = (val << 1) | int(b)

    return val


def text_to_bits(text: str) -> list[int]:
    return [int(b) for char in text for b in format(ord(char), "08b")]


def bits_to_text(bits: list[int]) -> str:
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

    errors = 0

    for i in range(max_len):
        expected = expected_bits[i] if i < len(expected_bits) else None
        received = received_bits[i] if i < len(received_bits) else None

        if expected != received:
            errors += 1

    return {
        "expected_bits": len(expected_bits),
        "received_bits": len(received_bits),
        "compared_bits": max_len,
        "bit_errors": errors,
        "ber": errors / max_len,
    }


def get_modulation_from_args(default: str) -> str:
    if len(sys.argv) >= 2:
        return sys.argv[1].strip().upper()
    return default


def load_text_from_args(default_text: str) -> str:
    """
    Permite transmitir un archivo .txt externo:
        python tx_final.py OOK_MANCHESTER mensaje.txt
        python tx_final.py ASK4_GRAY mensaje.txt

    Si no se pasa archivo, usa DEMO_TEXT.
    Se conserva ASCII para que TX/RX comparen 8 bits por carácter.
    """
    if len(sys.argv) >= 3:
        path = sys.argv[2]
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    return default_text


def tx_delay_for_modulation(modulation: str) -> float:
    if modulation == "ASK4_GRAY":
        return TX_DELAY_ASK4
    if modulation == "CSK_RGB":
        return TX_DELAY_CSK
    return TX_DELAY_OOK


# ─────────────────────────── CLASE TX ────────────────────────────────────────
class Tx:
    # Cabecera
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

    # Fiduciales
    FID_SIZE = 7
    QUIET = 1
    BORDER = 2

    VALID_MODULATIONS = {
        "OOK_MANCHESTER",
        "ASK4_GRAY",
        "CSK_RGB",
    }

    # Pilotos de 4 niveles.
    # Deben coincidir exactamente con rx.py.
    PILOT_LEVEL_POSITIONS = {
        0: [
            (8, 8), (31, 31),
            (10, 20), (20, 10),
        ],
        1: [
            (8, 9), (31, 30),
            (11, 20), (20, 11),
        ],
        2: [
            (8, 30), (31, 9),
            (28, 20), (20, 28),
        ],
        3: [
            (8, 31), (31, 8),
            (29, 20), (20, 29),
        ],
    }

    # Niveles físicos para ASK4.
    # Se separan bien para que la cámara pueda distinguirlos mejor.
    ASK4_BITS_TO_LEVEL = {
        (0, 0): 0.05,
        (0, 1): 0.35,
        (1, 0): 0.70,
        (1, 1): 0.95,
    }

    ASK4_LEVEL_TO_BITS = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0),
        3: (1, 1),
    }

    ASK4_LEVEL_VALUES = {
        0: 0.05,
        1: 0.35,
        2: 0.70,
        3: 0.95,
    }

    # Modulación en color tipo CSK/RGB.
    # Cada celda de color transporta 2 bits. Se usan pilotos de color
    # para que el receptor clasifique por cercanía a los colores reales capturados.
    CSK_BITS_TO_LEVEL = {
        (0, 0): 0,  # rojo
        (0, 1): 1,  # verde
        (1, 0): 2,  # azul
        (1, 1): 3,  # amarillo
    }

    CSK_LEVEL_TO_BITS = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0),
        3: (1, 1),
    }

    # Matplotlib interpreta imágenes RGB.
    CSK_LEVEL_VALUES = {
        0: np.array([1.0, 0.0, 0.0], dtype=float),  # rojo
        1: np.array([0.0, 1.0, 0.0], dtype=float),  # verde
        2: np.array([0.0, 0.0, 1.0], dtype=float),  # azul
        3: np.array([1.0, 1.0, 0.0], dtype=float),  # amarillo
    }

    def __init__(
        self,
        symbol_size: int = 40,
        modulation: str = "OOK_MANCHESTER",
        ask4_repeat: int = ASK4_REPEAT,
        csk_repeat: int = CSK_REPEAT,
    ):
        modulation = modulation.upper()

        if modulation not in self.VALID_MODULATIONS:
            raise ValueError(
                f"Modulación no soportada: {modulation}. "
                f"Opciones válidas: {sorted(self.VALID_MODULATIONS)}"
            )

        if symbol_size < 2 * (self.FID_SIZE + self.QUIET) + 1:
            raise ValueError(
                f"symbol_size={symbol_size} demasiado pequeño. "
                f"Mínimo: {2 * (self.FID_SIZE + self.QUIET) + 1}"
            )

        if ask4_repeat < 1:
            raise ValueError("ASK4_REPEAT debe ser >= 1")

        if csk_repeat < 1:
            raise ValueError("CSK_REPEAT debe ser >= 1")

        self.symbol_size = symbol_size
        self.modulation = modulation
        self.ask4_repeat = ask4_repeat
        self.csk_repeat = csk_repeat

        self._reserved_mask = self._build_reserved_mask()
        self._data_positions = self._build_data_positions()

        self._data_cells = len(self._data_positions)
        self._header_cells = self.HEADER_BITS
        self._payload_cells = self._data_cells - self._header_cells

        if self._payload_cells < 2:
            raise ValueError(
                f"Símbolo {symbol_size}×{symbol_size} demasiado pequeño: "
                f"solo {self._payload_cells} celdas para payload tras cabecera."
            )

        if self.modulation == "OOK_MANCHESTER":
            self._payload_bits_per_frame = self._payload_cells // 2

        elif self.modulation == "ASK4_GRAY":
            # Cada símbolo 4ASK representa 2 bits, pero se repite ask4_repeat veces.
            usable_ask4_symbols = self._payload_cells // self.ask4_repeat
            self._payload_bits_per_frame = usable_ask4_symbols * 2

        elif self.modulation == "CSK_RGB":
            # Cada celda de color CSK representa 2 bits.
            usable_csk_symbols = self._payload_cells // self.csk_repeat
            self._payload_bits_per_frame = usable_csk_symbols * 2

        else:
            raise RuntimeError("Modulación no reconocida.")

        self._texto: str | None = None
        self._binary: list[int] | None = None
        self.vec_imgs: list[np.ndarray] | None = None
        self._frame_cells_list: list[list[float]] | None = None
        self._stop = False

    # ─────────────────────────── ESPACIAL ─────────────────────────────────────
    def _patron_fiducial(self) -> np.ndarray:
        f = np.zeros((self.FID_SIZE, self.FID_SIZE), dtype=float)
        f[1:6, 1:6] = 1.0
        f[2:5, 2:5] = 0.0
        return f

    def _all_pilot_positions(self) -> list[tuple[int, int]]:
        positions = []

        for pts in self.PILOT_LEVEL_POSITIONS.values():
            positions.extend(pts)

        return positions

    def _build_reserved_mask(self) -> np.ndarray:
        N = self.symbol_size
        FQ = self.FID_SIZE + self.QUIET

        mask = np.zeros((N, N), dtype=bool)

        # Fiduciales + quiet zones
        for r0, c0 in [
            (0, 0),
            (0, N - FQ),
            (N - FQ, 0),
            (N - FQ, N - FQ),
        ]:
            mask[r0:r0 + FQ, c0:c0 + FQ] = True

        # Pilotos explícitos
        for r, c in self._all_pilot_positions():
            if not (0 <= r < N and 0 <= c < N):
                raise ValueError(f"Piloto fuera de la grilla: {(r, c)}")
            mask[r, c] = True

        return mask

    def _build_data_positions(self) -> list[tuple[int, int]]:
        return [
            (r, c)
            for r in range(self.symbol_size)
            for c in range(self.symbol_size)
            if not self._reserved_mask[r, c]
        ]

    def _draw_fiducials(self, symbol: np.ndarray) -> None:
        N = self.symbol_size
        FQ = self.FID_SIZE + self.QUIET
        fid = self._patron_fiducial()

        for r0, c0, dr, dc in [
            (0, 0, 0, 0),
            (0, N - FQ, 0, self.QUIET),
            (N - FQ, 0, self.QUIET, 0),
            (N - FQ, N - FQ, self.QUIET, self.QUIET),
        ]:
            symbol[r0:r0 + FQ, c0:c0 + FQ] = 1.0

            if symbol.ndim == 3:
                symbol[
                    r0 + dr:r0 + dr + self.FID_SIZE,
                    c0 + dc:c0 + dc + self.FID_SIZE,
                    :
                ] = fid[:, :, None]
            else:
                symbol[
                    r0 + dr:r0 + dr + self.FID_SIZE,
                    c0 + dc:c0 + dc + self.FID_SIZE
                ] = fid

    def _pilot_value_for_level(self, level: int) -> float:
        if self.modulation == "OOK_MANCHESTER":
            # En OOK usamos los pilotos como referencia binaria:
            # niveles 0 y 1 negros, niveles 2 y 3 blancos.
            return 0.0 if level in (0, 1) else 1.0

        if self.modulation == "ASK4_GRAY":
            return self.ASK4_LEVEL_VALUES[level]

        if self.modulation == "CSK_RGB":
            return self.CSK_LEVEL_VALUES[level]

        raise RuntimeError("Modulación no reconocida.")

    def _draw_pilots(self, symbol: np.ndarray) -> None:
        for level, positions in self.PILOT_LEVEL_POSITIONS.items():
            value = self._pilot_value_for_level(level)

            for r, c in positions:
                symbol[r, c] = value

    # ─────────────────────────── UTILIDADES BINARIAS ─────────────────────────
    @staticmethod
    def _int_to_bits(value: int, n_bits: int) -> list[int]:
        return [(value >> (n_bits - 1 - i)) & 1 for i in range(n_bits)]

    @staticmethod
    def _manchester_encode(bits: list[int]) -> list[int]:
        enc = []

        for b in bits:
            if b == 1:
                enc += [1, 0]
            else:
                enc += [0, 1]

        return enc

    @staticmethod
    def _manchester_decode(bits: list[int]) -> list[int] | None:
        if len(bits) % 2 != 0:
            return None

        dec = []

        for i in range(0, len(bits), 2):
            a = bits[i]
            b = bits[i + 1]

            if a == 1 and b == 0:
                dec.append(1)
            elif a == 0 and b == 1:
                dec.append(0)
            else:
                return None

        return dec

    @classmethod
    def _ask4_encode(cls, bits: list[int], repeat: int) -> list[float]:
        cells = []

        for i in range(0, len(bits), 2):
            b0 = bits[i]
            b1 = bits[i + 1] if i + 1 < len(bits) else 0
            level_value = cls.ASK4_BITS_TO_LEVEL[(b0, b1)]

            # Repetición espacial
            for _ in range(repeat):
                cells.append(level_value)

        return cells

    @classmethod
    def _ask4_decode_ideal(cls, cells: list[float], n_bits: int, repeat: int) -> list[int]:
        levels = [
            (cls.ASK4_LEVEL_VALUES[0], (0, 0)),
            (cls.ASK4_LEVEL_VALUES[1], (0, 1)),
            (cls.ASK4_LEVEL_VALUES[2], (1, 0)),
            (cls.ASK4_LEVEL_VALUES[3], (1, 1)),
        ]

        decoded = []

        for i in range(0, len(cells), repeat):
            group = cells[i:i + repeat]

            if not group:
                break

            value = float(np.median(group))
            _, bits_pair = min(levels, key=lambda item: abs(value - item[0]))

            decoded.extend(list(bits_pair))

            if len(decoded) >= n_bits:
                break

        return decoded[:n_bits]


    @classmethod
    def _csk_encode(cls, bits: list[int], repeat: int) -> list[np.ndarray]:
        cells = []

        for i in range(0, len(bits), 2):
            b0 = bits[i]
            b1 = bits[i + 1] if i + 1 < len(bits) else 0
            level = cls.CSK_BITS_TO_LEVEL[(b0, b1)]
            color_value = cls.CSK_LEVEL_VALUES[level]

            # Repetición espacial opcional para robustez.
            for _ in range(repeat):
                cells.append(color_value.copy())

        return cells

    @classmethod
    def _csk_decode_ideal(cls, cells: list, n_bits: int, repeat: int) -> list[int]:
        levels = [
            (cls.CSK_LEVEL_VALUES[0], (0, 0)),
            (cls.CSK_LEVEL_VALUES[1], (0, 1)),
            (cls.CSK_LEVEL_VALUES[2], (1, 0)),
            (cls.CSK_LEVEL_VALUES[3], (1, 1)),
        ]

        decoded = []

        for i in range(0, len(cells), repeat):
            group = cells[i:i + repeat]

            if not group:
                break

            value = np.median(np.array(group, dtype=float), axis=0)
            _, bits_pair = min(levels, key=lambda item: float(np.linalg.norm(value - item[0])))

            decoded.extend(list(bits_pair))

            if len(decoded) >= n_bits:
                break

        return decoded[:n_bits]

    @staticmethod
    def _validate_text(texto: str) -> None:
        bad = [(i, c) for i, c in enumerate(texto) if ord(c) > 127]

        if bad:
            muestra = ", ".join(f"'{c}' (pos {i})" for i, c in bad[:5])
            raise ValueError(
                f"El texto contiene caracteres no ASCII (>127): {muestra}. "
                "Usa solo caracteres ASCII para esta versión del módem."
            )

    # ─────────────────────────── ENCODING ─────────────────────────────────────
    def encode(self, texto: str) -> None:
        self._validate_text(texto)

        self._texto = texto
        self._binary = text_to_bits(texto)

        self._gen_img()

    def _build_header(self, n_total: int, idx: int, n_real: int, crc: int) -> list[int]:
        return (
            self._int_to_bits(self.PREAMBLE, self.PREAMBLE_BITS) +
            self._int_to_bits(n_total, self.NTOTAL_BITS) +
            self._int_to_bits(idx, self.NFRAME_BITS) +
            self._int_to_bits(n_real, self.NPAYLOAD_BITS) +
            self._int_to_bits(crc, self.CHECKSUM_BITS)
        )

    def _build_frames(self) -> list[list[float]]:
        if self._binary is None:
            raise RuntimeError("No hay bits para transmitir. Llama a encode().")

        data = list(self._binary)
        pf = self._payload_bits_per_frame

        chunks = [
            data[i:i + pf]
            for i in range(0, max(len(data), 1), pf)
        ]

        n_total = len(chunks)
        frames = []

        for idx, real_bits in enumerate(chunks):
            n_real = len(real_bits)
            crc = crc16(real_bits)

            header_cells = self._build_header(n_total, idx, n_real, crc)

            if self.modulation == "OOK_MANCHESTER":
                payload_cells = self._manchester_encode(real_bits)
                pad_value = 1.0

            elif self.modulation == "ASK4_GRAY":
                payload_cells = self._ask4_encode(real_bits, self.ask4_repeat)
                pad_value = 1.0

            elif self.modulation == "CSK_RGB":
                payload_cells = self._csk_encode(real_bits, self.csk_repeat)
                pad_value = np.array([1.0, 1.0, 1.0], dtype=float)

            else:
                raise RuntimeError("Modulación no reconocida.")

            frame_cells = [float(x) for x in header_cells] + list(payload_cells)

            used = len(frame_cells)
            remaining = self._data_cells - used

            if remaining < 0:
                raise RuntimeError(
                    f"Frame {idx}: usa {used} celdas, "
                    f"pero solo hay {self._data_cells} disponibles."
                )

            frame_cells += [pad_value] * remaining

            if len(frame_cells) != self._data_cells:
                raise RuntimeError(
                    f"Frame {idx}: {len(frame_cells)} celdas ≠ "
                    f"{self._data_cells} esperadas."
                )

            frames.append(frame_cells)

        return frames

    def _gen_img(self) -> None:
        N = self.symbol_size
        B = self.BORDER

        frames = self._build_frames()
        self._frame_cells_list = frames

        imgs = []

        for frame_cells in frames:
            if self.modulation == "CSK_RGB":
                symbol = np.ones((N, N, 3), dtype=float)
            else:
                symbol = np.ones((N, N), dtype=float)

            for (r, c), value in zip(self._data_positions, frame_cells):
                if self.modulation == "CSK_RGB" and isinstance(value, np.ndarray):
                    symbol[r, c, :] = value
                elif self.modulation == "CSK_RGB":
                    symbol[r, c, :] = float(value)
                else:
                    symbol[r, c] = float(value)

            self._draw_fiducials(symbol)
            self._draw_pilots(symbol)

            if self.modulation == "CSK_RGB":
                bordered = np.ones((N + 2 * B, N + 2 * B, 3), dtype=float)
                bordered[B:B + N, B:B + N, :] = symbol
            else:
                bordered = np.ones((N + 2 * B, N + 2 * B), dtype=float)
                bordered[B:B + N, B:B + N] = symbol

            imgs.append(bordered)

        self.vec_imgs = imgs

    # ─────────────────────────── LOOPBACK DIGITAL ─────────────────────────────
    def digital_loopback_decode(self) -> dict:
        if self._frame_cells_list is None:
            raise RuntimeError("No hay tramas generadas. Llama a encode().")

        frame_store: dict[int, list[int]] = {}
        expected_total = None
        crc_ok_count = 0
        crc_fail_count = 0
        invalid_count = 0

        for frame_cells in self._frame_cells_list:
            raw_bits = [1 if x >= 0.5 else 0 for x in frame_cells[:self.HEADER_BITS]]

            ptr = 0

            preamble = bits_to_int(raw_bits[ptr:ptr + self.PREAMBLE_BITS])
            ptr += self.PREAMBLE_BITS

            n_total = bits_to_int(raw_bits[ptr:ptr + self.NTOTAL_BITS])
            ptr += self.NTOTAL_BITS

            n_frame = bits_to_int(raw_bits[ptr:ptr + self.NFRAME_BITS])
            ptr += self.NFRAME_BITS

            n_payload = bits_to_int(raw_bits[ptr:ptr + self.NPAYLOAD_BITS])
            ptr += self.NPAYLOAD_BITS

            crc_rx = bits_to_int(raw_bits[ptr:ptr + self.CHECKSUM_BITS])
            ptr += self.CHECKSUM_BITS

            if preamble != self.PREAMBLE:
                invalid_count += 1
                continue

            if n_total <= 0 or n_frame < 0 or n_frame >= n_total:
                invalid_count += 1
                continue

            payload_start = self.HEADER_BITS

            if self.modulation == "OOK_MANCHESTER":
                payload_cells_count = n_payload * 2
                payload_cells = frame_cells[payload_start:payload_start + payload_cells_count]
                payload_cell_bits = [1 if x >= 0.5 else 0 for x in payload_cells]
                payload_bits = self._manchester_decode(payload_cell_bits)

                if payload_bits is None:
                    crc_fail_count += 1
                    continue

            elif self.modulation == "ASK4_GRAY":
                ask4_symbols = math.ceil(n_payload / 2)
                payload_cells_count = ask4_symbols * self.ask4_repeat
                payload_cells = frame_cells[payload_start:payload_start + payload_cells_count]
                payload_bits = self._ask4_decode_ideal(
                    payload_cells,
                    n_payload,
                    self.ask4_repeat
                )

            elif self.modulation == "CSK_RGB":
                csk_symbols = math.ceil(n_payload / 2)
                payload_cells_count = csk_symbols * self.csk_repeat
                payload_cells = frame_cells[payload_start:payload_start + payload_cells_count]
                payload_bits = self._csk_decode_ideal(
                    payload_cells,
                    n_payload,
                    self.csk_repeat
                )

            else:
                invalid_count += 1
                continue

            crc_calc = crc16(payload_bits)

            if crc_calc != crc_rx:
                crc_fail_count += 1
                continue

            crc_ok_count += 1
            expected_total = n_total
            frame_store[n_frame] = payload_bits

        reconstructed_bits = []

        if expected_total is not None and len(frame_store) == expected_total:
            for i in range(expected_total):
                reconstructed_bits.extend(frame_store[i])

        reconstructed_text = bits_to_text(reconstructed_bits)
        expected_bits = text_to_bits(self._texto or "")

        ber_metrics = compute_ber(
            expected_bits,
            reconstructed_bits[:len(expected_bits)]
        )

        return {
            "modulation": self.modulation,
            "expected_frames": expected_total,
            "decoded_frames": len(frame_store),
            "crc_ok": crc_ok_count,
            "crc_fail": crc_fail_count,
            "invalid": invalid_count,
            "expected_chars": len(self._texto or ""),
            "received_chars": len(reconstructed_text),
            "text_match": reconstructed_text == (self._texto or ""),
            "ber": ber_metrics["ber"],
            "bit_errors": ber_metrics["bit_errors"],
            "expected_bits": ber_metrics["expected_bits"],
            "received_bits": ber_metrics["received_bits"],
            "compared_bits": ber_metrics["compared_bits"],
        }

    def print_loopback_report(self) -> None:
        result = self.digital_loopback_decode()

        print("Prueba digital interna")
        print("──────────────────────")
        print(f"Modulación:             {result['modulation']}")
        print(f"Frames esperados:       {result['expected_frames']}")
        print(f"Frames decodificados:   {result['decoded_frames']}")
        print(f"CRC OK:                 {result['crc_ok']}")
        print(f"CRC fallidos:           {result['crc_fail']}")
        print(f"Frames inválidos:       {result['invalid']}")
        print(f"Caracteres esperados:   {result['expected_chars']}")
        print(f"Caracteres recibidos:   {result['received_chars']}")
        print(f"Bits esperados:         {result['expected_bits']}")
        print(f"Bits comparados:        {result['compared_bits']}")
        print(f"Errores de bit:         {result['bit_errors']}")
        print(f"BER:                    {result['ber']:.2e}")
        print(f"Texto exacto:           {'SI' if result['text_match'] else 'NO'}")
        print("")

    # ─────────────────────────── DRAW ─────────────────────────────────────────
    def draw(self, delay: float = 0.05, fullscreen: bool = True) -> None:
        if self.vec_imgs is None:
            raise RuntimeError("Llama a encode() antes de draw().")

        self._stop = False

        fig = plt.figure(figsize=(8, 8))
        fig.patch.set_facecolor("black")

        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.set_facecolor("black")
        ax.axis("off")

        try:
            fig.canvas.manager.set_window_title("TX Modem Optico")
        except Exception:
            pass

        try:
            manager = plt.get_current_fig_manager()

            if hasattr(manager, "toolbar") and manager.toolbar is not None:
                try:
                    manager.toolbar.hide()
                except Exception:
                    pass

            if fullscreen:
                manager.full_screen_toggle()

        except Exception:
            pass

        def on_key(event):
            if event.key == "q":
                self._stop = True
                plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", on_key)

        n = len(self.vec_imgs)

        print("\nTX en ejecución.")
        print("Presiona 'q' sobre la ventana del TX para detener.")
        print(f"Modulación activa: {self.modulation}")
        print(f"Delay por trama: {delay:.3f} s")
        print(f"Tramas por ciclo: {n}")
        print(f"Tiempo ideal de ciclo TX: {n * delay:.3f} s")

        if self.modulation == "ASK4_GRAY":
            print(f"ASK4_REPEAT: {self.ask4_repeat}")
            print("4ASK usa repetición espacial para mejorar robustez física.\n")
        else:
            print("")

        image_artist = None

        while not self._stop:
            for img in self.vec_imgs:
                if self._stop:
                    break

                if image_artist is None:
                    if img.ndim == 3:
                        image_artist = ax.imshow(
                            img,
                            interpolation="nearest",
                            aspect="equal"
                        )
                    else:
                        image_artist = ax.imshow(
                            img,
                            cmap="gray",
                            interpolation="nearest",
                            vmin=0,
                            vmax=1,
                            aspect="equal"
                        )
                else:
                    image_artist.set_data(img)

                fig.canvas.draw_idle()
                plt.pause(delay)

        try:
            plt.close(fig)
        except Exception:
            pass

    # ─────────────────────────── EXPORTACIÓN ──────────────────────────────────
    def save_first_frame(self, filename: str = "tx_first_frame.png") -> None:
        if self.vec_imgs is None or len(self.vec_imgs) == 0:
            raise RuntimeError("No hay frames generados.")

        if self.vec_imgs[0].ndim == 3:
            plt.imsave(filename, self.vec_imgs[0])
        else:
            plt.imsave(filename, self.vec_imgs[0], cmap="gray", vmin=0, vmax=1)
        print(f"Primer frame guardado en: {filename}")

    # ─────────────────────────── INFO ─────────────────────────────────────────
    def info(self) -> None:
        N = self.symbol_size
        nd = self._data_cells
        hc = self._header_cells
        pc = self._payload_cells
        pb = self._payload_bits_per_frame

        print("Información TX")
        print("──────────────")
        print(f"Modulación: {self.modulation}")
        print(f"Símbolo lógico: {N}×{N} = {N * N} celdas")
        print(f"Borde externo: {self.BORDER} celdas")
        print(f"Fiduciales: 4 patrones de {self.FID_SIZE}×{self.FID_SIZE}")
        print(f"Quiet zone por fiducial: {self.QUIET}")
        print(f"Pilotos por nivel: 4 niveles × 4 pilotos = 16 pilotos")
        print(f"Celdas reservadas totales: {int(self._reserved_mask.sum())}")
        print(f"Celdas de datos disponibles: {nd}")
        print(f"Cabecera: {hc} celdas")
        print(
            "  preámbulo 16b + N_total 16b + N_frame 16b "
            "+ N_payload 16b + CRC-16 16b"
        )

        if self.modulation == "OOK_MANCHESTER":
            print(f"Payload Manchester: {pc} celdas → {pb} bits útiles/frame")
            print(f"Aprox. chars útiles/frame: {pb // 8}")

        elif self.modulation == "ASK4_GRAY":
            print(f"Payload 4ASK con repetición {self.ask4_repeat}×")
            print(f"{pc} celdas payload → {pb} bits útiles/frame")
            print(f"Aprox. chars útiles/frame: {pb // 8}")

        elif self.modulation == "CSK_RGB":
            print(f"Payload CSK/RGB con repetición {self.csk_repeat}×")
            print(f"{pc} celdas payload → {pb} bits útiles/frame")
            print(f"Aprox. chars útiles/frame: {pb // 8}")
            print("Colores: rojo, verde, azul y amarillo, calibrados con pilotos.")

        if self.vec_imgs is not None and self._texto is not None:
            print(
                f"Texto: {len(self._texto)} chars → "
                f"{len(self._binary)} bits → "
                f"{len(self.vec_imgs)} trama(s)"
            )

        print("")


# ─────────────────────────── DEMOSTRACIONES ──────────────────────────────────
def save_modulation_examples(texto: str) -> None:
    os.makedirs("debug_tx", exist_ok=True)

    tx_ook = Tx(symbol_size=SYMBOL_SIZE, modulation="OOK_MANCHESTER")
    tx_ook.encode(texto)
    tx_ook.save_first_frame(os.path.join("debug_tx", "frame_OOK_MANCHESTER.png"))

    tx_ask4 = Tx(
        symbol_size=SYMBOL_SIZE,
        modulation="ASK4_GRAY",
        ask4_repeat=ASK4_REPEAT,
    )
    tx_ask4.encode(texto)
    tx_ask4.save_first_frame(os.path.join("debug_tx", "frame_ASK4_GRAY_REPEAT3.png"))

    tx_csk = Tx(
        symbol_size=SYMBOL_SIZE,
        modulation="CSK_RGB",
        ask4_repeat=ASK4_REPEAT,
        csk_repeat=CSK_REPEAT,
    )
    tx_csk.encode(texto)
    tx_csk.save_first_frame(os.path.join("debug_tx", "frame_CSK_RGB.png"))

    print("Ejemplos de modulación guardados en debug_tx:")
    print("  - frame_OOK_MANCHESTER.png")
    print("  - frame_ASK4_GRAY_REPEAT3.png")
    print("  - frame_CSK_RGB.png")
    print("")


# ─────────────────────────────── MAIN ────────────────────────────────────────
if __name__ == "__main__":
    modulation = get_modulation_from_args(MODULATION)
    tx_text = load_text_from_args(DEMO_TEXT)

    print("=" * 70)
    print("TX - MODEM OPTICO")
    print(f"Modulación seleccionada: {modulation}")
    print(f"Texto a transmitir: {len(tx_text)} caracteres")
    print("=" * 70)

    if SAVE_MODULATION_EXAMPLES:
        save_modulation_examples(tx_text)

    if RUN_DIGITAL_LOOPBACK_TEST:
        print("Prueba interna OOK_MANCHESTER")
        tx_test_ook = Tx(symbol_size=SYMBOL_SIZE, modulation="OOK_MANCHESTER")
        tx_test_ook.encode(tx_text)
        tx_test_ook.print_loopback_report()

        print("Prueba interna ASK4_GRAY")
        tx_test_ask = Tx(
            symbol_size=SYMBOL_SIZE,
            modulation="ASK4_GRAY",
            ask4_repeat=ASK4_REPEAT,
        )
        tx_test_ask.encode(tx_text)
        tx_test_ask.print_loopback_report()

        print("Prueba interna CSK_RGB")
        tx_test_csk = Tx(
            symbol_size=SYMBOL_SIZE,
            modulation="CSK_RGB",
            ask4_repeat=ASK4_REPEAT,
            csk_repeat=CSK_REPEAT,
        )
        tx_test_csk.encode(tx_text)
        tx_test_csk.print_loopback_report()

    tx = Tx(
        symbol_size=SYMBOL_SIZE,
        modulation=modulation,
        ask4_repeat=ASK4_REPEAT,
        csk_repeat=CSK_REPEAT,
    )

    tx.encode(tx_text)
    tx.info()

    if SAVE_FIRST_FRAME_PNG:
        tx.save_first_frame("tx_first_frame.png")

    tx_delay = tx_delay_for_modulation(modulation)
    tx.draw(delay=tx_delay, fullscreen=FULLSCREEN)
