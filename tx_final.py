import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────── CONFIGURACIÓN DE TX ─────────────────────────────
# Modo principal estable:
#   "OOK_MANCHESTER" = blanco/negro + Manchester. Es el modo de la demo.
#
# Modo opcional preparado:
#   "ASK4_GRAY" = 4 niveles de gris, 2 bits por celda de payload.
#   Este modo queda implementado en TX, pero requiere RX compatible.
MODULATION = "OOK_MANCHESTER"

SYMBOL_SIZE = 40
TX_DELAY = 0.05
FULLSCREEN = True
SAVE_FIRST_FRAME_PNG = True


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
    )  # 80 bits/celdas

    # Fiduciales
    FID_SIZE = 7
    QUIET = 1
    BORDER = 2

    # Pilotos explícitos para calibración brillo/umbral en RX.
    # Deben coincidir exactamente con rx.py.
    #
    # Estos pilotos quedan fuera de las esquinas y fuera de los fiduciales.
    PILOT_BLACK_POSITIONS = [
        (8, 8), (8, 31),
        (31, 8), (31, 31),
        (10, 20), (29, 20),
        (20, 10), (20, 29),
    ]

    PILOT_WHITE_POSITIONS = [
        (8, 9), (8, 30),
        (31, 9), (31, 30),
        (11, 20), (28, 20),
        (20, 11), (20, 28),
    ]

    VALID_MODULATIONS = {"OOK_MANCHESTER", "ASK4_GRAY"}

    def __init__(self, symbol_size: int = 40, modulation: str = "OOK_MANCHESTER"):
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

        self.symbol_size = symbol_size
        self.modulation = modulation

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
            # Cada bit real usa 2 celdas Manchester.
            self._payload_bits_per_frame = self._payload_cells // 2
        elif self.modulation == "ASK4_GRAY":
            # Cada celda de payload lleva 2 bits.
            self._payload_bits_per_frame = self._payload_cells * 2
        else:
            raise RuntimeError("Modulación no reconocida.")

        self._texto: str | None = None
        self._binary: list[int] | None = None
        self.vec_imgs: list[np.ndarray] | None = None
        self._stop = False

    # ─────────────────────────── ESTRUCTURA ESPACIAL ─────────────────────────
    def _patron_fiducial(self) -> np.ndarray:
        """
        Patrón tipo finder QR simplificado:
        negro externo, blanco medio, negro interno sobre fondo blanco.
        """
        f = np.zeros((self.FID_SIZE, self.FID_SIZE), dtype=float)
        f[1:6, 1:6] = 1.0
        f[2:5, 2:5] = 0.0
        return f

    def _build_reserved_mask(self) -> np.ndarray:
        """
        Reserva:
          - fiduciales + quiet zones en las esquinas.
          - pilotos explícitos blanco/negro.
        """
        N = self.symbol_size
        FQ = self.FID_SIZE + self.QUIET

        mask = np.zeros((N, N), dtype=bool)

        # Reservar fiduciales + quiet zones
        for r0, c0 in [
            (0, 0),
            (0, N - FQ),
            (N - FQ, 0),
            (N - FQ, N - FQ),
        ]:
            mask[r0:r0 + FQ, c0:c0 + FQ] = True

        # Reservar pilotos
        for r, c in self.PILOT_BLACK_POSITIONS + self.PILOT_WHITE_POSITIONS:
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
            # Zona quiet blanca
            symbol[r0:r0 + FQ, c0:c0 + FQ] = 1.0

            # Fiducial
            symbol[
                r0 + dr:r0 + dr + self.FID_SIZE,
                c0 + dc:c0 + dc + self.FID_SIZE
            ] = fid

    def _draw_pilots(self, symbol: np.ndarray) -> None:
        """
        Pilotos explícitos:
          negros = 0.0
          blancos = 1.0

        En RX se usarán para estimar umbral adaptativo:
          threshold = (media_negros + media_blancos) / 2
        """
        for r, c in self.PILOT_BLACK_POSITIONS:
            symbol[r, c] = 0.0

        for r, c in self.PILOT_WHITE_POSITIONS:
            symbol[r, c] = 1.0

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
    def _ask4_encode(bits: list[int]) -> list[float]:
        """
        4-ASK en escala de grises:
          00 -> 0.15
          01 -> 0.40
          10 -> 0.65
          11 -> 0.90

        Este modo queda implementado para cumplir segunda modulación.
        Requiere RX compatible para demodular 4 niveles.
        """
        levels = {
            (0, 0): 0.15,
            (0, 1): 0.40,
            (1, 0): 0.65,
            (1, 1): 0.90,
        }

        cells = []

        for i in range(0, len(bits), 2):
            b0 = bits[i]
            b1 = bits[i + 1] if i + 1 < len(bits) else 0
            cells.append(levels[(b0, b1)])

        return cells

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
        self._binary = [
            int(b)
            for char in texto
            for b in format(ord(char), "08b")
        ]

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
                payload_cells = self._ask4_encode(real_bits)
                pad_value = 1.0
            else:
                raise RuntimeError("Modulación no reconocida.")

            frame_cells = [float(x) for x in header_cells] + [float(x) for x in payload_cells]

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
        imgs = []

        for frame_cells in frames:
            # Fondo blanco dentro del símbolo.
            symbol = np.ones((N, N), dtype=float)

            # Datos y cabecera
            for (r, c), value in zip(self._data_positions, frame_cells):
                symbol[r, c] = float(value)

            # Fiduciales y pilotos se dibujan después para garantizar reserva.
            self._draw_fiducials(symbol)
            self._draw_pilots(symbol)

            # Borde blanco externo para separación visual del fondo oscuro.
            bordered = np.ones((N + 2 * B, N + 2 * B), dtype=float)
            bordered[B:B + N, B:B + N] = symbol

            imgs.append(bordered)

        self.vec_imgs = imgs

    # ─────────────────────────── DRAW EN TIEMPO REAL ─────────────────────────
    def draw(self, delay: float = 0.05, fullscreen: bool = True) -> None:
        """
        Muestra los símbolos en bucle continuo hasta que el usuario presione 'q'.

        Ventana con fondo oscuro y sin ejes para reducir interferencias visuales.
        """
        if self.vec_imgs is None:
            raise RuntimeError("Llama a encode() antes de draw().")

        self._stop = False

        fig = plt.figure(figsize=(8, 8))
        fig.patch.set_facecolor("black")

        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.set_facecolor("black")
        ax.axis("off")

        try:
            manager = plt.get_current_fig_manager()
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
        cycle = 0

        print("\nTX en ejecución.")
        print("Presiona 'q' sobre la ventana del TX para detener.")
        print(f"Delay por trama: {delay:.3f} s")
        print(f"Tramas por ciclo: {n}")
        print(f"Tiempo ideal de ciclo TX: {n * delay:.3f} s\n")

        image_artist = None

        while not self._stop:
            for i, img in enumerate(self.vec_imgs):
                if self._stop:
                    break

                if image_artist is None:
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

            if not self._stop:
                cycle += 1

        try:
            plt.close(fig)
        except Exception:
            pass

    # ─────────────────────────── EXPORTACIÓN DEBUG ───────────────────────────
    def save_first_frame(self, filename: str = "tx_first_frame.png") -> None:
        if self.vec_imgs is None or len(self.vec_imgs) == 0:
            raise RuntimeError("No hay frames generados.")

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
        print(f"Pilotos negros: {len(self.PILOT_BLACK_POSITIONS)}")
        print(f"Pilotos blancos: {len(self.PILOT_WHITE_POSITIONS)}")
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
            print(f"Payload 4-ASK: {pc} celdas → {pb} bits útiles/frame")
            print(f"Aprox. chars útiles/frame: {pb // 8}")

        if self.vec_imgs is not None and self._texto is not None:
            print(
                f"Texto: {len(self._texto)} chars → "
                f"{len(self._binary)} bits → "
                f"{len(self.vec_imgs)} trama(s)"
            )

        print("")


# ─────────────────────────────── DEMO ────────────────────────────────────────
if __name__ == "__main__":
    texto = (
        "La vision artificial permite interpretar imagenes mediante algoritmos "
        "que detectan patrones, formas y relaciones espaciales. Los sistemas "
        "modernos utilizan transformaciones geometricas, segmentacion y analisis "
        "de contornos para identificar objetos incluso bajo rotacion o perspectiva. "
        "Los fiduciales son referencias visuales usadas para calcular orientacion, "
        "escala y posicion, facilitando la reconstruccion proyectiva y la extraccion "
        "robusta de informacion contenida dentro de una region determinada."
    )

    tx = Tx(symbol_size=SYMBOL_SIZE, modulation=MODULATION)
    tx.encode(texto)
    tx.info()

    if SAVE_FIRST_FRAME_PNG:
        tx.save_first_frame("tx_first_frame.png")

    tx.draw(delay=TX_DELAY, fullscreen=FULLSCREEN)