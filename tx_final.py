import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────── CRC-16 (CCITT-FALSE) ────────────────────────────
def crc16(bits: list[int]) -> int:
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


# ─────────────────────────── CLASE TX ────────────────────────────────────────
class Tx:
    PREAMBLE      = 0xDEAD
    PREAMBLE_BITS = 16
    NTOTAL_BITS   = 16
    NFRAME_BITS   = 16
    NPAYLOAD_BITS = 16
    CHECKSUM_BITS = 16
    HEADER_BITS   = PREAMBLE_BITS + NTOTAL_BITS + NFRAME_BITS + NPAYLOAD_BITS + CHECKSUM_BITS  # 80

    FID_SIZE = 7
    QUIET    = 1
    BORDER   = 2

    def __init__(self, symbol_size: int = 32):
        if symbol_size < 2 * (self.FID_SIZE + self.QUIET) + 1:
            raise ValueError(
                f"symbol_size={symbol_size} demasiado pequeño. "
                f"Mínimo: {2 * (self.FID_SIZE + self.QUIET) + 1}"
            )

        self.symbol_size = symbol_size
        self._reserved_mask = self._build_reserved_mask()
        self._data_cells = int(self._reserved_mask.size - self._reserved_mask.sum())
        self._header_cells = self.HEADER_BITS
        self._payload_cells = self._data_cells - self._header_cells
        if self._payload_cells < 2:
            raise ValueError(
                f"Símbolo {symbol_size}×{symbol_size} demasiado pequeño: "
                f"solo {self._payload_cells} celdas para payload tras la cabecera."
            )
        self._payload_bits_per_frame = self._payload_cells // 2
        self._texto  : str | None       = None
        self._binary : list[int] | None = None
        self.vec_imgs: list | None      = None

    def _patron_fiducial(self) -> np.ndarray:
        f = np.zeros((7, 7), dtype=np.uint8)
        f[1:6, 1:6] = 1
        f[2:5, 2:5] = 0
        return f

    def _build_reserved_mask(self) -> np.ndarray:
        N  = self.symbol_size
        FQ = self.FID_SIZE + self.QUIET
        mask = np.zeros((N, N), dtype=bool)
        for r0, c0 in [(0, 0), (0, N-FQ), (N-FQ, 0), (N-FQ, N-FQ)]:
            mask[r0:r0+FQ, c0:c0+FQ] = True
        return mask

    @staticmethod
    def _int_to_bits(value: int, n_bits: int) -> list[int]:
        return [(value >> (n_bits - 1 - i)) & 1 for i in range(n_bits)]

    @staticmethod
    def _manchester_encode(bits: list[int]) -> list[int]:
        enc = []
        for b in bits:
            enc += [1, 0] if b == 1 else [0, 1]
        return enc

    @staticmethod
    def _validate_text(texto: str) -> None:
        bad = [(i, c) for i, c in enumerate(texto) if ord(c) > 127]
        if bad:
            muestra = ", ".join(f"'{c}' (pos {i})" for i, c in bad[:5])
            raise ValueError(
                f"El texto contiene caracteres no ASCII (>127): {muestra}. "
                "Usa encode/decode UTF-8 antes de transmitir."
            )

    def encode(self, texto: str) -> None:
        self._validate_text(texto)
        self._texto  = texto
        self._binary = [int(b) for char in texto for b in format(ord(char), '08b')]
        self._gen_img()

    def _build_frames(self) -> list[list[int]]:
        data = list(self._binary)
        pf   = self._payload_bits_per_frame
        chunks = [data[i:i+pf] for i in range(0, max(len(data), 1), pf)]
        n_total = len(chunks)
        frames  = []

        for idx, real_bits in enumerate(chunks):
            n_real = len(real_bits)
            crc    = crc16(real_bits)
            header_cells = (
                self._int_to_bits(self.PREAMBLE, self.PREAMBLE_BITS) +
                self._int_to_bits(n_total,        self.NTOTAL_BITS)   +
                self._int_to_bits(idx,             self.NFRAME_BITS)   +
                self._int_to_bits(n_real,          self.NPAYLOAD_BITS) +
                self._int_to_bits(crc,             self.CHECKSUM_BITS)
            )
            payload_cells = self._manchester_encode(real_bits)
            used = len(header_cells) + len(payload_cells)
            pad  = [1] * (self._data_cells - used)
            frame_cells = header_cells + payload_cells + pad

            if len(frame_cells) != self._data_cells:
                raise RuntimeError(
                    f"Frame {idx}: {len(frame_cells)} celdas ≠ "
                    f"{self._data_cells} esperadas."
                )
            frames.append(frame_cells)

        return frames

    def _gen_img(self) -> None:
        N  = self.symbol_size
        FQ = self.FID_SIZE + self.QUIET

        data_positions = [
            (r, c) for r in range(N) for c in range(N)
            if not self._reserved_mask[r, c]
        ]

        frames = self._build_frames()
        fid    = self._patron_fiducial()
        B      = self.BORDER
        imgs   = []

        for frame_cells in frames:
            symbol = np.ones((N, N), dtype=np.uint8)
            for (r, c), bit in zip(data_positions, frame_cells):
                symbol[r, c] = bit
            for r0, c0, dr, dc in [
                (0,    0,    0,        0),
                (0,    N-FQ, 0,        self.QUIET),
                (N-FQ, 0,    self.QUIET, 0),
                (N-FQ, N-FQ, self.QUIET, self.QUIET),
            ]:
                symbol[r0:r0+FQ, c0:c0+FQ] = 1
                symbol[r0+dr:r0+dr+self.FID_SIZE,
                       c0+dc:c0+dc+self.FID_SIZE] = fid

            bordered = np.ones((N + 2*B, N + 2*B), dtype=np.uint8)
            bordered[B:B+N, B:B+N] = symbol
            imgs.append(bordered)

        self.vec_imgs = imgs

    # ─────────────────────────── DRAW CON LOOP ───────────────────────────────
    def draw(self, delay: float = 0.15) -> None:
        """
        Muestra los símbolos en bucle continuo hasta que el usuario
        presione la tecla 'q' (o cierre la ventana).
        """
        if self.vec_imgs is None:
            raise RuntimeError("Llama a encode() antes de draw().")

        self._stop = False

        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor('#1a1a2e')

        def on_key(event):
            if event.key == 'q':
                self._stop = True
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', on_key)

        n = len(self.vec_imgs)
        cycle = 0

        while not self._stop:
            for i, img in enumerate(self.vec_imgs):
                if self._stop:
                    break

                ax.clear()
                ax.imshow(img, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
                ax.set_title(
                    f"Símbolo {i+1} / {n}  ·  ciclo {cycle+1}",
                    color='white', fontsize=11, pad=8
                )
                ax.axis('off')
                fig.suptitle(
                    f"TX · {self.symbol_size}×{self.symbol_size} · "
                    f"{self._payload_bits_per_frame} bits útiles/símbolo  "
                    f"·  [Q] para detener",
                    color='#a0a8d0', fontsize=9
                )
                plt.tight_layout()
                plt.draw()
                plt.pause(delay)

            if not self._stop:
                cycle += 1

        # Cierra si aún está abierta
        try:
            plt.close(fig)
        except Exception:
            pass

    # ─────────────────────────── INFO ────────────────────────────────────────
    def info(self) -> None:
        N  = self.symbol_size
        nd = self._data_cells
        hc = self._header_cells
        pc = self._payload_cells
        pb = self._payload_bits_per_frame

        print(f"Símbolo {N}×{N} = {N*N} celdas totales")
        print(f"  Reservadas (fiduciales+quiet): {self._reserved_mask.sum()}")
        print(f"  Celdas de datos totales:       {nd}")
        print(f"  ├─ Cabecera (raw, sin Manchester): {hc} celdas")
        print(f"  │    preámbulo 16b + N_total 16b + N_frame 16b "
              f"+ N_payload 16b + CRC-16 16b")
        print(f"  └─ Payload (Manchester):       {pc} celdas → {pb} bits → "
              f"~{pb // 8} chars/símbolo")

        if self.vec_imgs is not None and self._texto is not None:
            print(f"  Texto: {len(self._texto)} chars → "
                  f"{len(self._binary)} bits → "
                  f"{len(self.vec_imgs)} trama(s)")


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

    tx = Tx(symbol_size=40)
    tx.encode(texto)
    tx.info()
    tx.draw(delay=0.05)