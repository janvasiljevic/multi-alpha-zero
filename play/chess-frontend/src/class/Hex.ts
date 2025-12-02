import { SIZE, COLORS, A_FILE, A_RANK } from "../lib/constants.ts";
import { AlgebraicNotation } from "../lib/types.ts";
import { mod } from "../lib/utils/utils.ts";
import { algebraicToQRS, qrsToAlgebraic } from "../lib/utils/utils_algebraic.ts";
import { getX, getY } from "../lib/utils/utils_xy.ts";

const primSecToMemIdx = (prim: number, s1st: number, s2nd: number): number => {
  const col = prim + 4;
  const row = col < 4 ? s1st + 4 : -s2nd + 3;

  return row * 8 + col;
};

// Debug function
const qrsToMemIdx = (g_q: number, g_r: number, g_s: number) => {
  if (g_r < 0 && 0 <= g_s) return primSecToMemIdx(g_q, g_r, g_s);
  if (g_s < 0 && 0 <= g_q) return 32 + primSecToMemIdx(g_r, g_s, g_q);
  if (g_q < 0 && 0 <= g_r) return 64 + primSecToMemIdx(g_s, g_q, g_r);

  throw new Error("Invalid qrs");
};

// Where to draw letters and numbers
// ranks are always on the bottom of the hex, ranks are on the right side of the hex
const filesIdxs = [71, 79, 87, 0, 1, 2, 3, 4, 5, 6, 7, 56, 48, 40, 32];
const ranksIdxs = [7, 15, 23, 32, 33, 34, 35, 36, 37, 38, 39, 88, 80, 72, 64];

export class Hex {
  g_q: number;
  g_r: number;
  g_s: number;

  g_col: number;
  g_row: number;

  size: number;
  x: number;
  y: number;
  color: string;
  memIdx: number;

  notation: AlgebraicNotation;

  letter: string | null = null;
  number: string | null = null;

  constructor(g_q: number, g_r: number, g_s: number) {
    this.g_q = g_q;
    this.g_r = g_r;
    this.g_s = g_s;
    this.g_col = g_q;
    this.g_row = g_r + (g_q - (g_q & 1)) / 2;
    this.size = SIZE;

    this.x = getX(this.g_q, this.size);
    this.y = getY(this.g_q, this.g_r, this.size);
    this.color = this._get_fill();
    this.memIdx = qrsToMemIdx(g_q, g_r, g_s);

    if (filesIdxs.includes(this.memIdx)) {
      this.letter = A_FILE[filesIdxs.indexOf(this.memIdx)];
    }

    if (ranksIdxs.includes(this.memIdx)) {
      this.number = A_RANK[ranksIdxs.indexOf(this.memIdx)];
    }

    this.notation = qrsToAlgebraic(this.g_q, this.g_r);

    // This is just a test for conversion. Not needed in production, but a "mini sanity test"
    const [q, r] = algebraicToQRS(this.notation);

    if (q !== this.g_q || r !== this.g_r) {
      throw new Error(`Conversion failed: ${this.notation} -> ${q}, ${r}`);
    }
  }

  _get_fill() {
    if (mod(this.g_col, 2) === 0) {
      if (mod(this.g_row, 3) === 0) return COLORS["W"];
      else if (mod(this.g_row, 3) === 1) return COLORS["G"];
      return COLORS["B"];
    }

    if (mod(this.g_row, 3) === 0) return COLORS["B"];
    if (mod(this.g_row, 3) === 1) return COLORS["W"];

    return COLORS["G"];
  }
}
