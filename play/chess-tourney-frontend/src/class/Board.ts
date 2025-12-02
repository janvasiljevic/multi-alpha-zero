import { Hex } from "./Hex";

export class Board {
  viewbox = [0, 0, 0, 0];
  width = 0;
  height = 0;
  hexagons: Hex[] = [];

  public constructor(size: number) {
    const elMap = new Map<string, Hex>();

    for (let q = -4; q < 8; q++) {
      for (let r = -4; r < 4 - q; r++) {
        const s = -1 - q - r;
        elMap.set(`${q}${r}`, new Hex(q, r, s));
      }
    }

    for (let q = -7; q < 4; q++) {
      for (let r = -4 - q; r < 4; r++) {
        const s = -1 - q - r;
        elMap.set(`${q}${r}`, new Hex(q, r, s));
      }
    }

    for (const hex of elMap.values()) {
      this.viewbox[0] = Math.min(this.viewbox[0], hex.x);
      this.viewbox[1] = Math.min(this.viewbox[1], hex.y);
      this.viewbox[2] = Math.max(this.viewbox[2], hex.x);
      this.viewbox[3] = Math.max(this.viewbox[3], hex.y);
    }

    this.viewbox[0] -= size + 4;
    this.viewbox[1] -= size;
    this.viewbox[2] += size /2;
    this.viewbox[3] += size / 2; // Just enough for annotations

    this.width = this.viewbox[2] - this.viewbox[0] + size;
    this.height = this.viewbox[3] - this.viewbox[1] + size;

    this.hexagons = Array.from(elMap.values());
  }

  getViewbox(): string {
    return `${this.viewbox[0]} ${this.viewbox[1]} ${this.width} ${this.height}`;
  }

  getHexagons(): Hex[] {
    return this.hexagons;
  }
  getWidth(): number {
    return this.width;
  }
}
