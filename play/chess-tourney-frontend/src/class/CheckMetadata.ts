import { AttackInfo } from "tri-hex-chess";
import { getX, getY, SIZE } from "./utils.ts";

export class CheckMetadata {
  readonly ref: AttackInfo;
  readonly kingX: number;
  readonly kingY: number;
  readonly attackX: number;
  readonly attackY: number;
  readonly startX: number;
  readonly startY: number;
  readonly endX: number;
  readonly endY: number;

  constructor(ref: AttackInfo) {
    this.ref = ref;
    this.kingX = getX(ref.king.q, SIZE);
    this.kingY = getY(ref.king.q, ref.king.r, SIZE);
    this.attackX = getX(ref.attack.q, SIZE);
    this.attackY = getY(ref.attack.q, ref.attack.r, SIZE);

    const dx = this.kingX - this.attackX;
    const dy = this.kingY - this.attackY;
    const len = Math.sqrt(dx * dx + dy * dy);
    const unitDx = dx / len;
    const unitDy = dy / len;

    // Trim 30 pixels from start and end
    this.startX = this.attackX + unitDx * 40;
    this.startY = this.attackY + unitDy * 40;
    this.endX = this.kingX - unitDx * 40;
    this.endY = this.kingY - unitDy * 40;
  }
}
