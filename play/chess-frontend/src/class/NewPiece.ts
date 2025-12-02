import { ChessPiece, Color, Piece } from "tri-hex-chess";
import { getX, getY, rotate } from "../lib/utils/utils_xy";
import { SIZE } from "../lib/constants.ts";
import { libColorToString, libPieceToString } from "../lib/utils/utils.ts";
import { Hex } from "./Hex.ts";

type PieceConstructor =
  | {
      type: "ref";
      ref: ChessPiece;
      rotation?: number;
    }
  | {
      type: "manual";
      i: number;
      piece: Piece;
      color: Color;
      hex: Hex;
      rotation?: number;
    };

export class NewPiece {
  ref?: ChessPiece;

  i!: number;
  type!: Piece;
  color!: Color;
  uniqueID!: string;

  unmodifiedX!: number;
  unmodifiedY!: number;
  x: number = 0;
  y: number = 0;
  angle: number = 0;

  constructor(data: PieceConstructor) {
    if (data.type === "ref") {
      this.fromRef(data);
    } else {
      this.fromManual(data);
    }
  }

  private fromRef({ ref, rotation = 0 }: Extract<PieceConstructor, { type: "ref" }>) {
    this.ref = ref;
    this.i = ref.coordinates.i;
    this.type = ref.piece;
    this.color = ref.player;
    this.uniqueID = `${this.i}/${this.color}/${this.type}`;

    this.unmodifiedX = getX(ref.coordinates.q, SIZE);
    this.unmodifiedY = getY(ref.coordinates.q, ref.coordinates.r, SIZE);

    this.rotateToAngle(rotation);
  }

  private fromManual({ i, piece, color, hex, rotation = 0 }: Extract<PieceConstructor, { type: "manual" }>) {
    this.i = i;
    this.type = piece;
    this.color = color;
    this.uniqueID = `${this.i}/${this.color}/${this.type}`;

    this.unmodifiedX = hex.x;
    this.unmodifiedY = hex.y;

    this.rotateToAngle(rotation);
  }

  rotateToAngle(newAngle: number) {
    this.angle = newAngle;

    const rot = rotate(this.unmodifiedX, this.unmodifiedY, this.angle);

    this.x = rot.x;
    this.y = rot.y;
  }

  getUniqueID(): string {
    return this.uniqueID;
  }

  getPlayerString(): string {
    return libColorToString(this.color);
  }

  getPieceString(): string {
    return libPieceToString(this.type);
  }

  getSvgString(): string {
    return `${this.getPlayerString().toLowerCase()}${this.getPieceString().toUpperCase()}`;
  }

  toInternalPiece(): ChessPiece {
    return new ChessPiece(this.type, this.color, this.i);
  }
}
