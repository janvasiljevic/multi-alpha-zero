import type { Coordinates, TriHexChessWrapper } from "../../libs/wasm";
import { Color, Piece } from "tri-hex-chess";
import { NewPiece } from "./NewPiece.ts";
import type { Position } from "../api/model";

export const COLORS = {
  W: "#FFFFFF",
  G: "#C5C5C5",
  B: "#8E8E8E",
};

export const COLOR_ARR = [COLORS.W, COLORS.G, COLORS.B];

export const SIZE = 50;

export const A_FILE: AlgebraicFile[] = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o"];
export const A_RANK: AlgebraicRank[] = [
  "1",
  "2",
  "3",
  "4",
  "5",
  "6",
  "7",
  "8",
  "9",
  "10",
  "11",
  "12",
  "13",
  "14",
  "15",
];


export type Player = "W" | "G" | "B";

export type PlayerMaterial = {
  [key in Color]: number;
};

export type AlgebraicFile =
  | "a"
  | "b"
  | "c"
  | "d"
  | "e"
  | "f"
  | "g"
  | "h"
  | "i"
  | "j"
  | "k"
  | "l"
  | "m"
  | "n"
  | "o";
export type AlgebraicRank =
  | "1"
  | "2"
  | "3"
  | "4"
  | "5"
  | "6"
  | "7"
  | "8"
  | "9"
  | "10"
  | "11"
  | "12"
  | "13"
  | "14"
  | "15";

export type AlgebraicNotation = `${AlgebraicFile}${AlgebraicRank}`;

export const getX = (q: number, size: number) => size * (3 / 2) * q + size / 2;

export const getY = (q: number, r: number, size: number) =>
  size * ((-Math.sqrt(3) / 2) * q + -Math.sqrt(3) * r) -
  (Math.sqrt(3) * size) / 2;

export const getXFromCoord = (coord: Coordinates, size: number) =>
  getX(coord.q, size);
export const getYFromCoord = (coord: Coordinates, size: number) =>
  getY(coord.q, coord.r, size);

export const rotate = (x: number, y: number, angle: number) => {
  const rad = (angle * Math.PI) / 180;
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);

  return {
    x: x * cos - y * sin,
    y: x * sin + y * cos,
  };
};

export const getXYFromCoordWithRotation = (
  coord: Coordinates,
  size: number,
  rotation: number,
) => {
  const { x, y } = rotate(
    getX(coord.q, size),
    getY(coord.q, coord.r, size),
    rotation,
  );
  return { x, y };
};

export const getXYFromPosWithRotation = (
  coord: Position,
  size: number,
  rotation: number,
) => {
  const { x, y } = rotate(
    getX(coord.q, size),
    getY(coord.q, coord.r, size),
    rotation,
  );
  return { x, y };
};

// % != modulo in javascript, it's the remainder operator
export function mod(n: number, m: number) {
  return ((n % m) + m) % m;
}

export function libColorToString(color: Color): Player {
  return color === Color.White ? "W" : color === Color.Gray ? "G" : "B";
}

export function libColorToHumanString(color: Color): string {
  return color === Color.White ? "White" : color === Color.Gray ? "Gray" : "Black";
}

export function libPieceToString(piece: Piece) {
  switch (piece) {
    case Piece.Pawn:
      return "p";
    case Piece.Knight:
      return "n";
    case Piece.Bishop:
      return "b";
    case Piece.Rook:
      return "r";
    case Piece.Queen:
      return "q";
    case Piece.King:
      return "k";
  }
}

export function qrsToAlgebraic(g_q: number, g_r: number): AlgebraicNotation {
  return `${A_FILE[g_q + 7]}${A_RANK[g_r + 7]}`;
}

export function algebraicToQRS(algebraic: AlgebraicNotation): [number, number] {
  const file = algebraic[0];
  const rank = algebraic.slice(1);

  // check if file in A_FILE
  if (!A_FILE.includes(file as AlgebraicFile)) {
    throw new Error(`Invalid file: ${file}`);
  }

  // check if rank in A_RANK
  if (!A_RANK.includes(rank as AlgebraicRank)) {
    throw new Error(`Invalid rank: ${rank}`);
  }

  return [
    A_FILE.indexOf(file as AlgebraicFile) - 7,
    A_RANK.indexOf(rank as AlgebraicRank) - 7,
  ];
}

export const get3PointsCircle = (radius: number) => {
  const points = [];

  for (let angle = 0; angle < Math.PI * 2; angle += (Math.PI * 2) / 3) {
    const x = radius * Math.cos(angle + Math.PI / 2);
    const y = radius * Math.sin(-angle + Math.PI / 2);

    points.push([-x, y]);
  }

  return points;
};

export const getPieces = (
  game: TriHexChessWrapper,
  rotation: number,
): NewPiece[] => {
  const pieces: NewPiece[] = [];

  for (const piece of game.getPieces()) {
    pieces.push(new NewPiece({ type: "ref", ref: piece, rotation }));
  }

  return pieces;
};

