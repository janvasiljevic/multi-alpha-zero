import type { HexPlayer } from "../api/def/model";
import { useCallback, useState } from "react";

export const getPlayerColor = (p: HexPlayer): string => {
  switch (p) {
    case "P1":
      return "#E74C3C"; // Red
    case "P2":
      return "#3498DB"; // Blue
    case "P3":
      return "#F1C40F"; // Yellow
    default:
      // black
      return "#000000";
      throw new Error(`Unknown player: ${p}`);
  }
};

const players: HexPlayer[] = ["P1", "P2", "P3"];

export const getPlayerColorByIndex = (index: number): string => {
  if (index < 0 || index >= players.length) {
    throw new Error(`Index out of bounds: ${index}`);
  }
  return getPlayerColor(players[index]);
};

const CHESS_COLORS = ["#ffffff", "#2d6327", "#0a0a0a"];

export const getChessPlayerColorByIndex = (index: number): string => {
  if (index < 0 || index > 2) {
    throw new Error(`Index out of bounds: ${index}`);
  }

  return CHESS_COLORS[index];
};

type Dimensions = {
  width: number;
  height: number;
};

type CenteredTreeReturn = [
  Dimensions | undefined,
  { x: number; y: number },
  (containerElem: HTMLElement | null) => void,
];

export const useCenteredTree = (defaultTranslate = { x: 0, y: 0 }): CenteredTreeReturn => {
  const [translate, setTranslate] = useState(defaultTranslate);
  const [dimensions, setDimensions] = useState<Dimensions | undefined>(undefined);

  const containerRef = useCallback((containerElem: HTMLElement | null) => {
    if (containerElem !== null) {
      const { width, height } = containerElem.getBoundingClientRect();
      setDimensions({ width, height });
      setTranslate({ x: width / 2, y: height / 2 });
    }
  }, []);

  return [dimensions, translate, containerRef];
};
