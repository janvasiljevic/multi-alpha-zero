import { Color } from "tri-hex-chess";
import { get3PointsCircle, type PlayerMaterial } from "../class/utils.ts";
import type { Players } from "./Board.tsx";
import { useEffect, useState } from "react";

type Props = {
  material: PlayerMaterial;
  rotation: number;
  currentTurn: Color;
  myColor: Color | null;
  players: Players;
  hideMaterial: boolean;
};

const points = get3PointsCircle(480);
const arr = [Color.White, Color.Gray, Color.Black];

const MaterialCounter = ({
  material,
  rotation,
  currentTurn,
  players,
  myColor,
  hideMaterial
}: Props) => {
  const [waitingText, setWaitingText] = useState("Waiting");

  useEffect(() => {
    const interval = setInterval(() => {
      setWaitingText((prev) => {
        if (prev.endsWith("...")) {
          return "Waiting";
        }
        return prev + ".";
      });
    }, 300);
    return () => clearInterval(interval);
  }, []);

  return points.map(([x, y], i) => {
    const hasPlayer = !!players[arr[i]];

    return (
      <g transform={`translate(${x}, ${y})`} key={i}>
        <text
          x="0"
          y={0}
          fill="black"
          fontSize="20"
          textAnchor="middle"
          dominantBaseline="central"
          transform={`rotate(${-rotation})`}
          fontFamily="IBM Plex Mono"
          style={{ userSelect: "none" }}
        >
          {hasPlayer ? players[arr[i]]?.username : waitingText}
        </text>

        {/* Online indicator – now respects rotation */}
        {hasPlayer && myColor !== arr[i] && (
          <g transform={`rotate(${-rotation})`}>
            <circle
              cx={0}
              cy={30}
              r={6}
              fill={players[arr[i]]?.isConnectedToGame ? "limegreen" : "gray"}
              stroke="black"
              strokeWidth={1}
            />
          </g>
        )}


        {!hideMaterial && <text
          x="0"
          y={currentTurn === i ? 30 : -30}
          fill="black"
          fontSize="18"
          textAnchor="middle"
          dominantBaseline="central"
          transform={`rotate(${-rotation})`}
          fontFamily="IBM Plex Mono"
          style={{ userSelect: "none" }}
        >
          {`${material[arr[i]]} pik`}
        </text>}
      </g>
    );
  });
};

export default MaterialCounter;
