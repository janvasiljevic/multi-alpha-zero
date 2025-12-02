import { Color } from "tri-hex-chess";
import { PlayerMaterial } from "../lib/types";
import { get3PointsCircle } from "../utils";

type Props = {
  material: PlayerMaterial;
  rotation: number;
  currentTurn: Color;
};

const points = get3PointsCircle(480);
const arr = [Color.White, Color.Gray, Color.Black];

const MaterialCounter = ({ material, rotation, currentTurn }: Props) => {
  return points.map(([x, y], i) => {
    return (
      <g transform={`translate(${x}, ${y})`} key={i}>
        <text
          x="0"
          y={currentTurn === i ? 30 : -30}
          fill="black"
          fontSize="20"
          textAnchor="middle"
          dominantBaseline="central"
          transform={`rotate(${-rotation})`}
          fontFamily="IBM Plex Mono"
          style={{ userSelect: "none" }}
        >
          {`${material[arr[i]]} pik`}
        </text>
      </g>
    );
  });
};

export default MaterialCounter;
