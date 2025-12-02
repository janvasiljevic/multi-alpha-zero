import { Hex } from "../class/Hex.ts";
import { SIZE } from "../lib/constants.ts";

type Props = {
  hex: Hex;
  rotation: number;
  disableNumbers?: boolean;
  highlight?: boolean;
  onClick?: () => void;
};

const HexagonComponent = ({ hex, rotation, disableNumbers, highlight, onClick }: Props) => {
  return (
    <>
      <g transform={`translate(${hex.x}, ${hex.y})`}>
        <polygon
          points="50,0 25,-43.5 -25,-43.5 -50,0 -25,43.5 25,43.5"
          fill={hex.color}
          stroke="black"
          strokeWidth="0"
          onClick={onClick}
        />

        {highlight && (
          <polygon
            points="50,0 25,-43.5 -25,-43.5 -50,0 -25,43.5 25,43.5"
            fill="red"
            stroke="red"
            strokeWidth="4"
          />
        )}
      </g>

      {hex.letter && !disableNumbers && (
        <text
          textAnchor="middle"
          dominantBaseline="central"
          fontSize="12"
          fill="black"
          fontWeight={900}
          transform={`translate(${hex.x}, ${hex.y + SIZE + 5}) rotate(${-rotation})`}
          style={{ userSelect: "none" }}
        >
          {hex.letter}
        </text>
      )}

      {hex.number && !disableNumbers && (
        <text
          textAnchor="middle"
          dominantBaseline="central"
          fontSize="12"
          fill="black"
          fontWeight={900}
          transform={`translate(${hex.x + SIZE + 5}, ${hex.y - SIZE / 2 - 5}) rotate(${-rotation})`}
          style={{ userSelect: "none" }}
        >
          {hex.number}
        </text>
      )}
    </>
  );
};

export default HexagonComponent;
