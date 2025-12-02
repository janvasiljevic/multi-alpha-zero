import { Hex } from "../class/Hex.ts";

type Props = {
  hex: Hex;
  color: string;
  strokeWidth: string;
};

const HexagonOutline = ({ hex, color, strokeWidth }: Props) => {
  return (
    <g transform={`translate(${hex.x}, ${hex.y})`}>
      <polygon
        points="50,0 25,-43.5 -25,-43.5 -50,0 -25,43.5 25,43.5"
        fill={hex.color}
        stroke={color}
        strokeWidth={strokeWidth}
      />
    </g>
  );
};

export default HexagonOutline;
