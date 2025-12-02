import { Color } from "tri-hex-chess";
import { COLOR_ARR, get3PointsCircle } from "../class/utils.ts";

type Props = {
  turn: Color;
  myColor: Color | null;
};

const points = get3PointsCircle(430);

const TurnIndicator = ({ turn, myColor }: Props) => {
  return points.map(([x, y], i) => {
    const color = COLOR_ARR[i];

    // Calculate triangle points with the tip pointing inward
    const triangleSize = 10;
    const angleToCenter = Math.atan2(-y, -x);
    const trianglePoints = [
      // Base points
      [
        triangleSize * Math.cos(angleToCenter + Math.PI / 2),
        triangleSize * Math.sin(angleToCenter + Math.PI / 2),
      ],
      [
        triangleSize * Math.cos(angleToCenter - Math.PI / 2),
        triangleSize * Math.sin(angleToCenter - Math.PI / 2),
      ],
      // Tip point (inward)
      [
        triangleSize * Math.cos(angleToCenter),
        triangleSize * Math.sin(angleToCenter),
      ],
    ];

    const pointsStr = trianglePoints.map(([px, py]) => `${px},${py}`).join(" ");

    return (
      <g transform={`translate(${x}, ${y})`} key={i}>
        <polygon
          points={pointsStr}
          fill={turn === i ? color : "white"}
          stroke={turn === myColor ? "red" : "black"}
          strokeWidth={turn === i ? 20 : 0}
          style={{ filter: "brightness(0.8)" }}
        />
      </g>
    );
  });
};

export default TurnIndicator;
