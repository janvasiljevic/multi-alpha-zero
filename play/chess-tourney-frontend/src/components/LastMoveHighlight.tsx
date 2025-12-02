import { getXYFromPosWithRotation, SIZE } from "../class/utils.ts";
import type { Position } from "../api/model";

type Props = {
  from: Position;
  to: Position;
  isOldest: boolean;
  rotation: number;
};

const LastMoveHighlight = ({ from, to, isOldest, rotation }: Props) => {
  const { x: fromX, y: fromY } = getXYFromPosWithRotation(from, SIZE, rotation);
  const { x: toX, y: toY } = getXYFromPosWithRotation(to, SIZE, rotation);

  return (
    <>
      <g transform={`translate(${fromX}, ${fromY})`}>
        <polygon
          points="50,0 25,-43.5 -25,-43.5 -50,0 -25,43.5 25,43.5"
          fill="yellow"
          opacity={isOldest ? 0.08 : 0.25}
        />
      </g>
      <g transform={`translate(${toX}, ${toY})`}>
        <polygon
          points="50,0 25,-43.5 -25,-43.5 -50,0 -25,43.5 25,43.5"
          fill="yellow"
          opacity={isOldest ? 0.08 : 0.25}
        />
      </g>
    </>
  );
};

export default LastMoveHighlight;
