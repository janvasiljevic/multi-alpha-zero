import { Coordinates } from "tri-hex-chess";
import { SIZE } from "../lib/constants";
import { getXYFromCoordWithRotation } from "../lib/utils/utils_xy";

type Props = {
  from: Coordinates;
  to: Coordinates;
  isOldest: boolean;
  rotation: number;
};

const LastMoveHighlight = ({ from, to, isOldest, rotation }: Props) => {
  const { x: fromX, y: fromY } = getXYFromCoordWithRotation(from, SIZE, rotation);
  const { x: toX, y: toY } = getXYFromCoordWithRotation(to, SIZE, rotation);

  return (
    <>
      <g transform={`translate(${fromX}, ${fromY})`}>
        <polygon points="50,0 25,-43.5 -25,-43.5 -50,0 -25,43.5 25,43.5" fill="yellow" opacity={isOldest ? 0.1 : 0.2} />
      </g>
      <g transform={`translate(${toX}, ${toY})`}>
        <polygon points="50,0 25,-43.5 -25,-43.5 -50,0 -25,43.5 25,43.5" fill="yellow" opacity={isOldest ? 0.1 : 0.2} />
      </g>
    </>
  );
};

export default LastMoveHighlight;
