import { MoveType } from "tri-hex-chess";
import { NewMove } from "../class/NewMove.ts";

type Props = {
  move: NewMove;
};

const MoveHighlight = ({ move }: Props) => {
  return (
    <g transform={`translate(${move.x}, ${move.y}) scale(0.5)`}>
      <circle r={25} fill={move.color} opacity={0.6} />

      {move.ref.move_type === MoveType.CapturePromotion && (
        <circle r={25} fill="transparent" strokeWidth={2} stroke="red" />
      )}
    </g>
  );
};

export default MoveHighlight;
