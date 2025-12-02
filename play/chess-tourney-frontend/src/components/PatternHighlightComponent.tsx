import { PatternHighlight } from "tri-hex-chess";
import { getXYFromCoordWithRotation, SIZE } from "../class/utils.ts";

type Props = {
  highlight: PatternHighlight;
  rotation: number;
};

const PatternHighlightComponent = ({ highlight, rotation }: Props) => {
  const { x, y } = getXYFromCoordWithRotation(highlight.to, SIZE, rotation);
  return (
    <g transform={`translate(${x}, ${y}) scale(0.5)`}>
      <circle r={25} fill="green" opacity={0.6} />

      {/*{move.ref.move_type === MoveType.CapturePromotion && (*/}
      {/*  <circle r={25} fill="transparent" strokeWidth={2} stroke="red" />*/}
      {/*)}*/}
    </g>
  );
};

export default PatternHighlightComponent;
