import { type ChessMoveWrapperDto } from "../api/def/model";
import { getX, getY } from "../lib/utils/utils_xy.ts";
import { SIZE } from "../lib/constants.ts";
import Arrow from "./Arrow"; // Import the new component

type Props = {
  moves: ChessMoveWrapperDto[] | null;
  topN: number;
};

const Arrows = ({ moves, topN }: Props) => {
  if (!moves) {
    return null; // Return null instead of <></>
  }

  // Slice is slightly more efficient than filter for taking the top N items
  const topMoves = moves.slice(0, topN + 1);

  return (
    <>
      {/* No more <defs> needed! */}
      {topMoves.map((move, index) => {
        // Get original center-to-center coordinates
        let start_x = getX(move.inner.from.q, SIZE);
        let start_y = getY(move.inner.from.q, move.inner.from.r, SIZE);

        let end_x = getX(move.inner.to.q, SIZE);
        let end_y = getY(move.inner.to.q, move.inner.to.r, SIZE);

        // Calculate unit vector for trimming
        const dx = end_x - start_x;
        const dy = end_y - start_y;
        const len = Math.sqrt(dx * dx + dy * dy);
        const unitDx = len > 0 ? dx / len : 0;
        const unitDy = len > 0 ? dy / len : 0;

        const trim = 10;

        // Apply your 30px trim from the start and end of the visual arrow
        const trimmedStart = {
          x: start_x + unitDx * trim,
          y: start_y + unitDy * trim,
        };
        const trimmedEnd = {
          x: end_x - unitDx * trim,
          y: end_y - unitDy * trim,
        };

        const opacity = 0.6 + 0.4 * (1 - index / topMoves.length);

        return (
          <Arrow
            key={index}
            from={trimmedStart}
            to={trimmedEnd}
            text={`${move.confidence.toFixed(2)} (${move.prior.toFixed(2)})`}
            color={index === 0 ? "cyan" : "yellow"}
            strokeWidth={10}
            opacity={opacity}
          />
        );
      })}
    </>
  );
};
export default Arrows;
