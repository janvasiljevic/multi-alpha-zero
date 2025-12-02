import { memo } from "react";
import { NewPiece } from "../class/NewPiece";
import { Color } from "tri-hex-chess";

type Props = {
  piece: NewPiece;
  rotation: number; // Pass rotation so we can compare it in memo
  onClick?: () => void;
};

const PieceElementNonInteractive = memo(
  ({ piece, onClick }: Props) => {
    return (
      <>
        <g
          transform={`translate(${piece.x}, ${piece.y})`}
          onClick={onClick}
          style={{
            touchAction: "none",
            position: "absolute",
            transformOrigin: "center",
            cursor: "pointer",
          }}
        >
          <image
            style={{
              filter: "grayscale(1)" + (piece.color === Color.Gray ? " brightness(1.4)" : ""),
            }}
            href={`/pieces/${piece.getPlayerString().toLowerCase()}${piece.getPieceString().toUpperCase()}.svg`}
            type="image/svg+xml"
            width={70}
            height={70}
            transform="translate(-35, -35)"
          />

          <polygon
            points="50,0 25,-43.5 -25,-43.5 -50,0 -25,43.5 25,43.5"
            fill="transparent"
            className="dont-export-svg"
            strokeWidth="2"
            style={{ cursor: "pointer", outline: "none" }}
          />
        </g>
      </>
    );
  },
  (prev, next) => {
    return prev.piece.getUniqueID() === next.piece.getUniqueID() && prev.rotation === next.rotation && prev.onClick === next.onClick;
  },
);

export default PieceElementNonInteractive;
