import { useDraggable } from "@dnd-kit/core";
import { memo } from "react";
import { NewPiece } from "../class/NewPiece";

type Props = {
  piece: NewPiece;
  rotation: number;
  isInteractive: boolean;
  onMouseEnter?: (piece: NewPiece) => void;
  onMouseLeave?: (piece: NewPiece) => void;
};

const PieceElement = memo(
  ({ piece, isInteractive, onMouseEnter, onMouseLeave }: Props) => {
    const { attributes, listeners, setNodeRef, transform, isDragging } =
      useDraggable({
        id: piece.getUniqueID(),
        data: piece,
      });

    const dragProps = isInteractive ? { ...attributes, ...listeners } : {};
    const nodeRef = isInteractive ? setNodeRef : undefined;

    return (
      <>
        <g
          transform={
            transform
              ? `translate(${transform.x + piece.x}, ${transform.y + piece.y})`
              : `translate(${piece.x}, ${piece.y})`
          }
          style={{
            touchAction: "none",
            opacity: isDragging ? 0 : 1,
            transition: transform ? "none" : "transform 0.4s",
            position: "absolute",
            zIndex: isDragging ? 10 : 0,
            transformOrigin: "center",
            cursor: isInteractive ? "grab" : "default",
          }}
          // ref={(e) => setNodeRef(e as unknown as HTMLElement)}
          ref={nodeRef as any}
        >
          <image
            // style={{
            //   filter:
            //     "grayscale(1)" +
            //     (piece.color === Color.Gray ? " brightness(1.4)" : ""),
            // }}
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
            style={{
              cursor: isInteractive ? "grab" : "default",
              outline: "none",
            }}
            onMouseEnter={() => {
              if (onMouseEnter && !isDragging) onMouseEnter(piece);
            }}
            onMouseLeave={() => {
              if (onMouseLeave) onMouseLeave(piece);
            }}
            {...dragProps}
          />
        </g>

        {isInteractive && isDragging && (
          <g transform={`translate(${piece.x}, ${piece.y})`} opacity={0.3}>
            <image
              // style={{
              //   filter: "grayscale(1)",
              // }}
              href={`/pieces/${piece.getPlayerString().toLowerCase()}${piece.getPieceString().toUpperCase()}.svg`}
              type="image/svg+xml"
              width={70}
              height={70}
              transform="translate(-35, -35)"
            />
          </g>
        )}
      </>
    );
  },
  (prev, next) => {
    return (
      prev.piece.getUniqueID() === next.piece.getUniqueID() &&
      prev.rotation === next.rotation &&
      prev.isInteractive === next.isInteractive
    );
  },
);

export default PieceElement;
