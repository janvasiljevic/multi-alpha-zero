import { useEffect } from "react";
import { NewMove } from "../class/NewMove.ts";
import { Piece } from "tri-hex-chess";

type Props = {
  move: NewMove;
  callback: (pieceType: NewMove | null, promotionType: Piece | null) => void;
};

const PromotionMenu = ({ move, callback }: Props) => {
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      const target = e.target as Element;
      const type = target.getAttribute("data-type");

      if (!type) return;

      const typeNum: Piece = parseInt(type);

      if (typeNum) {
        callback(move, typeNum);
      } else {
        callback(null, null);
      }
    };

    document.addEventListener("click", handleClick);
    return () => document.removeEventListener("click", handleClick);
  }, [move, callback]);

  return (
    <g style={{ isolation: "isolate" }}>
      <rect x="-1000" y="-1000" width="2000" height="2000" fill="transparent" />

      <rect x="-3000" y="-3000" width="3000" height="3000" fill="transparent" />
      {move.getPromotionSquares().map((sq, i) => (
        <g key={i} transform={`translate(${sq.x}, ${sq.y})`}>
          <polygon
            points="50,0 25,-43.5 -25,-43.5 -50,0 -25,43.5 25,43.5"
            fill="black"
            opacity={0.6}
            style={{ cursor: "pointer" }}
            data-type={sq.type}
          />
          <image
            href={`/pieces/w${sq.text}.svg`}
            data-type={sq.type}
            type="image/svg+xml"
            width={70}
            height={70}
            transform="translate(-35, -35)"
            style={{ cursor: "pointer" }}
          />
        </g>
      ))}
    </g>
  );
};

export default PromotionMenu;
