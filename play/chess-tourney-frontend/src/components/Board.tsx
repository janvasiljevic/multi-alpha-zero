import {
  type RefObject,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import {
  Color,
  PatternHighlight,
  Piece,
  TriHexChessWrapper,
} from "../../libs/wasm";
import { CheckMetadata } from "../class/CheckMetadata.ts";
import { NewPiece } from "../class/NewPiece.ts";
import { DndContext, DragOverlay } from "@dnd-kit/core";
import { NewMove } from "../class/NewMove.ts";
import { snapCenterToCursor } from "@dnd-kit/modifiers";
import PromotionMenu from "./PromotionMenu.tsx";
import PieceElement from "./PieceElement.tsx";
import HexagonOutline from "./HexagonOutline.tsx";
import { Board } from "../class/Board.ts";
import useResizeObserver from "@react-hook/resize-observer";
import HexagonComponent from "./HexagonComponent.tsx";
import CheckHighlight from "./CheckHighlight.tsx";
import MaterialCounter from "./MaterialCounter.tsx";
import MoveHighlight from "./MoveHighlight.tsx";
import { getPieces, type PlayerMaterial, SIZE } from "../class/utils.ts";
import TurnIndicator from "./TurnIndicator.tsx";
import type { GameStatus, Move, Player, PlayerUpdate } from "../api/model";
import LastMoveHighlight from "./LastMoveHighlight.tsx";
import PatternHighlightComponent from "./PatternHighlightComponent.tsx";

type InternalPlayer = Player & {
  isMe: boolean;
};

export type Players = {
  [Color.White]: InternalPlayer | null;
  [Color.Gray]: InternalPlayer | null;
  [Color.Black]: InternalPlayer | null;
};

export const playerStatusToPlayers = (
  playerStatus: PlayerUpdate,
  myId: number,
): Players => {
  const players: Players = {
    [Color.White]: playerStatus.white
      ? {
          ...playerStatus.white,
          isMe: playerStatus.white.id === myId,
        }
      : null,
    [Color.Gray]: playerStatus.grey
      ? {
          ...playerStatus.grey,
          isMe: playerStatus.grey.id === myId,
        }
      : null,
    [Color.Black]: playerStatus.black
      ? {
          ...playerStatus.black,
          isMe: playerStatus.black.id === myId,
        }
      : null,
  };

  return players;
};

type Props = {
  fenOutside: string | null;
  players: Players | null;
  rotation: number;
  color: null | Color;
  gameStatus: GameStatus;
  onMovePlayed: (move: NewMove, promotionPiece: Piece | null) => void;
  highlightedMoves: Move[];
  hideMaterial: boolean;
  showPatterns?: boolean;
};

const board = new Board(SIZE);

export const useSize = (target: RefObject<SVGSVGElement> | null) => {
  const [size, setSize] = useState<DOMRect | null>(null);

  useLayoutEffect(() => {
    if (!target) return;
    if (!target.current) return;

    setSize(target.current.getBoundingClientRect());
  }, [target]);

  // Where the magic happens
  useResizeObserver(target, (entry) => setSize(entry.contentRect));
  return size;
};

const BoardComponent = ({
  fenOutside,
  players,
  rotation,
  color,
  gameStatus,
  onMovePlayed,
  highlightedMoves,
  hideMaterial,
  showPatterns,
}: Props) => {
  const state = useRef<TriHexChessWrapper | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  // @ts-ignore
  const size = useSize(svgRef);

  const [currentTurn, setCurrentTurn] = useState<Color>(Color.White);
  const [pieces, setPieces] = useState<NewPiece[]>([]);
  const [checks, setChecks] = useState<CheckMetadata[]>([]);
  const [material, setMaterial] = useState<PlayerMaterial>({
    [Color.White]: 0,
    [Color.Gray]: 0,
    [Color.Black]: 0,
  });

  const [draggingPiece, setDraggingPiece] = useState<NewPiece | null>(null);
  const [promotionMove, setPromotionMove] = useState<NewMove | null>(null);
  const [highlighted, setHighlighted] = useState<NewMove[]>([]);

  const [movementPatternHighlights, setMovementPatternHighlights] = useState<
    PatternHighlight[]
  >([]);

  useEffect(() => {
    // Cleanup
    return () => {
      movementPatternHighlights.forEach((highlight) => {
        highlight.free();
      });
    };
  }, [movementPatternHighlights]);

  useEffect(() => {
    if (!fenOutside) return;

    if (state.current) {
      // Do not update if the FEN is the same
      if (state.current.getFen() === fenOutside) {
        return;
      } else {
        state.current.setFen(fenOutside);
        updateFromGameState();
      }
    } else {
      state.current = TriHexChessWrapper.new(fenOutside);
      updateFromGameState();
    }
  }, [fenOutside]);

  const updateFromGameState = () => {
    if (!state.current) return;

    const gameState = state.current.getGameState();

    // setCurrentTurn(gameState.turn);
    setChecks(
      state.current.getCheckMetadata().map((check) => new CheckMetadata(check)),
    );
    setMaterial((p) => {
      if (!state.current) return p;
      const material = state.current.getMaterial();
      return {
        [Color.White]: material.white,
        [Color.Gray]: material.gray,
        [Color.Black]: material.black,
      };
    });
    // setFen(state.current.getFen());
    setCurrentTurn(gameState.turn);
    setPieces(getPieces(state.current, rotation));
  };

  const promotionCallback = (move: NewMove | null, piece: Piece | null) => {
    setPromotionMove(null);

    if (!move || !state.current) return;

    if (!move || !piece) {
      throw new Error("Assert: move and piece should not be null");
    }

    state.current.commitMove(move.ref, piece, true);

    updateFromGameState();

    onMovePlayed(move, piece);
  };

  // FIXME: This is a little janky :)
  const s = size ? Math.min((size.width / board.getWidth()) * 70 + 8, 70) : 70;

  return (
    <DndContext
      onDragStart={(e) => {
        if (!state.current || gameStatus != "InProgress") return;

        const piece: NewPiece = e.active.data.current as NewPiece;

        if (piece.color !== color && color !== currentTurn) return;

        setDraggingPiece(piece);

        setHighlighted(
          state.current
            .queryMoves(piece.ref!.coordinates)
            .map((move) => new NewMove(move, rotation)),
        );
      }}
      onDragEnd={(e) => {
        setDraggingPiece(null); // Stop the dragging animation
        setHighlighted([]); // Clear the highlighted moves

        const svgElement = svgRef.current;

        if (!svgElement || !state.current) {
          throw new Error(
            "Assert: svgElement|state.current should not be null",
          );
        }

        const point = svgElement.createSVGPoint();
        const event = e.activatorEvent as PointerEvent;
        const ctm = svgElement.getScreenCTM();

        if (!ctm) return;

        point.x = event.clientX;
        point.y = event.clientY;

        const transformedPoint = point.matrixTransform(
          svgElement.getScreenCTM()?.inverse(),
        );

        const scaleX = 1 / ctm.a; // Horizontal scale factor (m11 of the 4x4 matrix)
        const scaleY = 1 / ctm.d; // Vertical scale factor (m22 of the 4x4 matrix)

        const f_x_1 = transformedPoint.x + e.delta.x * scaleX;
        const f_y_1 = transformedPoint.y + e.delta.y * scaleY;

        let move: NewMove | null = null;

        for (const hex of highlighted) {
          const d = Math.sqrt(
            Math.pow(f_x_1 - hex.x, 2) + Math.pow(f_y_1 - hex.y, 2),
          );

          if (d < 45) {
            move = hex;
            break;
          }
        }

        if (!move) return;

        if (move.isPromotion()) {
          setPromotionMove(move);

          return;
        }

        state.current.commitMove(move.ref, undefined, true);

        updateFromGameState();

        onMovePlayed(move, null);

        // if (move.isCapture()) playCaptureSound();
        // else playMoveSound();
      }}
    >
      <svg
        viewBox={board.getViewbox()}
        width="95%"
        style={{ maxHeight: "90vh" }}
        height={"100%"}
        ref={svgRef}
      >
        <g
          transform={`rotate(${rotation})`}
          style={{ opacity: fenOutside ? 1 : 0.0, transition: "opacity 0.5s" }}
        >
          {/* Show all hexagons (chessboard background) and algebraic notation at the side */}
          {board.getHexagons().map((hex, index) => (
            <HexagonOutline
              key={index}
              hex={hex}
              color="black"
              strokeWidth="6"
            />
          ))}

          {/* Show all hexagons (chessboard background) and algebraic notation at the side */}
          {board.getHexagons().map((hex, index) => (
            <HexagonComponent key={index} hex={hex} rotation={rotation} />
          ))}

          {/* Show all present checks */}
          {checks.map((check, index) => (
            <CheckHighlight check={check} key={index} />
          ))}

          {players && (
            <MaterialCounter
              hideMaterial={hideMaterial}
              material={material}
              rotation={rotation}
              currentTurn={color !== null ? color : Color.White}
              players={players}
              myColor={color}
            />
          )}

          <TurnIndicator turn={currentTurn} myColor={color} />
        </g>

        {highlightedMoves.map((move, index) => (
          <LastMoveHighlight
            from={move.from}
            to={move.to}
            isOldest={index == 0 && highlightedMoves.length > 1}
            rotation={rotation}
            key={index}
          />
        ))}

        {/* Show all chess pieces */}
        {pieces.map((piece) => (
          <PieceElement
            key={piece.getUniqueID()}
            piece={piece}
            rotation={rotation}
            isInteractive={piece.color === color}
            onMouseEnter={(piece: NewPiece) => {
              if (!state.current) return;
              if (!piece.ref) {
                setMovementPatternHighlights([]);
                return;
              }

              setMovementPatternHighlights(
                state.current.getPieceMovementPatterns(
                  piece.ref!.coordinates,
                ) || [],
              );
            }}
            onMouseLeave={(_: NewPiece) => {
              setMovementPatternHighlights([]);
            }}
          />
        ))}

        {/* Possible piece moves when a piece is active (hovering for placement) */}
        {highlighted.map((hex, index) => (
          <MoveHighlight key={index} move={hex} />
        ))}

        {/* Movement pattern highlights */}
        {showPatterns &&
          movementPatternHighlights.map((highlight, index) => (
            <PatternHighlightComponent
              key={index}
              highlight={highlight}
              rotation={rotation}
            />
          ))}

        {/* Show 4 promotion hexagons */}
        {promotionMove && (
          <PromotionMenu move={promotionMove} callback={promotionCallback} />
        )}
      </svg>

      {draggingPiece && (
        <DragOverlay modifiers={[snapCenterToCursor]}>
          <svg>
            <image
              href={`/pieces/${draggingPiece.getSvgString()}.svg`}
              type="image/svg+xml"
              width={s}
              height={s}
              transform={`translate(${s / 8}, ${s / 8})`}
              style={{
                filter: "grayscale(1)",
              }}
            />
          </svg>
        </DragOverlay>
      )}
    </DndContext>
  );
};

export default BoardComponent;
