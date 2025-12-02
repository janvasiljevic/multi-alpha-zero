import { NewPiece } from "../class/NewPiece.ts";
import { Button, Checkbox, CopyButton, Flex, NumberInput, Paper, Select, SimpleGrid, Text } from "@mantine/core";
import { DndContext, DragOverlay } from "@dnd-kit/core";
import HexagonOutline from "./HexagonOutline.tsx";
import HexagonComponent from "./HexagonComponent.tsx";
import PieceElement from "./PieceElement.tsx";
import { snapCenterToCursor } from "@dnd-kit/modifiers";
import { Board } from "../class/Board.ts";
import { SIZE } from "../lib/constants.ts";
import { Castling, Color, Piece, TriHexChessWrapper, UnvalidatedBoard } from "../../libs/wasm";
import { Hex } from "../class/Hex.ts";
import EditorPieceBox from "./EditorPieceBox.tsx";
import PieceElementNonInteractive from "./PieceElementNonInteractive.tsx";
import { useEffect, useRef, useState } from "react";
import { getPieces, useSize } from "../utils.ts";
import { IconCopy } from "@tabler/icons-react";

type Props = {
  initialFen?: string;
  confirmationCallback: (fen: string) => void;
  cancelCallback: () => void;
};

const board = new Board(SIZE);

export type Tool = {
  mode: "move" | "add" | "remove";
  piece?: {
    color: Color;
    type: Piece;
  };
};

type CanCastle = {
  king: boolean;
  queen: boolean;
};

type CastlingComboBoxProps = {
  canCastle: CanCastle;
  index: number;
  setCanCastleForIndex: (index: number, canCastle: CanCastle) => void;
};

const INDEX_TO_NAME = ["W", "G", "B"];

const CastlingComboBox = ({ canCastle, index, setCanCastleForIndex }: CastlingComboBoxProps) => {
  return (
    <Flex gap="sm" justify="center" align="center">
      <Flex direction="column" gap="2px" align="center" justify="center">
        <Flex gap="xs" align="center" justify="center">
          <Text size="xs" fw="500">
            {INDEX_TO_NAME[index]}K
          </Text>
          <Checkbox
            size="xs"
            checked={canCastle.king}
            onChange={(e) => {
              setCanCastleForIndex(index, { ...canCastle, king: e.currentTarget.checked });
            }}
          />
        </Flex>
        <Flex gap="xs" align="center" justify="center">
          <Text size="xs" fw="500">
            {INDEX_TO_NAME[index]}Q
          </Text>
          <Checkbox
            size="xs"
            checked={canCastle.queen}
            onChange={(e) => {
              setCanCastleForIndex(index, { ...canCastle, queen: e.currentTarget.checked });
            }}
          />
        </Flex>
      </Flex>
    </Flex>
  );
};

const ChessEditorView = ({ initialFen, confirmationCallback, cancelCallback }: Props) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const size = useSize(svgRef);

  const [pieces, setPieces] = useState<NewPiece[]>([]);
  const [draggingPiece, setDraggingPiece] = useState<NewPiece | null>(null);
  const [tool, setTool] = useState<Tool>({ mode: "move", piece: undefined });
  const [castlingAvailable, setCastlingAvailable] = useState<CanCastle[]>([
    { king: true, queen: true },
    { king: true, queen: true },
    { king: true, queen: true },
  ]);
  const [activeColor, setActiveColor] = useState<Color>(Color.White);
  const [thirdMoveCount, setThirdMoveCount] = useState<number>(0);
  const [turnCounter, setTurnCounter] = useState<number>(0);
  const [currentFen, setCurrentFen] = useState<string>("");

  const updateCurrentFen = () => {
    const unvalidated_board = new UnvalidatedBoard();

    unvalidated_board.set_pieces(pieces.map((p) => p.toInternalPiece()));
    unvalidated_board.set_castlings(castlingAvailable.map((c) => new Castling(c.king, c.queen)));
    unvalidated_board.set_third_move_count(thirdMoveCount);
    unvalidated_board.set_turn_counter(turnCounter);
    unvalidated_board.set_turn(activeColor);

    try {
      const fen = unvalidated_board.to_fen();
      setCurrentFen(fen);
    } catch (e) {
      setCurrentFen(e);
    }
  };

  useEffect(() => {
    updateCurrentFen();
  }, [pieces, castlingAvailable, activeColor, thirdMoveCount, turnCounter]);

  const confirmPlacement = () => {
    const unvalidated_board = new UnvalidatedBoard();

    unvalidated_board.set_pieces(pieces.map((p) => p.toInternalPiece()));
    unvalidated_board.set_castlings(castlingAvailable.map((c) => new Castling(c.king, c.queen)));
    unvalidated_board.set_third_move_count(thirdMoveCount);
    unvalidated_board.set_turn_counter(turnCounter);
    unvalidated_board.set_turn(activeColor);

    try {
      const fen = unvalidated_board.to_fen();

      confirmationCallback(fen);
    } catch (e) {
      alert(e);
    }
  };

  useEffect(() => {
    if (!initialFen) return;

    const temporary = TriHexChessWrapper.new(initialFen);

    const castling = temporary.getCastlingRights();

    for (let i = 0; i < 3; i++) {
      setCastlingAvailable((prev) => {
        const newState = [...prev];
        newState[i] = {
          king: castling[i].can_castle_king_side,
          queen: castling[i].can_castle_queen_side,
        };
        return newState;
      });
    }

    setTurnCounter(temporary.getGameState().turn_counter);
    setThirdMoveCount(temporary.getGameState().third_move_count);
    setPieces(getPieces(temporary, 0));
    setActiveColor(temporary.getGameState().turn);

    updateCurrentFen();
  }, [initialFen]);

  // FIXME: This is a little janky :)
  const s = size ? Math.min((size.width / board.getWidth()) * 70 + 8, 70) : 70;

  return (
    <Flex w="100vw" h="100vh" p="sm" gap="sm" justify="center" align="center" direction="row" bg="gray.0">
      <Flex direction="column" style={{ flexGrow: 1 }} h="100%" w="100%" align="center" pt="lg" pos="relative" gap="sm">
        <Flex
          gap="md"
          justify="center"
          align="center"
          direction="row"
          style={{ flexGrow: 0, flexShrink: 0 }}
          w="100%"
          h="auto"
        ></Flex>
        <Flex direction="row" gap="sm" style={{ flexGrow: 1, minHeight: 0, position: "relative" }}>
          <DndContext
            onDragStart={(e) => {
              const piece: NewPiece = e.active.data.current as NewPiece;

              setDraggingPiece(piece);
            }}
            onDragEnd={(e) => {
              setDraggingPiece(null); // Stop the dragging animation

              const svgElement = svgRef.current;

              if (!svgElement) {
                throw new Error("Assert: svgElement|state.current should not be null");
              }

              const point = svgElement.createSVGPoint();
              const event = e.activatorEvent as PointerEvent;
              const ctm = svgElement.getScreenCTM();

              if (!ctm) return;

              point.x = event.clientX;
              point.y = event.clientY;

              const transformedPoint = point.matrixTransform(svgElement.getScreenCTM()?.inverse());

              const scaleX = 1 / ctm.a; // Horizontal scale factor (m11 of the 4x4 matrix)
              const scaleY = 1 / ctm.d; // Vertical scale factor (m22 of the 4x4 matrix)

              const f_x_1 = transformedPoint.x + e.delta.x * scaleX;
              const f_y_1 = transformedPoint.y + e.delta.y * scaleY;

              let closestHex: Hex | null = null;
              let minDistance = Infinity;

              for (const hex of board.getHexagons()) {
                const distance = Math.sqrt(Math.pow(f_x_1 - hex.x, 2) + Math.pow(f_y_1 - hex.y, 2));

                if (distance < minDistance && distance < 100) {
                  minDistance = distance;
                  closestHex = hex;
                }
              }

              if (!draggingPiece) return;

              if (closestHex) {
                // copy all other pieces except the dragging one and if there is one on the target square, remove it (capture)
                const newPieces = pieces.filter(
                  (p) => p.getUniqueID() !== draggingPiece.getUniqueID() && p.i !== closestHex!.memIdx,
                );

                const newPiece = new NewPiece({
                  type: "manual",
                  i: closestHex.memIdx,
                  piece: draggingPiece.type,
                  color: draggingPiece.color,
                  hex: closestHex,
                });

                newPieces.push(newPiece);
                setPieces(newPieces);
              } else {
                // delete the piece
                setPieces((prev) => prev.filter((p) => p.getUniqueID() !== draggingPiece.getUniqueID()));
              }
            }}
          >
            <Flex justify="center" align="center" style={{ position: "relative" }}>
              <svg
                viewBox={board.getViewbox()}
                width="95%"
                style={{ maxHeight: "90vh", position: "relative" }}
                height={"100%"}
                ref={svgRef}
              >
                <g>
                  {board.getHexagons().map((hex, index) => (
                    <HexagonOutline key={index} hex={hex} color="black" strokeWidth="6" />
                  ))}

                  {board.getHexagons().map((hex, index) => (
                    <HexagonComponent
                      key={index}
                      hex={hex}
                      rotation={0}
                      onClick={() => {
                        if (tool.mode === "add" && tool.piece) {
                          // Add piece at hex
                          const newPiece = new NewPiece({
                            type: "manual",
                            i: hex.memIdx,
                            piece: tool.piece.type,
                            color: tool.piece.color,
                            hex: hex,
                          });

                          // Remove any existing piece at that location (capture)
                          const newPieces = pieces.filter((p) => p.i !== hex.memIdx);
                          newPieces.push(newPiece);
                          setPieces(newPieces);
                        }
                      }}
                    />
                  ))}
                </g>

                {tool.mode !== "move" &&
                  pieces.map((piece) => (
                    <PieceElementNonInteractive
                      key={piece.getUniqueID()}
                      piece={piece}
                      rotation={0}
                      onClick={() => {
                        if (tool.mode === "remove") {
                          const newPieces = pieces.filter((p) => p.getUniqueID() !== piece.getUniqueID());
                          setPieces(newPieces);
                        } else if (tool.mode === "add") {
                          // Remove and add new piece
                          const newPiece = new NewPiece({
                            type: "manual",
                            i: piece.i,
                            piece: tool.piece!.type,
                            color: tool.piece!.color,
                            hex: board.getHexagons().find((h) => h.memIdx === piece.i)!,
                          });
                          const newPieces = pieces.filter((p) => p.getUniqueID() !== piece.getUniqueID());
                          newPieces.push(newPiece);
                          setPieces(newPieces);
                        }
                      }}
                    />
                  ))}

                {tool.mode === "move" &&
                  pieces.map((piece) => <PieceElement key={piece.getUniqueID()} piece={piece} rotation={0} />)}
              </svg>

              <Flex pos="absolute" top={0} left={0} style={{ zIndex: 10 }}>
                <EditorPieceBox color={Color.Black} orientation="vertical" setTool={setTool} tool={tool} />
              </Flex>
              <Flex pos="absolute" top={0} right={0} style={{ zIndex: 10 }}>
                <EditorPieceBox color={Color.Gray} orientation="vertical" setTool={setTool} tool={tool} />
              </Flex>

              <Flex
                pos="absolute"
                bottom={0}
                left={0}
                right={0}
                style={{ margin: "auto", zIndex: 10 }}
                justify="center"
              >
                <EditorPieceBox color={Color.White} orientation="horizontal" setTool={setTool} tool={tool} />
              </Flex>
            </Flex>

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

          <Flex direction="column" style={{ flexShrink: 0 }} gap="sm" align="center" justify="center" maw={300}>
            <Paper withBorder style={{ overflowY: "hidden", minHeight: 0 }} pos="relative" w="100%">
              <Flex direction="column" h="100%" style={{ minHeight: 0 }} p="sm" gap="sm">
                <Flex direction="column">
                  <Text size="sm" fw="500">
                    Castling
                  </Text>
                  <Paper withBorder radius="sm" p="xs">
                    <Flex direction="row" gap="md" align="center" justify="center">
                      {castlingAvailable.map((canCastle, index) => (
                        <CastlingComboBox
                          key={index}
                          index={index}
                          canCastle={canCastle}
                          setCanCastleForIndex={(idx, newCanCastle) => {
                            setCastlingAvailable((prev) => {
                              const newState = [...prev];
                              newState[idx] = newCanCastle;
                              return newState;
                            });
                          }}
                        />
                      ))}
                    </Flex>
                  </Paper>
                </Flex>
                <Select
                  label="Player"
                  value={activeColor.toString()}
                  onChange={(val) => {
                    if (!val) return;
                    setActiveColor(parseInt(val) as Color);
                  }}
                  data={[
                    { value: Color.White.toString(), label: "White" },
                    { value: Color.Gray.toString(), label: "Gray" },
                    { value: Color.Black.toString(), label: "Black" },
                  ]}
                />

                <SimpleGrid cols={2} spacing="sm" w="100%">
                  <NumberInput
                    label="Third Move Count"
                    value={thirdMoveCount}
                    min={0}
                    onChange={(val) => setThirdMoveCount((val as number) || 0)}
                  />
                  <NumberInput
                    label="Turn Counter"
                    value={turnCounter}
                    min={0}
                    onChange={(val) => setTurnCounter((val as number) || 0)}
                  />
                </SimpleGrid>

                <Button onClick={confirmPlacement} variant="light">
                  Confirm Placement
                </Button>

                <Button onClick={cancelCallback} variant="light" color="red">
                  Cancel
                </Button>
              </Flex>
            </Paper>
          </Flex>
        </Flex>

        <CopyButton value={currentFen}>
          {({ copied, copy }) => (
            <Button
              color={copied ? "gray" : "gray.9"}
              onClick={copy}
              variant="light"
              size="compact-md"
              leftSection={<IconCopy />}
              // The button needs to be a flexible item that can shrink
              style={{
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
                maxWidth: "95%",
              }}
            >
              {/* The text inside handles its own overflow */}
              <Text
                style={{
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {copied ? "Copied fen" : currentFen || ""}
              </Text>
            </Button>
          )}
        </CopyButton>
      </Flex>
    </Flex>
  );
};

export default ChessEditorView;
