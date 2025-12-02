import { DndContext, DragOverlay } from "@dnd-kit/core";
import { snapCenterToCursor } from "@dnd-kit/modifiers";
import {
  ActionIcon,
  Button,
  CopyButton,
  Flex,
  Group,
  Modal,
  NumberInput,
  Paper,
  Select,
  SimpleGrid,
  Switch,
  Text,
} from "@mantine/core";
import { useDisclosure, useHotkeys } from "@mantine/hooks";
import {
  IconArrowBackUp,
  IconBrain,
  IconBug,
  IconChess,
  IconCopy,
  IconEdit,
  IconHexagonLetterP,
  IconRotate,
  IconTree,
  IconX,
} from "@tabler/icons-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Color, Piece, TriHexChessWrapper } from "tri-hex-chess";
import useSound from "use-sound";
import captureMp3 from "./assets/capture.mp3";
import moveMp3 from "./assets/move-self.mp3";
import { Board } from "./class/Board.ts";
import { CheckMetadata } from "./class/CheckMetadata.ts";
import { NewMove } from "./class/NewMove.ts";
import { NewPiece } from "./class/NewPiece.ts";
import CheckHighlight from "./components/CheckHighlight";
import HexagonComponent from "./components/HexagonComponent.tsx";
import HexagonOutline from "./components/HexagonOutline.tsx";
import LastMoveHighlight from "./components/LastMoveHighlight";
import MaterialCounter from "./components/MaterialCounter.tsx";
import MoveHighlight from "./components/MoveHighlight";
import PieceElement from "./components/PieceElement.tsx";
import PromotionMenu from "./components/PromotionMenu";
import { SIZE } from "./lib/constants.ts";
import { PlayerMaterial } from "./lib/types";
import { fens } from "./lib/utils/saved_fens";
import { getPieces, LastMove, rotateArray, useSize } from "./utils.ts";
import { ClockTimes } from "./useClock.tsx";
import Clock, { ClockHandle } from "./components/Clock.tsx";
import ClockSetting, { ClockSettings } from "./components/ClockSetting.tsx";
import { PolicyMovesOut, ThinkResultForChessMoveWrapperDtoAndChessBoardDto } from "./api/def/model";
import { usePostChessPolicyMoves, usePostChessThink } from "./api/def/default/default.ts";
import Arrows from "./components/Arrows.tsx";
import { ChessThinkTreeDrawer } from "./components/ChessThinkTreeDrawer.tsx";
import HistoryInner from "./components/HistoryInner.tsx";
import PositionEvaluation from "./components/PositionEvaluation.tsx";
import DebugVew from "./components/DebugVew.tsx";
import ChessEditorView from "./components/ChessEditorView.tsx";

const board = new Board(SIZE);

export type HistoryItem = {
  fen: string;
  clockTimes: ClockTimes;
  turn_counter: number;
  color: Color;
  lan: String;
};

function App() {
  const state = useRef<TriHexChessWrapper | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  if (state.current === null) state.current = TriHexChessWrapper.new(fens[0].fen);

  // The following state variables are controlled by WASM
  const [gameWon, setGameWon] = useState<Color | undefined>(undefined);
  const [currentTurn, setCurrentTurnA] = useState<Color>(Color.White);
  const [pieces, setPieces] = useState<NewPiece[]>([]);
  const [checks, setChecks] = useState<CheckMetadata[]>([]);
  const [material, setMaterial] = useState<PlayerMaterial>({
    [Color.White]: 0,
    [Color.Gray]: 0,
    [Color.Black]: 0,
  });
  const [fen, setFen] = useState("");

  const [debugView, setDebugView] = useState(false);
  const [editView, setEditView] = useState(true);

  const [clockSettings, setClockSettings] = useState<null | ClockSettings>({
    incrementMs: 0,
    incrementOnTimeoutMs: 0,
    minutesMs: 0,
    useClock: false,
  });

  const [rotateWithPlayer, setRotateWithPlayer] = useState(false);

  const setRotate = (rotate: boolean) => {
    setRotateWithPlayer(rotate);

    // TODO: Maybe make this dynamic, not only on next turn (this is why we have a wrapper fn).
  };

  const confirmClockSettings = (settings: ClockSettings) => {
    setClockSettings(settings);

    if (settings.useClock) {
      setCurrentTurnA(Color.White);
      clockRef.current?.resetClock([settings.minutesMs, settings.minutesMs, settings.minutesMs]);
    }
  };

  const setCurrentTurn = (turn: Color) => {
    // for debugging purposes
    setCurrentTurnA(turn);
  };

  const size = useSize(svgRef);
  const [advanceTurns, setAdvanceTurns] = useState(true);
  const [lastMoves, setLastMoves] = useState<LastMove[]>([]);
  const [highlighted, setHighlighted] = useState<NewMove[]>([]);
  const [rotation, setRotation] = useState(0); // 0, 120, 240 - Rotation of the board
  const [draggingPiece, setDraggingPiece] = useState<NewPiece | null>(null);
  const [playMoveSound] = useSound(moveMp3, { volume: 0.5 });
  const [playCaptureSound] = useSound(captureMp3, { volume: 0.5 });
  const [promotionMove, setPromotionMove] = useState<NewMove | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [modalOpen, setModalOpen] = useState(true);

  const clockRef = useRef<ClockHandle>(null);

  const [thinkResult, setThinkResult] = useState<ThinkResultForChessMoveWrapperDtoAndChessBoardDto | null>(null);
  const [policyResult, setPolicyResult] = useState<PolicyMovesOut | null>(null);

  const { mutateAsync: thinkApi, isPending: isLoadingThink } = usePostChessThink();
  const { mutateAsync: policyApi, isPending: isLoadingPolicy } = usePostChessPolicyMoves();

  const [playouts, setPlayouts] = useState(200);
  const [explorationFactor, setExplorationFactor] = useState(1.4);
  const [contempt, setContempt] = useState(0);
  const [virtualLossWeight, setVirtualLossWeight] = useState(1.0);

  const policyCallback = useCallback(async () => {
    if (!fen) return;

    try {
      const policyResult = await policyApi({
        data: {
          fen,
          number_of_moves: 9000,
        },
      });

      console.log("Policy result:", policyResult);

      setPolicyResult(policyResult);
      setThinkResult(null);
    } catch (error) {
      console.error("Error getting policy:", error);
    }
  }, [fen]);

  const thinkCallback = useCallback(
    async (numberOfRollouts: number, explorationFactor: number) => {
      if (!fen) {
        return;
      }

      try {
        const thinkResult = await thinkApi({
          data: {
            board_state: fen,
            number_of_rollouts: numberOfRollouts,
            exploration_factor: explorationFactor,
            contempt: -0.1,
            virtual_loss_weight: 1.0,
          },
        });

        setPolicyResult(null);
        setThinkResult(thinkResult);
      } catch (error) {
        console.error("Error thinking:", error);
      }
    },
    [fen],
  );

  // Handles state (mobving to other player when the time is up)
  const onClockTimeout = (c: Color) => {
    if (!state.current || !clockRef.current) {
      throw new Error("Assert: state.current|clockRef.current should not be null");
    }
    // we don't add increment since the useClock will do that

    // just prevent some re-rendering problem - probably strict mode
    if (state.current.getGameState().turn !== c) return;

    history.push({
      clockTimes: clockRef.current.getLastTimes(),
      fen: state.current.getFen(),
      color: c,
      turn_counter: state.current.getGameState().turn_counter,
      lan: "/",
    });

    setHistory([...history]);

    state.current.skipToNextPlayer();

    updateFromGameState();
  };

  useHotkeys([["a", () => setAdvanceTurns(!advanceTurns)]]);

  const updateFromGameState = () => {
    if (!state.current) return;

    setThinkResult(null);

    const gameState = state.current.getGameState();

    let newRotation = (gameState.turn_counter - 1) * 360 + gameState.turn * 120;

    if (!rotateWithPlayer) newRotation = 0;

    setCurrentTurn(gameState.turn);
    setGameWon(gameState.won);
    setChecks(state.current.getCheckMetadata().map((check) => new CheckMetadata(check)));
    setMaterial((p) => {
      if (!state.current) return p;
      const material = state.current.getMaterial();
      return { [Color.White]: material.white, [Color.Gray]: material.gray, [Color.Black]: material.black };
    });
    setFen(state.current.getFen());
    setRotation(newRotation);
    setPieces(getPieces(state.current, newRotation));
  };

  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const fen = urlParams.get("fen");

    if (fen) {
      state.current = TriHexChessWrapper.new(fen);
      updateFromGameState();
    }
  }, []);

  const promotionCallback = (move: NewMove | null, piece: Piece | null) => {
    setPromotionMove(null);

    if (!move || !state.current) return;
    if (!move || !piece) {
      throw new Error("Assert: move and piece should not be null");
    }

    let clockTimes: ClockTimes = [0, 0, 0];

    if (clockRef.current) {
      clockRef.current.addIncrement(currentTurn);
      clockTimes = clockRef.current.getLastTimes();
    }

    setHistory([
      ...history,
      {
        clockTimes,
        fen: state.current.getFen(),
        color: currentTurn,
        turn_counter: state.current.getGameState().turn_counter,
        lan: move.notationLan,
      },
    ]);

    setLastMoves([{ from: move.ref.from, to: move.ref.to }, ...lastMoves.slice(0, 1)]);

    state.current.commitMove(move.ref, piece, advanceTurns);

    updateFromGameState();

    if (move.isCapture()) playCaptureSound();
    else playMoveSound();
  };

  useEffect(() => {
    if (!state.current) return;

    updateFromGameState();

    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const takeBack = () => {
    if (!state.current) {
      throw new Error("Assert: state.current should not be null");
    }

    // get the last fen from the history
    const lastFen = history.pop();

    if (!lastFen) return;

    state.current.setFen(lastFen.fen.toString());

    if (clockRef.current) {
      clockRef.current.setClockTimes(lastFen.clockTimes);
    }

    lastMoves.shift();

    setHistory([...history]);

    updateFromGameState();
  };

  useHotkeys([
    ["ArrowLeft", () => takeBack()],
    [
      "ArrowRight",
      () => {
        if (!state.current) {
          throw new Error("Assert: state.current should not be null");
        }

        const nextMoves = state.current.queryAllMoves();

        if (nextMoves.length === 0) return;

        const move = new NewMove(nextMoves[0], rotation);

        let clockTimes: ClockTimes = [0, 0, 0];

        if (clockRef.current) {
          clockRef.current.addIncrement(currentTurn);

          clockTimes = clockRef.current.getLastTimes();
        }

        setHistory([
          ...history,
          {
            clockTimes,
            fen: state.current.getFen(),
            color: currentTurn,
            turn_counter: state.current.getGameState().turn_counter,
            lan: move.notationLan,
          },
        ]);

        setLastMoves([{ from: move.ref.from, to: move.ref.to }, ...lastMoves.slice(0, 1)]);

        state.current.commitMove(move.ref, undefined, advanceTurns);

        updateFromGameState();

        if (move.isCapture()) playCaptureSound();
        else playMoveSound();
      },
    ],
  ]);

  const [opened, { open: openDrawer, close: closeDrawer }] = useDisclosure(false);

  // FIXME: This is a little janky :)
  const s = size ? Math.min((size.width / board.getWidth()) * 70 + 8, 70) : 70;

  return (
    <Flex w="100vw" h="100vh" p="sm" gap="sm" justify="center" align="center" direction="row" bg="gray.0">
      {debugView && (
        <Flex w="100%" h="100vh" pos="absolute" top={0} left={0} bg="white" style={{ zIndex: 5 }}>
          <Button pos="absolute" top={10} right={10} onClick={() => setDebugView(false)} style={{ zIndex: 10 }}>
            Close debug view
          </Button>
          <DebugVew fen={fen} />
        </Flex>
      )}

      {editView && (
        <Flex w="100%" h="100vh" pos="absolute" top={0} left={0} bg="white" style={{ zIndex: 5 }}>
          <ChessEditorView
            initialFen={fen}
            cancelCallback={() => {
              setEditView(false);
            }}
            confirmationCallback={(fen) => {
              if (!state.current) return;

              state.current.setFen(fen);
              updateFromGameState();

              setEditView(false);
            }}
          />
        </Flex>
      )}

      <ChessThinkTreeDrawer opened={opened} onClose={closeDrawer} onHexClick={() => {}} thinkResult={thinkResult} />

      <Flex direction="column">
        {!clockSettings && <ClockSetting onConfirm={confirmClockSettings} />}
        <Modal
          opened={gameWon !== undefined && modalOpen}
          onClose={() => setModalOpen(true)}
          title={(gameWon === Color.White ? "Beli" : gameWon === Color.Gray ? "Sivi" : "Črni") + " je zmagal."}
          size="sm"
          withCloseButton
        >
          <SimpleGrid cols={3}>
            {rotateArray(["wP", "gP", "bP"], -1 + (gameWon || 0)).map((piece, i) => (
              <svg width="100%" height="100%" key={i}>
                <image
                  href={`/pieces/${piece}.svg`}
                  type="image/svg+xml"
                  width={i === 1 ? "100%" : "80%"}
                  height={i === 1 ? "100%" : "80%"}
                  opacity={i === 1 ? 1 : 0.8}
                  style={{
                    filter: "grayscale(1)",
                  }}
                />
              </svg>
            ))}
          </SimpleGrid>

          <Text>
            Odigranih je bilo {state.current?.getGameState().turn_counter} potez. Beli ima {material[Color.White]}{" "}
            materiala, sivi {material[Color.Gray]}, črni pa {material[Color.Black]}.
          </Text>

          <Group mt="xs" justify="space-between">
            <Button onClick={() => setModalOpen(false)} variant="white" color="black">
              Zapri okno
            </Button>

            <Button color="pink.9" onClick={() => window.location.reload()} leftSection={<IconChess />}>
              Nova igra
            </Button>
          </Group>
        </Modal>
      </Flex>

      <Flex direction="column" style={{ flexGrow: 1 }} h="100%" w="100%" align="center" pt="lg" pos="relative">
        <Flex w="100%" justify="center" pos="absolute">
          <PositionEvaluation thinkResult={thinkResult} isLoading={isLoadingThink} />
        </Flex>
        <Flex direction="row" gap="sm" style={{ flexGrow: 1, minHeight: 0 }}>
          <DndContext
            onDragStart={(e) => {
              if (!state.current) return;
              const piece: NewPiece = e.active.data.current as NewPiece;

              setDraggingPiece(piece);
              setHighlighted(
                state.current.queryMoves(piece.ref.coordinates).map((move) => new NewMove(move, rotation)),
              );
            }}
            onDragEnd={(e) => {
              setDraggingPiece(null); // Stop the dragging animation
              setHighlighted([]); // Clear the highlighted moves

              const svgElement = svgRef.current;

              if (!svgElement || !state.current) {
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

              let move: NewMove | null = null;

              for (const hex of highlighted) {
                const d = Math.sqrt(Math.pow(f_x_1 - hex.x, 2) + Math.pow(f_y_1 - hex.y, 2));

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

              let clockTimes: ClockTimes = [0, 0, 0];

              if (clockRef.current) {
                clockRef.current.addIncrement(currentTurn);

                clockTimes = clockRef.current.getLastTimes();
              }

              setHistory([
                ...history,
                {
                  clockTimes,
                  fen: state.current.getFen(),
                  color: currentTurn,
                  turn_counter: state.current.getGameState().turn_counter,
                  lan: move.notationLan,
                },
              ]);

              state.current.commitMove(move.ref, undefined, advanceTurns);

              setLastMoves([{ from: move.ref.from, to: move.ref.to }, ...lastMoves.slice(0, 1)]);

              updateFromGameState();

              if (move.isCapture()) playCaptureSound();
              else playMoveSound();
            }}
          >

            <svg viewBox={board.getViewbox()} width="95%" style={{ maxHeight: "90vh" }} height={"100%"} ref={svgRef}>
              <g transform={`rotate(${rotation})`} style={{ transition: "0.2s" }}>
                {/* Show all hexagons (chessboard background) and algebraic notation at the side */}
                {board.getHexagons().map((hex, index) => (
                  <HexagonOutline key={index} hex={hex} color="black" strokeWidth="6" />
                ))}

                {/* Show all hexagons (chessboard background) and algebraic notation at the side */}
                {board.getHexagons().map((hex, index) => (
                  <HexagonComponent key={index} hex={hex} rotation={rotation} />
                ))}

                {/* Show all present checks */}
                {checks.map((check, index) => (
                  <CheckHighlight check={check} key={index} />
                ))}

                <MaterialCounter material={material} rotation={rotation} currentTurn={currentTurn} />

                <Arrows moves={thinkResult?.moves || []} topN={3} />
                <Arrows moves={policyResult?.moves || []} topN={9} />

                {clockSettings?.useClock && (
                  <Clock
                    rotation={rotation}
                    ref={clockRef}
                    currentTurn={currentTurn}
                    isClockRunning={!!clockSettings && gameWon === undefined}
                    onTimeout={onClockTimeout}
                    config={
                      clockSettings
                        ? {
                            defaultTime: clockSettings.minutesMs,
                            increment: clockSettings.incrementMs,
                            incrementOnTimeout: clockSettings.incrementOnTimeoutMs,
                          }
                        : undefined
                    }
                  />
                )}
              </g>

              {/* Show last played moves */}
              {lastMoves.map(({ from, to }, index) => (
                <LastMoveHighlight key={index} from={from} to={to} isOldest={index === 0} rotation={rotation} />
              ))}

              {/* Show all chess pieces */}
              {pieces.map((piece) => (
                <PieceElement key={piece.getUniqueID()} piece={piece} rotation={rotation} />
              ))}

              {/* Possible piece moves when a piece is active (hovering for placement) */}
              {highlighted.map((hex, index) => (
                <MoveHighlight key={index} move={hex} />
              ))}

              {/* Show 4 promotion hexagons */}
              {promotionMove && <PromotionMenu move={promotionMove} callback={promotionCallback} />}
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

          <Flex direction="column" style={{ flexShrink: 0 }} gap="sm" align="center" h="100%" justify="center">
            <Paper withBorder h="90%" style={{ overflowY: "hidden", minHeight: 0 }} pos="relative" w="100%">
              <Flex direction="column" h="100%" style={{ minHeight: 0 }}>
                <Flex
                  p="lg"
                  style={{ borderBottom: "3px solid #eee" }}
                  justify="space-between"
                  align="center"
                  direction="column"
                  gap="lg"
                >
                  <Group w="100%">
                    <ActionIcon
                      size="lg"
                      color="grape"
                      variant="light"
                      onClick={() => thinkCallback(playouts, explorationFactor)}
                      loading={isLoadingThink}
                    >
                      <IconBrain />
                    </ActionIcon>

                    <ActionIcon
                      size="lg"
                      variant="light"
                      color="green"
                      disabled={!thinkResult}
                      onClick={() => openDrawer()}
                      loading={isLoadingThink}
                    >
                      <IconTree />
                    </ActionIcon>

                    <ActionIcon
                      size="lg"
                      color="teal"
                      variant="light"
                      onClick={() => policyCallback()}
                      loading={isLoadingPolicy}
                    >
                      <IconHexagonLetterP />
                    </ActionIcon>

                    <ActionIcon
                      variant="light"
                      size="lg"
                      color="orange"
                      onClick={() => {
                        takeBack();
                      }}
                      disabled={history.length === 0}
                    >
                      <IconArrowBackUp />
                    </ActionIcon>

                    <ActionIcon
                      variant="light"
                      size="lg"
                      color="red"
                      onClick={() => {
                        setDebugView(true);
                      }}
                    >
                      <IconBug />
                    </ActionIcon>

                    <ActionIcon
                      variant="light"
                      size="lg"
                      color="lime"
                      onClick={() => {
                        setEditView(true);
                      }}
                    >
                      <IconEdit />
                    </ActionIcon>
                  </Group>

                  {/*  2x grid for ai settings*/}
                  <SimpleGrid cols={2} spacing="md">
                    <NumberInput
                      label="Exploration"
                      value={explorationFactor}
                      onChange={(value) => {
                        if (value) {
                          setExplorationFactor(value as number);
                        }
                      }}
                      step={0.1}
                      min={0}
                      max={5}
                      size="sm"
                      style={{ width: 100 }}
                    />

                    <Select
                      label="Playouts"
                      value={playouts.toString()}
                      onChange={(value) => {
                        if (value) {
                          setPlayouts(parseInt(value));
                        }
                      }}
                      data={["0", "1", "50", "200", "500", "650", "1200", "2400", "5000", "10000"]}
                      size="sm"
                      style={{ width: 100 }}
                    />

                    <NumberInput
                      label="Contempt"
                      value={contempt}
                      onChange={(value) => {
                        if (value) {
                          setContempt(value as number);
                        }
                      }}
                      min={-1}
                      max={1}
                      size="sm"
                      style={{ width: 100 }}
                    />

                    <NumberInput
                      label="Virtual loss"
                      value={virtualLossWeight}
                      onChange={(value) => {
                        if (value) {
                          setVirtualLossWeight(value as number);
                        }
                      }}
                      min={0}
                      max={5}
                      size="sm"
                      style={{ width: 100 }}
                    />
                  </SimpleGrid>

                  <Switch
                    checked={rotateWithPlayer}
                    onChange={(event) => setRotate(event.currentTarget.checked)}
                    color="teal"
                    size="md"
                    label="Board rotation"
                    thumbIcon={
                      rotateWithPlayer ? (
                        <IconRotate size={12} color="var(--mantine-color-teal-6)" stroke={3} />
                      ) : (
                        <IconX size={12} color="var(--mantine-color-red-6)" stroke={3} />
                      )
                    }
                  />
                </Flex>
                <HistoryInner history={history} />
              </Flex>
            </Paper>
          </Flex>
        </Flex>

        <CopyButton value={fen}>
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
                {copied ? "Copied fen" : fen || ""}
              </Text>
            </Button>
          )}
        </CopyButton>
      </Flex>
    </Flex>
  );
}

export default App;
