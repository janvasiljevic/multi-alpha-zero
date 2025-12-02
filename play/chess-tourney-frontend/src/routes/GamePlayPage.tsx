import {
  GameRelation,
  GameStatus,
  type Move,
  type PlayerUpdate,
  PromotionPiece,
  type WsEvent,
} from "../api/model";
import { useNavigate, useParams } from "@tanstack/react-router";
import { type JSX, useEffect, useRef, useState } from "react";
import {
  useGamesLeaveGame,
  useGamesMakeMove,
  useGamesStartGame,
} from "../api/game-management/game-management.ts";
import { Badge, Button, Flex, Paper, Text, ThemeIcon } from "@mantine/core";
import { gameDetailRoute, gamesRoute } from "../routes.tsx";
import BoardComponent, {
  type Players,
  playerStatusToPlayers,
} from "../components/Board.tsx";
import { Color, Piece } from "../../libs/wasm";
import { showNotification } from "@mantine/notifications";
import type { NewMove } from "../class/NewMove.ts";
import { useAuthStore } from "../authStore.ts";
import { useDisclosure } from "@mantine/hooks";
import EndGameModal from "../components/EndGameModal.tsx";
import { libColorToHumanString } from "../class/utils.ts";
import { IconThumbDown, IconTie, IconTrophy } from "@tabler/icons-react";
import AddBotToGameModal from "../components/AddBotToGameModal.tsx";
import AutoResetTimer from "../components/AutoResetTimer.tsx";
import RemovePlayerModal from "../components/RemovePlayerModal.tsx";
import { isGameOver } from "../common.ts";

type HandlerMap = {
  [K in WsEvent["kind"]]: Extract<WsEvent, { kind: K }>["value"] extends infer V
    ? (value: V) => void
    : never;
};

export function handleWsEvent<E extends WsEvent>(
  event: E,
  handlers: Partial<HandlerMap>,
) {
  const handler = handlers[event.kind] as
    | ((value: E["value"]) => void)
    | undefined;

  if (handler) {
    handler(event.value);
  } else {
    console.warn(`No handler for event kind "${event.kind}"`);
  }
}

const getSpectatingBool = (players: PlayerUpdate, myId: number): boolean => {
  const isPlayer = Object.values(players).some(
    (p) => p !== null && p.id === myId,
  );
  return !isPlayer;
};

const GameDetailPage = () => {
  const { id } = useParams({ from: gameDetailRoute.id });
  const auth = useAuthStore();
  const navigate = useNavigate({ from: gameDetailRoute.id });

  const { mutateAsync: startGameApi } = useGamesStartGame();
  const { mutateAsync: leaveGameApi } = useGamesLeaveGame();
  const { mutateAsync: makeMoveApi } = useGamesMakeMove();

  const [fen, setFen] = useState<string | null>(null);
  const [players, setPlayers] = useState<Players | null>(null);
  const [myColor, setMyColor] = useState<Color | null>(null);
  const myColorRef = useRef<Color | null>(null);

  const [gameStatus, setGameStatus] = useState<GameStatus>(GameStatus.Waiting);
  const [iAmOwner, setIAmOwner] = useState<boolean>(false);
  const [ownerUsername, setOwnerUsername] = useState<string | null>(null);
  const [spectating, setSpectating] = useState<boolean>(false);
  const [gameName, setGameName] = useState<string>("");
  const [suggestedMoveTimeSeconds, setSuggestedMoveTimeSeconds] = useState<
    number | null
  >(null);
  const [hideMaterial, setHideMaterial] = useState<boolean>(false);

  const [endGameModalOpen, endGameModal] = useDisclosure(false);
  const [addBotModalOpen, addBotModal] = useDisclosure(false);
  const [removePlayerModalOpen, removePlayerModal] = useDisclosure(false);
  const [trainingMode, setTrainingMode] = useState<boolean>(false);

  const [highlightedMoves, setHighlightedMoves] = useState<Move[]>([]);

  const addMoveToHighlight = (move: Move) => {
    // truncate to max 2 moves
    setHighlightedMoves((prev) => {
      const newMoves = [...prev, move];
      while (newMoves.length > 2) {
        newMoves.shift();
      }
      return newMoves;
    });
  };

  const playerCount = players
    ? Object.values(players).filter((p) => p !== null).length
    : 0;

  const canStartGame = playerCount === 3 && gameStatus === GameStatus.Waiting;

  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (wsRef.current) return;

    //@ts-ignore
    const wsUrl = import.meta.env.VITE_WSS_API || "ws://localhost:3000/ws";

    const ws = new WebSocket(wsUrl + "?game_id=" + id, [
      "Authorization",
      auth.token || "",
    ]);

    wsRef.current = ws;

    ws.onopen = () => console.log("WebSocket connection established");
    ws.onmessage = (msg) => {
      const raw = JSON.parse(msg.data) as { kind: string; value: any };

      const mappedKind = (() => {
        switch (raw.kind) {
          case "WsEventJoined":
            return "joined";
          case "WsEventLeft":
            return "left";
          case "WsEventStarted":
            return "started";
          case "WsEventEnded":
            return "ended";
          case "WsEventMoveMade":
            return "moveMade";
          case "WsEventOnJoin":
            return "onJoin";
          case "WsEventPlayerUpdate":
            return "playerUpdate";
          default:
            throw new Error(`Unknown event kind: ${raw.kind}`);
        }
      })();

      const data = { kind: mappedKind, value: raw.value } as WsEvent;

      handleWsEvent(data, {
        joined: (v) => {
          setPlayers(playerStatusToPlayers(v.players, auth.user?.id || -1));

          showNotification({
            title: "Player Joined",
            message: `${v.username} joined the game.`,
            color: "green",
          });
        },
        playerUpdate: (v) => {
          setPlayers(playerStatusToPlayers(v.players, auth.user?.id || -1));
        },
        left: (v) => {
          setPlayers(playerStatusToPlayers(v.players, auth.user?.id || -1));

          showNotification({
            title: "Player Left",
            message: `${v.username} left the game.`,
            color: "yellow",
          });
        },
        started: (v) => {
          setGameStatus(v.game.status);

          showNotification({
            title: "Game Started",
            message: `The game has started!`,
            color: "blue",
          });
        },
        ended: (v) => {
          setGameStatus(v.game.status);
          setPlayers(
            v.game.players
              ? playerStatusToPlayers(v.game.players, auth.user?.id || -1)
              : null,
          );
          setFen(v.game.fen);
        },
        moveMade: (v) => {
          setFen(v.newFen);
          addMoveToHighlight(v.move);

          let newTurn: Color;

          switch (v.newTurn) {
            case "White":
              newTurn = Color.White;
              break;
            case "Grey":
              newTurn = Color.Gray;
              break;
            case "Black":
              newTurn = Color.Black;
              break;
          }

          if (newTurn === myColorRef.current) {
            showNotification({
              title: "Your Turn",
              message: `It's your turn to move.`,
              color: "red",
              position: "top-center",
              autoClose: 5000,
              withBorder: true,
              radius: "xl",
              py: "md",
            });
          }
        },
        onJoin: (v) => {
          if (v.game.players?.black?.id === auth.user?.id) {
            setMyColor(Color.Black);
            myColorRef.current = Color.Black;
          } else if (v.game.players?.grey?.id === auth.user?.id) {
            setMyColor(Color.Gray);
            myColorRef.current = Color.Gray;
          } else if (v.game.players?.white?.id === auth.user?.id) {
            setMyColor(Color.White);
            myColorRef.current = Color.White;
          }

          if (v.game.players) {
            setSpectating(
              getSpectatingBool(v.game.players, auth.user?.id || -1),
            );
          }
          setTrainingMode(v.game.training_mode);
          setHideMaterial(v.game.material_masked);
          setSuggestedMoveTimeSeconds(v.game.suggested_move_time_seconds);
          setGameName(v.game.name);
          setOwnerUsername(v.game.ownerUsername);
          setIAmOwner(v.game.ownerId === auth.user?.id);
          setPlayers(
            v.game.players
              ? playerStatusToPlayers(v.game.players, auth.user?.id || -1)
              : null,
          );
          setGameStatus(v.game.status);
          setFen(v.game.fen);
        },
      });
    };
    ws.onerror = (err) => {
      showNotification({
        title: "WebSocket Error",
        message:
          "An error occurred with the WebSocket connection: " +
          JSON.stringify(err),
        color: "red",
      });
    };

    return () => {
      // This will be called twice by StrictMode before mount
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, []);

  const handleLeaveGame = async () => {
    try {
      await leaveGameApi({ gameId: id });

      await navigate({ to: gamesRoute.id });
    } catch (error) {
      console.error("Error leaving the game:", error);

      await navigate({ to: gamesRoute.id });
    }
  };

  const handleStopSpectating = async () => {
    await navigate({ to: gamesRoute.id });
  };

  const handleStartGame = async () => {
    try {
      await startGameApi({ gameId: id });
    } catch (error) {}
  };

  const handleMakeMove = async (
    move: NewMove,
    promotionPiece: Piece | null,
  ) => {
    try {
      let apiPiece: PromotionPiece | undefined = undefined;

      if (promotionPiece !== null) {
        switch (promotionPiece) {
          case Piece.Knight:
            apiPiece = PromotionPiece.Knight;
            break;
          case Piece.Bishop:
            apiPiece = PromotionPiece.Bishop;
            break;
          case Piece.Rook:
            apiPiece = PromotionPiece.Rook;
            break;
          case Piece.Queen:
            apiPiece = PromotionPiece.Queen;
            break;
          default:
            throw new Error("Invalid promotion piece");
        }
      }

      await makeMoveApi({
        gameId: id,
        data: {
          fromIndex: move.ref.from.i,
          toIndex: move.ref.to.i,
          promotion: apiPiece,
        },
      });
    } catch (error) {
      console.error("Error making move:", error);
    }
  };

  const isAdmin = auth.user?.type === "Admin";

  return (
    <Flex w="100vw" h="100vh" direction="column" align="center" gap="md">
      <EndGameModal
        gameId={id}
        opened={endGameModalOpen}
        onClose={endGameModal.close}
      />
      <AddBotToGameModal
        opened={addBotModalOpen}
        onClose={addBotModal.close}
        gameId={parseInt(id)}
        players={players}
      />
      <RemovePlayerModal
        opened={removePlayerModalOpen}
        onClose={removePlayerModal.close}
        gameId={parseInt(id)}
        players={players}
      />
      <Flex p="sm" w="100%" justify="center">
        <Paper withBorder radius="md" w="100%" mt="sm">
          <Flex
            gap="sm"
            p="sm"
            align="center"
            direction="row"
            justify="space-between"
          >
            <Flex align="baseline" gap="sm" wrap="wrap">
              <Text size="xs">#{id}</Text>
              <Text size="md" fw={700}>
                {gameName}
              </Text>
              {ownerUsername && (
                <Badge color="gray" variant="light">
                  Owner: {ownerUsername} {iAmOwner && "(You)"}
                </Badge>
              )}
            </Flex>

            <Flex gap="sm" wrap="wrap" align="center">
              {spectating && (
                <Button
                  color="yellow"
                  onClick={handleStopSpectating}
                  size="compact-md"
                >
                  Stop Spectating
                </Button>
              )}
              {!spectating && (
                <Button onClick={handleLeaveGame} color="red" size="compact-md">
                  Leave
                </Button>
              )}

              {(isAdmin || iAmOwner) && gameStatus === GameStatus.Waiting && (
                <Flex gap="sm">
                  <Button
                    onClick={addBotModal.open}
                    color="blue"
                    size="compact-md"
                  >
                    Add Bot
                  </Button>

                  <Button
                    onClick={removePlayerModal.open}
                    color="orange"
                    size="compact-md"
                  >
                    Remove Player
                  </Button>
                </Flex>
              )}

              {isAdmin && gameStatus === GameStatus.InProgress && (
                <Button
                  color="red"
                  onClick={endGameModal.open}
                  size="compact-md"
                >
                  End Game (A)
                </Button>
              )}
            </Flex>
          </Flex>
        </Paper>
      </Flex>

      <Flex
        pos="relative"
        style={{
          flexGrow: 1,
          minHeight: 0,
          width: "100%",
        }}
        h="100%"
        justify="center"
        align="center"
        direction="column"
      >
        {isGameOver(gameStatus) && (
          <Flex
            pos="absolute"
            left={0}
            right={0}
            top={0}
            p="md"
            justify="center"
          >
            <EndGameShowcase players={players} />
          </Flex>
        )}

        <BoardComponent
          fenOutside={fen}
          players={players}
          rotation={myColor !== null ? myColor * 120 : 0}
          color={myColor}
          gameStatus={gameStatus}
          onMovePlayed={handleMakeMove}
          highlightedMoves={highlightedMoves}
          hideMaterial={hideMaterial}
          showPatterns={trainingMode}
        />
        {suggestedMoveTimeSeconds && (
          <AutoResetTimer
            duration={suggestedMoveTimeSeconds || 0}
            active={gameStatus === GameStatus.InProgress}
            resetKey={highlightedMoves}
          />
        )}

        {gameStatus != GameStatus.InProgress && (
          <Flex
            pos="absolute"
            w="100%"
            h="100%"
            align="center"
            justify="center"
            direction="column"
          >
            <Badge
              size="xl"
              variant="dot"
              color={canStartGame ? "green" : "orange"}
              style={(theme) => ({
                boxShadow: theme.shadows.md,
                zIndex: 10,
              })}
            >
              {gameStatus === GameStatus.Waiting &&
                playerCount != 3 &&
                `Waiting for players to join (${playerCount}/3)`}
              {canStartGame &&
                !iAmOwner &&
                "Waiting for the owner to start the game"}
              {canStartGame &&
                (iAmOwner || isAdmin) &&
                "You can start the game"}
              {gameStatus === GameStatus.FinishedDraw && "Game ended in a draw"}
              {gameStatus == GameStatus.FinishedWin &&
                "Game over - we have a winner!"}
              {gameStatus == GameStatus.FinishedSemiDraw &&
                "Game ended in a semi-draw"}
            </Badge>

            {canStartGame && (iAmOwner || isAdmin) && (
              <Button
                mt="md"
                onClick={handleStartGame}
                bg="white"
                variant="outline"
                color="green"
                radius="xl"
                style={(t) => ({
                  shadow: t.shadows.xl,
                })}
              >
                Start Game
              </Button>
            )}
          </Flex>
        )}
      </Flex>
    </Flex>
  );
};

type EndGameShowcaseProps = {
  players: Players | null;
};

const relationToColor = (relation: GameRelation) => {
  switch (relation) {
    case "winner":
      return "#10d335";
    case "loser":
      return "#e50718";
    case "draw":
      return "#e0b00d";
    default:
      return "gray";
  }
};

const relationToIcon = (relation: GameRelation): JSX.Element => {
  switch (relation) {
    case "winner":
      return <IconTrophy />;
    case "loser":
      return <IconThumbDown />;
    case "draw":
      return <IconTie />;
    default:
      return <IconTie />;
  }
};

const EndGameShowcase = ({ players }: EndGameShowcaseProps) => {
  if (!players) return <></>;

  return (
    <Flex direction="column" align="center" gap="md">
      <Text>The game has ended! Here are the final players:</Text>
      <Flex gap="sm">
        {[Color.White, Color.Gray, Color.Black].map((color) => {
          const player = players[color];
          if (!player) return <></>;

          return (
            <Paper
              key={color}
              p="sm"
              withBorder
              radius="md"
              bg="gray.0"
              style={{ borderColor: relationToColor(player.relation!) }}
            >
              <Flex key={color} direction="column" align="center" gap="2px">
                <Badge color="gray">{libColorToHumanString(color)}</Badge>
                <Text>{player ? player.username : "No player"}</Text>
                <Text c={relationToColor(player.relation!)}>
                  {player.relation || "No result"}
                </Text>
                <ThemeIcon
                  color={relationToColor(player.relation!)}
                  variant="transparent"
                  size="lg"
                >
                  {relationToIcon(player.relation!)}
                </ThemeIcon>
              </Flex>
            </Paper>
          );
        })}
      </Flex>
    </Flex>
  );
};

export default GameDetailPage;
