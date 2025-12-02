import {
  GameStatus,
  type PlayerColor,
  type PlayerUpdate,
  type WsEvent,
} from "../api/model";
import { useNavigate, useParams } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import {
  Badge,
  Box,
  Button,
  Flex,
  Paper,
  ScrollArea,
  Text,
} from "@mantine/core";
import { gameHistoryRoute, gamesRoute } from "../routes.tsx";
import BoardComponent, {
  type Players,
  playerStatusToPlayers,
} from "../components/Board.tsx";
import { useAuthStore } from "../authStore.ts";
import { useHistoryGetGameHistory } from "../api/game-history/game-history.ts";
import { useHotkeys } from "@mantine/hooks";

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

const entryWidth = 80;
const entryHeight = 30;

const playerColorToIndex = (color: PlayerColor): number => {
  switch (color) {
    case "White":
      return 0;
      break;
    case "Grey":
      return 1;
      break;
    case "Black":
      return 2;
      break;
  }
};

const GameHistoryPage = () => {
  const { id } = useParams({ from: gameHistoryRoute.id });
  const auth = useAuthStore();
  const navigate = useNavigate({ from: gameHistoryRoute.id });

  const { data } = useHistoryGetGameHistory(parseInt(id));

  const [currentIndex, setCurrentIndex] = useState(0);
  const [currentFen, setCurrentFen] = useState<string | null>(null);
  const [players, setPlayers] = useState<Players | null>(null);

  useEffect(() => {
    setCurrentIndex(0);
    if (data && data.history.length > 0) {
      if (data.game.players) {
        setPlayers(
          playerStatusToPlayers(
            data.game.players as PlayerUpdate,
            auth.user?.id || -1,
          ),
        );
      }
      setCurrentFen(data.history[0].fen);
    }
  }, [data]);

  const incrementIndex = () => {
    if (data && currentIndex < data.history.length - 1) {
      setCurrentIndex(currentIndex + 1);
      setCurrentFen(data.history[currentIndex + 1].fen);
    }
  };

  const decrementIndex = () => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
      setCurrentFen(data!.history[currentIndex - 1].fen);
    }
  };

  useHotkeys([
    ["ArrowRight", incrementIndex],
    ["ArrowLeft", decrementIndex],
  ]);

  const goBackToGamesList = async () => {
    await navigate({ to: gamesRoute.id });
  };

  const iAmOwner = data?.game.ownerId === auth.user?.id;

  return (
    <Flex w="100vw" h="100vh" direction="column" align="center" gap="md">
      <Flex w="100%" justify="center" p="sm">
        <Paper withBorder radius="md" w="100%">
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
                {data?.game.ownerUsername}
              </Text>
              {data?.game.ownerUsername && (
                <Badge color="gray" variant="light">
                  Owner: {data?.game.ownerUsername} {iAmOwner && "(You)"}
                </Badge>
              )}
            </Flex>

            <Button variant="light" color="blue" onClick={goBackToGamesList}>
              Back
            </Button>
          </Flex>
        </Paper>
      </Flex>

      <Flex w="100%" gap="xs" direction="column" align="center">
        <Text size="xs" c="dimmed">
          Use ← → arrows to navigate moves
        </Text>
        <ScrollArea h={entryHeight * 3 + 20} w="100%" type="always">
          <Box
            pos="relative"
            w={(data?.maxTurns || 0) * entryWidth}
            h={entryHeight * 3}
          >
            {(data?.history || []).map((entry, index) => (
              <Flex
                left={(entry.turn_counter - 1) * entryWidth}
                top={playerColorToIndex(entry.color) * entryHeight}
                w={entryWidth}
                h={entryHeight}
                pos="absolute"
                key={index}
                direction="column"
                align="center"
                gap="xs"
                style={(t) => ({
                  border: `1px solid`,
                  borderColor: t.colors.gray[1],
                  backgroundColor:
                    index === currentIndex ? t.colors.blue[0] : "transparent",
                  cursor: "pointer",
                })}
                onClick={() => {
                  setCurrentIndex(index);

                  if (data) {
                    setCurrentFen(data.history[index].fen);
                  }
                }}
                justify="center"
              >
                <Text ff="monospace" size="sm">
                  {entry.move_uci}
                </Text>
              </Flex>
            ))}
          </Box>
        </ScrollArea>
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
        {currentFen && (
          <BoardComponent
            showPatterns={true}
            fenOutside={currentFen}
            players={players}
            hideMaterial={false}
            rotation={0}
            color={null}
            gameStatus={GameStatus.FinishedWin}
            onMovePlayed={() => {}}
            highlightedMoves={[]}
          />
        )}
      </Flex>
    </Flex>
  );
};

// type EndGameShowcaseProps = {
//   players: Players | null;
// };
//
// const relationToColor = (relation: GameRelation) => {
//   switch (relation) {
//     case "winner":
//       return "#10d335";
//     case "loser":
//       return "#e50718";
//     case "draw":
//       return "#e0b00d";
//     default:
//       return "gray";
//   }
// };
//
// const relationToIcon = (relation: GameRelation): JSX.Element => {
//   switch (relation) {
//     case "winner":
//       return <IconTrophy />;
//     case "loser":
//       return <IconThumbDown />;
//     case "draw":
//       return <IconTie />;
//     default:
//       return <IconTie />;
//   }
// };
//
// const EndGameShowcase = ({ players }: EndGameShowcaseProps) => {
//   if (!players) return <></>;
//
//   return (
//     <Flex direction="column" align="center" gap="md">
//       <Text>The game has ended! Here are the final players:</Text>
//       <Flex gap="sm">
//         {[Color.White, Color.Gray, Color.Black].map((color) => {
//           const player = players[color];
//           if (!player) return <></>;
//
//           return (
//             <Paper
//               key={color}
//               p="sm"
//               withBorder
//               radius="md"
//               bg="gray.0"
//               style={{ borderColor: relationToColor(player.relation!) }}
//             >
//               <Flex key={color} direction="column" align="center" gap="2px">
//                 <Badge color="gray">{libColorToHumanString(color)}</Badge>
//                 <Text>{player ? player.username : "No player"}</Text>
//                 <Text c={relationToColor(player.relation!)}>
//                   {player.relation || "No result"}
//                 </Text>
//                 <ThemeIcon
//                   color={relationToColor(player.relation!)}
//                   variant="transparent"
//                   size="lg"
//                 >
//                   {relationToIcon(player.relation!)}
//                 </ThemeIcon>
//               </Flex>
//             </Paper>
//           );
//         })}
//       </Flex>
//     </Flex>
//   );
// };

export default GameHistoryPage;
