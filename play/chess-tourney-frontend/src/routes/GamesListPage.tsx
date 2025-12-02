import {
  GameFilterParameter,
  type GameStatus,
  type Player,
  type PlayerColor,
} from "../api/model";
import {
  useGamesCreateGame,
  useGamesJoinGame,
  useGamesListGames,
} from "../api/game-management/game-management.ts";
import { showNotification } from "@mantine/notifications";
import {
  ActionIcon,
  Badge,
  Button,
  Container,
  Flex,
  Grid,
  type MantineColor,
  Modal,
  NavLink,
  NumberInput,
  Paper,
  Select,
  Skeleton,
  Switch,
  Text,
  TextInput,
  ThemeIcon,
  Title,
  Tooltip,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { useForm } from "@mantine/form";
import {
  IconAccessible,
  IconChess,
  IconChessFilled,
  IconChevronRight,
  IconDeviceTvOld,
  IconEye,
  IconEyeX,
  IconFilter,
  IconHistory,
  IconPlus,
} from "@tabler/icons-react";
import { useNavigate } from "@tanstack/react-router";
import { gameDetailRoute, gameHistoryRoute, gamesRoute } from "../routes.tsx";
import Navbar from "../components/Navbar.tsx";
import { isGameOver } from "../common.ts";
import { useState } from "react";
import { useTournamentsListTournaments } from "../api/tournament-management/tournament-management.ts";

const gameStatusToBadgeColor = (status: GameStatus): MantineColor => {
  switch (status) {
    case "Waiting":
      return "#868e96";
    case "InProgress":
      return "#1971c2";
    case "FinishedWin":
      return "#2f9e44";
    case "FinishedDraw":
      return "#f76707";
    case "FinishedSemiDraw":
      return "#fcc419";
  }
};

export const GamesListPage = () => {
  const [filterGames, setFilterGames] = useState<
    GameFilterParameter | undefined
  >(undefined);

  const [tournamentIdFilter, setTournamentIdFilter] = useState<
    number | undefined
  >(undefined);

  const {
    data,
    isLoading,
    refetch: refetchGamesApi,
  } = useGamesListGames({
    status: filterGames,
    tournamentId: tournamentIdFilter,
  });

  const [opened, { open, close }] = useDisclosure(false);

  const { mutateAsync: createGameApi } = useGamesCreateGame();
  const { mutateAsync: joinGameApi } = useGamesJoinGame();

  const { data: tournamentData } = useTournamentsListTournaments();

  const navigate = useNavigate({
    from: gamesRoute.id,
  });

  const form = useForm({
    initialValues: {
      gameName: "",
      suggestedMoveTimeSeconds: 30,
      hidePlayerNames: false,
      hideMaterial: false,
      tournamentId: undefined,
      trainingMode: false,
    },
    validate: {
      gameName: (value) =>
        value.trim().length > 0 ? null : "Game name is required",
    },
  });

  const createGame = async (values: typeof form.values) => {
    try {
      const newGame = await createGameApi({
        data: {
          name: values.gameName,
          material_masked: values.hideMaterial,
          names_masked: values.hidePlayerNames,
          suggested_move_time_seconds:
            values.suggestedMoveTimeSeconds == 0
              ? undefined
              : values.suggestedMoveTimeSeconds,
          tournamentId: values.tournamentId
            ? parseInt(values.tournamentId)
            : undefined,
          training_mode: values.trainingMode,
        },
      });
      showNotification({
        title: "Game Created",
        message: `Game "${newGame.name}" created successfully.`,
        color: "green",
      });
      await refetchGamesApi();
      close();
      form.reset();
    } catch (error: any) {}
  };

  const joinGame = async (gameId: number, color: PlayerColor) => {
    try {
      await joinGameApi({
        gameId: gameId,
        data: {
          color: color,
        },
      });

      showNotification({
        title: "Joined Game",
        message: `Successfully joined game with ID "${gameId}".`,
        color: "green",
      });

      await goToRoom(gameId);
    } catch (error: any) {}
  };

  const goToRoom = async (gameId: number) => {
    await navigate({
      to: gameDetailRoute.id,
      params: { id: gameId.toString() },
    });
  };

  const goToHistory = async (gameId: number) => {
    await navigate({
      to: gameHistoryRoute.id,
      params: { id: gameId.toString() },
    });
  };

  return (
    <Flex w="100%" mih="100vh" bg="gray.0" direction="column">
      <Navbar />
      <Modal opened={opened} onClose={close} title="Create New Game">
        <form onSubmit={form.onSubmit((values) => createGame(values))}>
          <Flex direction="column" gap="sm">
            <TextInput
              label="Game Name"
              required
              {...form.getInputProps("gameName")}
              mr="lg"
            />
            <NumberInput
              label="Move Time (seconds)"
              description="Just a suggestion on how much time each player should take per move."
              {...form.getInputProps("suggestedMoveTimeSeconds")}
              mr="lg"
              min={0}
            />

            <Select
              label="Tournament (optional)"
              placeholder="Select a tournament"
              data={
                tournamentData
                  ? tournamentData.map((tournament) => ({
                      value: tournament.id.toString(),
                      label: tournament.name,
                    }))
                  : []
              }
              {...form.getInputProps("tournamentId")}
              clearable
              mr="lg"
            />

            <Switch
              description="Players' names will be hidden during the game. Zero bias mode (somewhat)"
              label="Hide Names"
              {...form.getInputProps("hidePlayerNames", { type: "checkbox" })}
              mr="lg"
            />
            <Switch
              label="Hide Material"
              description="Players' material counts will be hidden during the game."
              {...form.getInputProps("hideMaterial", { type: "checkbox" })}
            />
            <Switch
              label="Training Mode"
              description="In training mode, players can hover over all pieces to see their possible moves."
              {...form.getInputProps("trainingMode", { type: "checkbox" })}
            />
            <Button type="submit">Create New Game</Button>
          </Flex>
        </form>
      </Modal>
      <Container size="xl" w="100%" py="lg">
        <Paper withBorder radius="sm" bg="gray.1" p="lg" mb="lg">
          <Flex w="100%" justify="space-between" align="center">
            <Flex gap="sm" wrap="wrap">
              <Select
                value={filterGames}
                onChange={(value) =>
                  setFilterGames(
                    value ? (value as GameFilterParameter) : undefined,
                  )
                }
                placeholder="Filter by Status"
                data={[
                  { value: "Waiting", label: "Waiting" },
                  { value: "InProgress", label: "In Progress" },
                  { value: "Finished", label: "Finished" },
                ]}
                clearable
                leftSection={<IconFilter />}
                style={{ minWidth: 200 }}
              />
              <Select
                value={
                  tournamentIdFilter ? tournamentIdFilter.toString() : undefined
                }
                onChange={(value) =>
                  setTournamentIdFilter(value ? parseInt(value) : undefined)
                }
                placeholder="Filter by Tournament"
                data={
                  tournamentData
                    ? tournamentData.map((tournament) => ({
                        value: tournament.id.toString(),
                        label: tournament.name,
                      }))
                    : []
                }
                clearable
                leftSection={<IconFilter />}
                style={{ minWidth: 200 }}
              />
            </Flex>
            <ActionIcon onClick={open} variant="filled" color="blue" size="lg">
              <IconPlus />
            </ActionIcon>
          </Flex>
        </Paper>

        <Grid>
          {isLoading &&
            Array.from({ length: 12 }).map((_, index) => (
              <Grid.Col
                key={index}
                span={{
                  xs: 12,
                  sm: 6,
                  md: 4,
                  lg: 3,
                }}
              >
                <Skeleton height={200} radius="sm" />
              </Grid.Col>
            ))}

          {(data || []).map((game) => (
            <Grid.Col
              key={game.gameId}
              span={{
                xs: 12,
                sm: 6,
                md: 4,
                lg: 3,
              }}
              style={{ display: "flex" }}
            >
              <Paper
                withBorder
                radius="sm"
                bg="white"
                p="lg"
                h="100%"
                w="100%"
                pos="relative"
                style={(_) => ({
                  borderColor: gameStatusToBadgeColor(game.status),
                })}
              >
                <Flex pos="absolute" top="-10px" left="10px">
                  <Badge color={gameStatusToBadgeColor(game.status)}>
                    {game.status}
                  </Badge>
                </Flex>
                <Flex direction="column" pos="relative" w="100%">
                  <Flex gap="sm" align="center" justify="space-between">
                    <Flex align="baseline" gap="2">
                      <Title order={6}>{game.name}</Title>
                      <Text size="xs" c={"gray.6"}>
                        #{game.gameId}
                      </Text>
                    </Flex>

                    {!isGameOver(game.status) ? (
                      <ActionIcon
                        variant="light"
                        color="green"
                        onClick={() => goToRoom(game.gameId)}
                      >
                        <IconDeviceTvOld />
                      </ActionIcon>
                    ) : (
                      <ActionIcon
                        variant="light"
                        color="red"
                        onClick={() => goToHistory(game.gameId)}
                      >
                        <IconHistory />
                      </ActionIcon>
                    )}
                  </Flex>

                  <Flex direction="column" mt="sm" gap="xs">
                    <PlayerStatus
                      status={game.players?.white || null}
                      color="White"
                      id={game.gameId}
                      fn={joinGame}
                    />
                    <PlayerStatus
                      status={game.players?.grey || null}
                      color="Grey"
                      id={game.gameId}
                      fn={joinGame}
                    />
                    <PlayerStatus
                      status={game.players?.black || null}
                      color="Black"
                      id={game.gameId}
                      fn={joinGame}
                    />
                  </Flex>

                  <Flex mt="sm" justify="flex-end" gap="xs">
                    <Tooltip
                      label={
                        game.players_masked
                          ? "Player names are hidden"
                          : "Player names are visible"
                      }
                    >
                      <ThemeIcon
                        variant="subtle"
                        color={game.players_masked ? "gray" : "orange"}
                      >
                        {game.players_masked ? <IconEyeX /> : <IconEye />}
                      </ThemeIcon>
                    </Tooltip>

                    <Tooltip
                      label={
                        game.material_masked
                          ? "Material counts are hidden"
                          : "Material counts are visible"
                      }
                    >
                      <ThemeIcon
                        variant="subtle"
                        color={game.material_masked ? "gray" : "blue"}
                      >
                        <IconChess />
                      </ThemeIcon>
                    </Tooltip>

                    <Tooltip
                      label={
                        game.training_mode
                          ? "Training mode enabled"
                          : "Training mode disabled"
                      }
                    >
                      <ThemeIcon
                        variant="subtle"
                        color={game.training_mode ? "blue" : "gray"}
                      >
                        <IconAccessible />
                      </ThemeIcon>
                    </Tooltip>
                  </Flex>
                </Flex>
              </Paper>
            </Grid.Col>
          ))}
        </Grid>
      </Container>
    </Flex>
  );
};
type PlayerStatusProps = {
  status: Player | null;
  color: PlayerColor;
  fn: (id: number, c: PlayerColor) => void;
  id: number;
};
const PlayerStatus = ({ status, color, id, fn }: PlayerStatusProps) => {
  let icon;

  switch (color) {
    case "White":
      icon = <IconChess />;
      break;
    case "Grey":
      icon = <IconChess color="grey" />;
      break;
    case "Black":
      icon = <IconChessFilled />;
      break;
  }

  return (
    <Flex justify="center" w="100%" align="center">
      {icon}
      <NavLink
        variant="filled"
        label={!status ? `Join as ${color} player` : status.username}
        onClick={() => fn(id, color)}
        disabled={!!status}
        rightSection={
          !status && (
            <IconChevronRight
              size={12}
              stroke={1.5}
              className="mantine-rotate-rtl"
            />
          )
        }
      ></NavLink>
    </Flex>
  );
};
