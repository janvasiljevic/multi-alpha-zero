import { LeaderboardSortBy } from "../api/model";
import React, { useState } from "react";
import {
  Badge,
  Chip,
  Container,
  Flex,
  Grid,
  Paper,
  Select,
  Skeleton,
  Text,
  ThemeIcon,
} from "@mantine/core";
import Navbar from "../components/Navbar.tsx";
import { useLeaderboardGetLeaderboard } from "../api/leaderboard/leaderboard.ts";
import {
  IconFilter,
  IconMoodHappy,
  IconRobot,
  IconSortDescending,
} from "@tabler/icons-react";
import { useTournamentsListTournaments } from "../api/tournament-management/tournament-management.ts";

const InnerGridItem = ({ children }: { children: React.ReactNode }) => {
  return (
    <Grid.Col
      span={{
        xs: 12,
        sm: 6,
        md: 3,
        lg: 3,
      }}
      style={{ display: "flex" }}
    >
      {children}
    </Grid.Col>
  );
};

export const LeaderBoardPage = () => {
  const [includeBots, setIncludeBots] = React.useState(true);
  const [sortBy, setSortBy] = React.useState<LeaderboardSortBy>("WinRate");

  const [tournamentIdFilter, setTournamentIdFilter] = useState<
    number | undefined
  >(undefined);

  const { data: tournamentData } = useTournamentsListTournaments();

  const { data, isLoading } = useLeaderboardGetLeaderboard({
    includeBots,
    sortBy,
    tournamentId: tournamentIdFilter,
  });

  return (
    <Flex w="100%" mih="100vh" bg="gray.0" direction="column">
      <Navbar />
      <Container size="xl" w="100%" py="lg">
        <Paper withBorder radius="sm" bg="gray.1" p="lg" mb="lg">
          <Flex gap="md" direction="row" wrap="wrap" align="baseline">
            <Chip
              checked={includeBots}
              onChange={setIncludeBots}
              color="orange"
            >
              Include Bots
            </Chip>

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
            <Select
              value={sortBy}
              leftSection={<IconSortDescending />}
              onChange={(value) => setSortBy(value as LeaderboardSortBy)}
              data={[
                { value: "Wins", label: "Wins" },
                { value: "GamesPlayed", label: "Games Played" },
                { value: "WinRate", label: "Win Rate" },
                { value: "LossRate", label: "Loss Rate" },
                { value: "Losses", label: "Losses" },
                { value: "Draws", label: "Draws" },
              ]}
              w={200}
            />
          </Flex>
        </Paper>

        <Flex w="100%" gap="md" wrap="wrap">
          {isLoading &&
            Array.from({ length: 12 }).map((_, index) => (
              <Skeleton height={200} radius="sm" key={index} />
            ))}

          {(data || []).map((entry, index) => (
            <Paper withBorder radius="sm" p="md" key={entry.userId} w={"100%"}>
              <Flex w="100%" direction="row" gap="sm" align="center">
                <Flex direction="column" align="center">
                  <Badge
                    mb="-5px"
                    size="sm"
                    color={entry.isBot ? "orange" : "blue"}
                  >
                    #{index + 1}
                  </Badge>
                  <ThemeIcon
                    size="xl"
                    radius="md"
                    variant="light"
                    color={entry.isBot ? "orange" : "blue"}
                  >
                    {entry.isBot ? <IconRobot /> : <IconMoodHappy />}
                  </ThemeIcon>
                </Flex>
                <Flex direction="column" w="100%">
                  <Text fw="bold" fz="lg" mb="sm">
                    {entry.username} {entry.isBot ? "(Bot)" : ""}
                  </Text>
                  <Grid>
                    <InnerGridItem>
                      <Text>
                        Wins: {entry.wins} ({(entry.winRate * 100).toFixed(0)}%)
                      </Text>
                    </InnerGridItem>
                    <InnerGridItem>
                      <Text>
                        Losses: {entry.losses} (
                        {(entry.lossRate * 100).toFixed(0)}%)
                      </Text>
                    </InnerGridItem>
                    <InnerGridItem>
                      <Text>Draws: {entry.draws}</Text>
                    </InnerGridItem>
                    <InnerGridItem>
                      <Text>Played: {entry.gamesPlayed}</Text>
                    </InnerGridItem>
                  </Grid>
                </Flex>
              </Flex>
            </Paper>
          ))}
        </Flex>
      </Container>
    </Flex>
  );
};
