import type { ReadBot, UpdateBot } from "../api/model";
import React from "react";
import {
  Autocomplete,
  Button,
  Container,
  Flex,
  Grid,
  Group,
  NumberInput,
  Paper,
  Skeleton,
  TextInput,
  Title,
} from "@mantine/core";
import Navbar from "../components/Navbar.tsx";
import {
  useBotsCreate,
  useBotsDelete,
  useBotsListBots,
  useBotsModelKeys,
  useBotsUpdate,
} from "../api/bot-management/bot-management.ts";
import { useForm } from "@mantine/form";
import { showNotification } from "@mantine/notifications";
import { useTournamentsCreateTournament } from "../api/tournament-management/tournament-management.ts";

export interface CreateBot {
  username: string;
  model_key?: string;
  playouts_per_move: number;
  exploration_factor: number;
  virtual_loss: number;
  contempt: number;
  password: string;
}

const GridItem = ({ children }: { children: React.ReactNode }) => {
  return (
    <Grid.Col
      span={{
        xs: 12,
        sm: 6,
        md: 6,
        lg: 4,
      }}
      style={{
        display: "flex",
        alignItems: "start"
      }}
    >
      {children}
    </Grid.Col>
  );
};

const BotItem = ({
  bot,
  modelsKey,
  updateBot,
  refetch,
  deleteBot,
}: {
  bot: ReadBot;
  modelsKey: string[];
  updateBot: (values: UpdateBot, botId: number) => Promise<void>;
  deleteBot: (botId: number) => Promise<void>;
  refetch: () => void;
}) => {
  const form = useForm<CreateBot>({
    initialValues: {
      username: bot.username,
      model_key: bot.model_key,
      playouts_per_move: bot.playouts_per_move,
      exploration_factor: bot.exploration_factor,
      virtual_loss: bot.virtual_loss,
      contempt: bot.contempt,
      password: "",
    },
  });

  const handleEdit = async (values: UpdateBot) => {
    await updateBot(values, bot.bot_id);
    refetch();
  };

  return (
    <Paper
      withBorder
      radius="sm"
      bg="gray.0"
      p="lg"
      h="100%"
      w="100%"
      pos="relative"
    >
      <form onSubmit={form.onSubmit(handleEdit)}>
        <Flex direction="column" pos="relative" w="100%">
          <Flex gap="sm" align="center" justify="space-between">
            <Flex align="baseline" gap="2">
              <Title order={4}>{bot.username || "Unnamed Bot"}</Title>
            </Flex>
          </Flex>
          <Grid>
            <GridItem>
              <TextInput w="100%" label="Bot ID" disabled value={bot.bot_id} />
            </GridItem>
            <GridItem>
              <TextInput
                w="100%"
                label="User ID"
                disabled
                value={bot.user_id}
              />
            </GridItem>

            <GridItem>
              <TextInput
                w="100%"
                label="Username"
                {...form.getInputProps("username")}
              />
            </GridItem>
            <GridItem>
              <Autocomplete
                w="100%"
                label="Model Key"
                data={modelsKey || []}
                {...form.getInputProps("model_key")}
              />
            </GridItem>
            <GridItem>
              <NumberInput
                w="100%"
                label="Playouts Per Move"
                {...form.getInputProps("playouts_per_move")}
              />
            </GridItem>
            <GridItem>
              <NumberInput
                w="100%"
                label="Exploration Factor"
                {...form.getInputProps("exploration_factor")}
              />
            </GridItem>
            <GridItem>
              <NumberInput
                w="100%"
                label="Virtual Loss"
                {...form.getInputProps("virtual_loss")}
              />
            </GridItem>
            <GridItem>
              <NumberInput
                w="100%"
                label="Contempt"
                {...form.getInputProps("contempt")}
              />
            </GridItem>
          </Grid>
          <Group w="100%" justify="flex-end" mt="md">
            <Button
              onClick={() => deleteBot(bot.bot_id)}
              color="red"
              variant="outline"
            >
              Delete
            </Button>
            <Button type="submit">Edit</Button>
          </Group>
        </Flex>
      </form>
    </Paper>
  );
};

export const BotsListPage = () => {
  const { data, isLoading, refetch: refetchGamesApi } = useBotsListBots();
  const { data: modelsKey } = useBotsModelKeys();
  const { mutateAsync: createBot } = useBotsCreate();
  const { mutateAsync: updateBot } = useBotsUpdate();
  const { mutateAsync: deleteBot } = useBotsDelete();

  const { mutateAsync: createTournament } = useTournamentsCreateTournament();
  const [tournamentName, setTournamentName] = React.useState("");

  const form = useForm<CreateBot>({
    initialValues: {
      username: "",
      model_key: "",
      playouts_per_move: 100,
      exploration_factor: 1.4,
      virtual_loss: 3,
      contempt: 0,
      password: "",
    },
  });

  const handleFormSubmit = async (values: CreateBot) => {
    try {
      await createBot({
        data: {
          ...values,
          model_key:
            (values.model_key?.length || 0) > 0 ? values.model_key : undefined,
        },
      });
      await refetchGamesApi();
    } catch (error) {
      if ((error as any).response?.status === 400) {
        console.log(error);
        showNotification({
          title: "Error",
          message: (error as any).response || "Bad Request",
          color: "red",
        });
        return;
      } else {
        showNotification({
          title: "Error",
          message: "An unexpected error occurred",
          color: "red",
        });
      }
    }
    form.reset();
  };

  const handleEdit = async (values: UpdateBot, botId: number) => {
    try {
      await updateBot({
        botId,
        data: {
          ...values,
          model_key:
            (values.model_key?.length || 0) > 0 ? values.model_key : undefined,
        },
      });
      showNotification({
        title: "Success",
        message: "Bot updated successfully",
        color: "green",
      });
    } catch (error) {
      if ((error as any).response?.status === 400) {
        showNotification({
          title: "Error",
          message: (error as any).response?.data || "Bad Request",
          color: "red",
        });
      } else {
        showNotification({
          title: "Error",
          message: "An unexpected error occurred",
          color: "red",
        });
      }
    }
  };

  const handleDelete = async (botId: number) => {
    try {
      await deleteBot({ botId });
      showNotification({
        title: "Success",
        message: "Bot deleted successfully",
        color: "green",
      });
      await refetchGamesApi();
    } catch (error) {
      showNotification({
        title: "Error",
        message: "An unexpected error occurred",
        color: "red",
      });
    }
  };

  return (
    <Flex w="100%" mih="100vh" bg="gray.0" direction="column">
      <Navbar />
      <Container size="xl" w="100%" py="lg">
        <Paper withBorder radius="sm" bg="gray.1" p="lg" mb="lg">
          <Grid w="100%">
            <GridItem>
              <TextInput
                w="100%"
                label="Tournament Name"
                value={tournamentName}
                onChange={(e) => setTournamentName(e.currentTarget.value)}
              />
            </GridItem>
            <GridItem>
              <Button
                mt="xl"
                onClick={async () => {
                  try {
                    await createTournament({
                      data: {
                        name: tournamentName,
                      },
                    });
                    showNotification({
                      title: "Success",
                      message: "Tournament created successfully",
                      color: "green",
                    });
                    setTournamentName("");
                  } catch (error) {
                    showNotification({
                      title: "Error",
                      message: "An unexpected error occurred",
                      color: "red",
                    });
                  }
                }}
              >
                Create Tournament
              </Button>
            </GridItem>
          </Grid>
        </Paper>

        <Paper withBorder radius="sm" bg="gray.1" p="lg" mb="lg">
          <form onSubmit={form.onSubmit(handleFormSubmit)}>
            <Flex direction="column" gap="md">
              <Grid w="100%">
                <GridItem>
                  <TextInput
                    w="100%"
                    label="Bot Username"
                    {...form.getInputProps("username")}
                  />
                </GridItem>
                <GridItem>
                  <Autocomplete
                    w="100%"
                    label="Model Key"
                    data={modelsKey || []}
                    {...form.getInputProps("model_key")}
                  />
                </GridItem>
                <GridItem>
                  <NumberInput
                    w="100%"
                    label="Playouts Per Move"
                    {...form.getInputProps("playouts_per_move")}
                  />
                </GridItem>
                <GridItem>
                  <NumberInput
                    w="100%"
                    label="Exploration Factor"
                    {...form.getInputProps("exploration_factor")}
                  />
                </GridItem>
                <GridItem>
                  <NumberInput
                    w="100%"
                    label="Virtual Loss"
                    {...form.getInputProps("virtual_loss")}
                  />
                </GridItem>
                <GridItem>
                  <NumberInput
                    w="100%"
                    label="Contempt"
                    {...form.getInputProps("contempt")}
                  />
                </GridItem>
                <GridItem>
                  <TextInput
                    w="100%"
                    label="Password"
                    type="password"
                    {...form.getInputProps("password")}
                  />
                </GridItem>
              </Grid>
              <Group w="100%" justify="flex-end">
                <Button type="submit">Create Bot</Button>
              </Group>
            </Flex>
          </form>
        </Paper>

        <Flex w="100%" gap="md" wrap="wrap">
          {isLoading &&
            Array.from({ length: 12 }).map((_, index) => (
              <Skeleton height={200} radius="sm" key={index} />
            ))}

          {(data || []).map((bot) => (
            <BotItem
              key={bot.bot_id}
              bot={bot}
              modelsKey={modelsKey || []}
              updateBot={handleEdit}
              deleteBot={handleDelete}
              refetch={refetchGamesApi}
            />
          ))}
        </Flex>
      </Container>
    </Flex>
  );
};
