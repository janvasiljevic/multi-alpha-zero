import {
  Alert,
  Badge,
  Button,
  Flex,
  Group,
  Modal,
  Paper,
  Select,
  Text,
} from "@mantine/core";
import { showNotification } from "@mantine/notifications";
import type { Players } from "./Board.tsx";
import { Color } from "../../libs/wasm";
import { libColorToHumanString } from "../class/utils.ts";
import { useState } from "react";
import { useUsersListBotUsers } from "../api/user-management/user-management.ts";
import { useBotsAssignBot } from "../api/bot-management/bot-management.ts";
import type { PlayerColor } from "../api/model";

type Props = {
  opened: boolean;
  onClose: () => void;
  gameId: number;
  players: Players | null;
};

const AddBotToGameModal = ({ opened, onClose, gameId, players }: Props) => {
  const { data: botData } = useUsersListBotUsers({
    query: {
      enabled: opened,
    },
  });

  const { mutateAsync: addBotToGameApi } = useBotsAssignBot();

  const [botId, setBotId] = useState<number | null>(null);

  const availableSpots = players
    ? Object.values(players).filter((p) => p == null).length
    : 0;
  const [selectedColor, setSelectedColor] = useState<Color | null>(null);

  const handleSubmit = async () => {
    if (botId == null || selectedColor == null) {
      showNotification({
        title: "Error",
        message: "Please select both a bot and a color slot.",
      });

      return;
    }

    try {
      let color: PlayerColor = "White";

      switch (selectedColor) {
        case Color.White:
          color = "White";
          break;
        case Color.Gray:
          color = "Grey";
          break;
        case Color.Black:
          color = "Black";
          break;
        default:
          throw new Error("Invalid color selected");
      }

      await addBotToGameApi({
        data: {
          botId: botId,
          gameId: gameId,
          color: color,
        },
      });
      showNotification({
        title: "Success",
        message: "Game ended successfully.",
        color: "green",
      });
      onClose();
    } catch (error: any) {
      showNotification({
        title: "Error",
        message:
          error.response?.data?.message ||
          "An unexpected error occurred",
        color: "red",
      });
    }
  };
  if (players == null) {
    return <></>;
  }

  return (
    <Modal opened={opened} onClose={onClose} title="Add Bot to Game">
      <Flex direction="column" gap="md">
        {availableSpots == 0 && (
          <Alert>
            No available spots to add a bot. Please make sure at least one
            player slot is empty.
          </Alert>
        )}

        <Text> Select a color slot for the bot</Text>
        <Flex gap="sm">
          {[Color.White, Color.Gray, Color.Black].map((color) => {
            const player = players[color];

            return (
              <Paper
                key={color}
                p="sm"
                withBorder
                radius="md"
                bg="gray.0"
                style={{
                  borderColor: selectedColor === color ? "blue" : "",
                  cursor: player == null ? "pointer" : "default",
                }}
                onClick={() => {
                  if (player == null) {
                    setSelectedColor(color);
                  }
                }}
              >
                <Flex key={color} direction="column" align="center" gap="2px">
                  <Badge color="gray">{libColorToHumanString(color)}</Badge>
                  <Text>{player ? player.username : "Spot available"}</Text>
                </Flex>
              </Paper>
            );
          })}
        </Flex>

        {botData != null && botData.length === 0 && (
          <Alert>No bots available. Please create a bot user first.</Alert>
        )}

        <Select
          label="Select Bot"
          placeholder="Select a bot to add"
          data={
            botData
              ? botData.map((bot) => ({
                  value: bot.id.toString(),
                  label: bot.username,
                }))
              : []
          }
          value={botId ? botId.toString() : null}
          onChange={(val) => setBotId(val ? parseInt(val) : null)}
        />

        {selectedColor == null && (
          <Alert variant="light" color="orange">
            Please select a color slot to add the bot.
          </Alert>
        )}

        {botId == null && (
          <Alert variant="light" color="orange">
            Please select a bot to add.
          </Alert>
        )}

        <Group mt="md">
          <Button
            variant="outline"
            onClick={onClose}
            disabled={botId == null || selectedColor == null}
          >
            Cancel
          </Button>
          <Button onClick={handleSubmit}>Submit</Button>
        </Group>
      </Flex>
    </Modal>
  );
};

export default AddBotToGameModal;
