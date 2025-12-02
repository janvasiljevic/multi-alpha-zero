import {
  Alert,
  Badge,
  Flex,
  Modal,
  Paper,
  Text,
  UnstyledButton,
} from "@mantine/core";
import { showNotification } from "@mantine/notifications";
import type { Players } from "./Board.tsx";
import { useUsersRemoveUserFromGame } from "../api/user-management/user-management.ts";
import { Color } from "../../libs/wasm";
import { libColorToHumanString } from "../class/utils.ts";

type Props = {
  opened: boolean;
  onClose: () => void;
  gameId: number;
  players: Players | null;
};

const RemovePlayerModal = ({ opened, onClose, gameId, players }: Props) => {
  const { mutateAsync: removeApi } = useUsersRemoveUserFromGame();

  const handleSubmit = async (gameId: number, userId: number) => {
    try {
      await removeApi({
        gameId,
        userId,
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
          "An error occurred while ending the game.",
        color: "red",
      });
    }
  };

  return (
    <Modal opened={opened} onClose={onClose} title="End game - admin panel">
      <Flex direction="column" gap="md">
        <Alert>
          Are you sure you want to remove this player from the game?
        </Alert>

        <Flex gap="sm" direction="column">
          {[Color.White, Color.Gray, Color.Black].map((color) => {
            const player = players ? players[color] : null;

            return (
              <UnstyledButton
                onClick={async () => {
                  if (player) {
                    await handleSubmit(gameId, player.id);
                  }
                }}
                key={color}
                style={{ width: "100%" }}
                disabled={!player}
              >
                <Paper key={color} p="sm" withBorder radius="md" bg="gray.0">
                  <Flex key={color} direction="column" align="center" gap="2px">
                    <Badge color="gray">{libColorToHumanString(color)}</Badge>
                    <Text>{player ? player.username : "NO PLAYER"}</Text>
                  </Flex>
                </Paper>
              </UnstyledButton>
            );
          })}
        </Flex>
      </Flex>
    </Modal>
  );
};

export default RemovePlayerModal;
