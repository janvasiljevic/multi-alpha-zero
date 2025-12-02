import { Button, Flex, Group, Modal, Select } from "@mantine/core";
import { GameRelation } from "../api/model";
import { useState } from "react";
import { useGamesFinishGameIn } from "../api/game-management/game-management.ts";
import { showNotification } from "@mantine/notifications";

type Props = {
  opened: boolean;
  onClose: () => void;
  gameId: number;
};

const EndGameModal = ({ opened, onClose, gameId }: Props) => {
  const [outcomeWhite, setOutcomeWhite] = useState<GameRelation | null>(null);
  const [outcomeGrey, setOutcomeGrey] = useState<GameRelation | null>(null);
  const [outcomeBlack, setOutcomeBlack] = useState<GameRelation | null>(null);

  const { mutateAsync } = useGamesFinishGameIn();

  const handleSubmit = async () => {
    if (!outcomeWhite || !outcomeGrey || !outcomeBlack) {
      showNotification({
        title: "Error",
        message: "Please select outcomes for all players.",
      });
      return;
    }

    try {
      await mutateAsync({
        gameId,
        data: {
          black: outcomeBlack,
          grey: outcomeGrey,
          white: outcomeWhite,
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
          "An error occurred while ending the game.",
        color: "red",
      });
    }
  };

  const options = [
    { value: "winner", label: "Winner" },
    { value: "loser", label: "Loser" },
    { value: "draw", label: "Draw" },
  ];

  return (
    <Modal opened={opened} onClose={onClose} title="End game - admin panel">
      <Flex direction="column" gap="md">
        <Select
          label="White player"
          placeholder="Select outcome"
          data={options}
          value={outcomeWhite ?? undefined}
          onChange={(val) => setOutcomeWhite(val as GameRelation)}
        />
        <Select
          label="Grey player"
          placeholder="Select outcome"
          data={options}
          value={outcomeGrey ?? undefined}
          onChange={(val) => setOutcomeGrey(val as GameRelation)}
        />
        <Select
          label="Black player"
          placeholder="Select outcome"
          data={options}
          value={outcomeBlack ?? undefined}
          onChange={(val) => setOutcomeBlack(val as GameRelation)}
        />

        <Group mt="md">
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={!outcomeWhite || !outcomeGrey || !outcomeBlack}
          >
            Submit
          </Button>
        </Group>
      </Flex>
    </Modal>
  );
};

export default EndGameModal;
