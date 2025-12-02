import { useCallback, useEffect, useState } from "react";
import HexBoard from "./components/HexBoard.tsx";
import {
  useGetHexDefaultRadius,
  usePostHexMakeMove,
  usePostHexThink,
} from "./api/def/default/default.ts";
import type { BoardState, ThinkResult } from "./api/def/model";
import GameInfo from "./components/GameInfo.tsx";
import { Button, Container, Flex, Group, Select, Text } from "@mantine/core";
import ThinkResultInfo from "./components/ThinkResultInfo.tsx";
import { useDisclosure } from "@mantine/hooks";
import { ThinkTreeDrawer } from "./components/ThinkTreeDrawer.tsx";

function App() {
  const [boardState, setBoardState] = useState<BoardState | null>(null);
  const [thinkResult, setThinkResult] = useState<ThinkResult | null>(null);

  const [modelKey, setModelKey] = useState<string>("CanonicalHex4");

  const [opened, { open: openDrawer, close: closeDrawer }] =
    useDisclosure(false);

  const { data: gameStateData, isLoading } = useGetHexDefaultRadius(4);

  const { mutateAsync: commitMoveApi } = usePostHexMakeMove();
  const { mutateAsync: thinkApi, isPending: isLoadingThink } =
    usePostHexThink();

  useEffect(() => {
    if (gameStateData) {
      setBoardState(gameStateData);
    }
  }, [gameStateData]);

  const handleHexClick = useCallback(
    async (q: number, r: number) => {
      if (isLoading || !boardState) {
        return;
      }

      try {
        const updatedGameState = await commitMoveApi({
          data: {
            board: boardState,
            coord: {
              q,
              r,
            },
          },
        });
        setBoardState(updatedGameState);
        setThinkResult(null);
      } catch (error) {
        console.error("Error making move:", error);
      }
    },
    [boardState, isLoading],
  );

  const thinkCallback = useCallback(
    async (numberOfRollouts: number) => {
      if (isLoading || !boardState) {
        return;
      }

      try {
        const thinkResult = await thinkApi({
          data: {
            board_state: boardState,
            number_of_rollouts: numberOfRollouts,
            key: modelKey,
          },
        });

        setThinkResult(thinkResult);
      } catch (error) {
        console.error("Error thinking:", error);
      }
    },
    [boardState, isLoading, modelKey],
  );

  if (!boardState) {
    return <Text>Loading...</Text>;
  }

  return (
    <Flex bg="gray.0" mih="100vh" direction="column" w="100%" align="center" justify="center">
      <ThinkTreeDrawer
        opened={opened}
        onClose={closeDrawer}
        onHexClick={() => {}}
        thinkResult={thinkResult}
      />

      <Container size="xl">
        <Flex direction="column" align="center" gap="2rem" mt="lg">
          <HexBoard
            board={boardState}
            onHexClick={handleHexClick}
            thinkResult={thinkResult}
          />
          <GameInfo gameState={boardState} />
        </Flex>
        <Flex gap="sm" direction="column" my="lg" align="center">
          {/*Ste key */}
          <Select
            label="Model Key"
            value={modelKey}
            onChange={(value) => {
              if (value) {
                setModelKey(value);
              }
            }}
            data={[
              "CanonicalHex4",
              "RelativeCanonicalHex4",
            ]}
            size="md"
            mb="md"
          />

          <Group>
            {[
              ["S", 50],
              ["M", 200],
              ["L", 500],
              ["XL", 1200],
              ["XXL", 5000],
              ["XXXL", 10000],
            ].map(([label, rollouts]) => (
              <Button
                key={label}
                onClick={() => thinkCallback(rollouts as number)}
                variant="light"
                size="sm"
                loading={isLoadingThink}
              >
                {label} ({rollouts})
              </Button>
            ))}
          </Group>
          <Button
            onClick={openDrawer}
            variant="light"
            size="sm"
            loading={isLoadingThink}
          >
            Show Think Tree
          </Button>
          <ThinkResultInfo results={thinkResult} />
        </Flex>
      </Container>
    </Flex>
  );
}

export default App;
