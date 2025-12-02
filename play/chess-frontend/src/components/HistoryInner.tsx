import { HistoryItem } from "../App.tsx";
import { Box, Flex, Text } from "@mantine/core";

type Props = {
  history: HistoryItem[];
};

const HistoryInner = ({ history }: Props) => {
  const maxTurn = Math.max(...history.map((item) => item.turn_counter));

  const turns: Record<number, HistoryItem[]> = {};

  history.forEach((item) => {
    if (!turns[item.turn_counter]) turns[item.turn_counter] = [];

    turns[item.turn_counter].push(item);
  });

  return (
    <Flex w={300} direction="column" gap="0" mih="0%" style={{ overflowY: "auto", flexGrow: 1 }}>
      {Array.from({ length: maxTurn }, (_, i) => i + 1).map((turn) => {
        const row = turns[turn] || [];
        return (
          <Flex key={turn} gap="xs">
            <Box w="35px" bg="gray.3" style={{ borderRight: "1px solid #ccc" }} pl="4px" pr="2px">
              <Text size="sm" fw="bolder">
                {turn}
              </Text>
            </Box>
            <Flex gap="sm">
              {[0, 1, 2].map((playerIndex) => {
                const move = row.find((item) => item.color === playerIndex);
                return (
                  <Flex key={playerIndex} w="60px" align="center" justify="flex-start">
                    <Text size="xs">{move ? move.lan : ""}</Text>
                  </Flex>
                );
              })}
            </Flex>
          </Flex>
        );
      })}
    </Flex>
  );
};

export default HistoryInner;
