import { Box, Flex, type MantineSize, Text } from "@mantine/core";
import { getPlayerColorByIndex } from "../../misc/util.ts";

type Props = {
  scores: number[];
  size: MantineSize;
};

const ScoresDisplay = ({ scores, size }: Props) => {
  return (
    <Flex direction="row" align="center" justify="center" gap="4px">
      {scores.map((score, index) => {
        const normalized_score = (score + 1) / 2; // Normalize score to [0, 1]

        return (
          <Box
            key={index}
            style={{
              borderTop: `4px solid ${getPlayerColorByIndex(index)}`,
            }}
          >
            <Text
              ff="monospace"
              size={size}
              style={{
                color: `color-mix(in hsl, green ${normalized_score * 100}%, red ${(1 - normalized_score) * 100}%)`,
              }}
            >
              {score >= 0 && "+"}
              {score.toFixed(2)}
            </Text>
          </Box>
        );
      })}
    </Flex>
  );
};
export default ScoresDisplay;
