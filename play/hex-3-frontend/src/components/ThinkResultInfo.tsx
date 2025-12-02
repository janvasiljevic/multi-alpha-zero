import type { ThinkResult } from "../api/def/model";
import { Flex, Text } from "@mantine/core";
import ScoresDisplay from "./ScoresDisplay.tsx";

type Props = {
  results: ThinkResult | null;
};

const ThinkResultInfo = ({ results }: Props) => {
  if (!results) {
    return <Text c="dimmed">No think results yet.</Text>;
  }

  return (
    <Flex direction="column" align="center" gap="md">
      <ScoresDisplay scores={results.root_position_eval} size="lg" />

      <Text c={"dimmed"}>Thinking took {results.duration_ms} ms</Text>
    </Flex>
  );
};

export default ThinkResultInfo;
