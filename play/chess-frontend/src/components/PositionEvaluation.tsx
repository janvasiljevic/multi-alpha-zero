import { ThinkResultForChessMoveWrapperDtoAndChessBoardDto } from "../api/def/model";
import { Badge, Box, Flex, Paper, Text } from "@mantine/core";
import { getChessPlayerColorByIndex } from "../../misc/util.ts";

type Props = {
  thinkResult: ThinkResultForChessMoveWrapperDtoAndChessBoardDto | null;
  isLoading: boolean;
};

const PositionEvaluation = ({ thinkResult, isLoading }: Props) => {
  const evaluation = thinkResult ? thinkResult.root_position_eval : [0, 0, 0];
  const showEval = !!thinkResult;

  return (
    <>
      <Badge
        variant="dot"
        color="grape"
        pos="absolute"
        top={showEval ? -100 : 0}
        style={{ transition: "opacity 0.3s, top 0.3s" }}
        opacity={showEval ? 0.0 : 1.0}
      >
        Please evaluate the position
      </Badge>
      <Paper
        withBorder
        p="2px"
        px="lg"
        bg={isLoading ? "gray.1" : "white"}
        pos="absolute"
        top={showEval ? 0 : -100}
        style={{ transition: "opacity 0.3s, top 0.3s", borderRadius: "50px" }}
        opacity={showEval ? 1.0 : 0.0}
      >
        <Flex w="100%" justify="space-between" gap="md">
          {[0, 1, 2].map((n) => (
            <Flex justify="center" align="center" gap="xs" key={n}>
              <Box
                style={{
                  borderRadius: "50%",
                  border: "1px solid black",
                  width: "12px",
                  height: "12px",
                  background: getChessPlayerColorByIndex(n),
                }}
              ></Box>
              <Text>{evaluation[n].toFixed(2)}</Text>
            </Flex>
          ))}
        </Flex>
      </Paper>
    </>
  );
};
export default PositionEvaluation;
