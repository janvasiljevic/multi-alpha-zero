import React from "react";
import type { BoardState, HexPlayer } from "../api/def/model";
import { Flex, Text } from "@mantine/core";
import { getPlayerColor } from "../misc/util.ts";

interface GameInfoProps {
  gameState: BoardState;
}

function GameInfo({
  gameState: { current_turn, eliminated_players, winner },
}: GameInfoProps) {
  const getPlayerStyle = (player: HexPlayer): React.CSSProperties => {
    return {
      color: getPlayerColor(player),
      fontWeight: "bold",
    };
  };

  const renderStatus = () => {
    if (winner) {
      return (
        <Text>
          <span style={getPlayerStyle(winner)}>{winner}</span> wins!
        </Text>
      );
    }
    return (
      <Text>
        It's <span style={getPlayerStyle(current_turn)}>{current_turn}</span>{" "}
        turn.
      </Text>
    );
  };

  return (
    <Flex direction="column" justify="center" align="center">
      {renderStatus()}
      <Flex>
        {eliminated_players.length > 0 && (
          <>
            {eliminated_players.map((player) => (
              <Text key={player} span style={getPlayerStyle(player)} pr="xs">
                {player}
              </Text>
            ))}
            <Text span>eliminated</Text>
          </>
        )}
      </Flex>
    </Flex>
  );
}

export default React.memo<GameInfoProps>(GameInfo);
