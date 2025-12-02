import React from "react";
import type {
  BoardTileState,
  HexPlayer,
  ThinkResultItem,
} from "../api/def/model";
import { Flex, Text } from "@mantine/core";

interface HexCellProps {
  q: number;
  r: number;
  cellState: BoardTileState;
  onClick: (q: number, r: number) => void;
  eliminatedPlayers: HexPlayer[];
  thinkResult: ThinkResultItem | null;
}

function HexCell({
  q,
  r,
  cellState,
  onClick,
  eliminatedPlayers,
  thinkResult,
}: HexCellProps) {
  const handleClick = () => {
    if (cellState === null) {
      onClick(q, r);
    }
  };

  const isEliminated =
    cellState !== null && eliminatedPlayers.includes(cellState);



  return (
    <div className="hex-cell" onClick={handleClick}>
      {cellState !== null && (
        <div
          className={`hex-player ${cellState} ${isEliminated ? "eliminated" : ""}`}
        >
          {cellState}
        </div>
      )}
      {thinkResult !== null && (
        <Flex pos="absolute" align="center" justify="center" w="100%" h="100%">
          <Text
            style={{
              color: `color-mix(in hsl, green ${thinkResult.renormalized_score * 100}%, red ${(1 - thinkResult.renormalized_score) * 100}%)`,
              fontWeight: "bold",
              fontSize: "1.2rem",
            }}
          >
            {(thinkResult.score * 100).toFixed(0)}%
          </Text>
        </Flex>
      )}
      <div className="hex-content" />
    </div>
  );
}

export default React.memo<HexCellProps>(HexCell);
