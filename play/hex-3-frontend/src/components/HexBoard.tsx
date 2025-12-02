import React from "react";
import HexCell from "./HexCell";
import type {
  BoardState,
  ThinkResult,
  ThinkResultItem,
} from "../api/def/model";
import { Box, Flex } from "@mantine/core";

interface HexBoardProps {
  board: BoardState;
  onHexClick: (q: number, r: number) => void;
  thinkResult: ThinkResult | null;
}

function HexBoard({ board, onHexClick, thinkResult }: HexBoardProps) {
  const rows = [];

  const radius = board.radius;


  for (let r = -radius; r <= radius; r++) {
    const cells = [];
    const qStart = Math.max(-radius, -r - radius);
    const qEnd = Math.min(radius, -r + radius);

    for (let q = qStart; q <= qEnd; q++) {
      const key = `${q},${r}`;
      const tile = board.board.find((tile) => tile.q === q && tile.r === r);

      if (!tile || tile.state === undefined) {
        console.error(`No tile found for coordinates (${q}, ${r})`);
        continue;
      }

      let scoreResult: null | ThinkResultItem = null;

      if (thinkResult) {
        scoreResult =
          thinkResult.moves.find((item) => item.q === q && item.r === r) ||
          null;
      }

      cells.push(
        <HexCell
          key={key}
          q={q}
          r={r}
          thinkResult={scoreResult}
          cellState={tile.state}
          eliminatedPlayers={board.eliminated_players}
          onClick={onHexClick}
        />,
      );
    }

    rows.push(
      <Flex key={r} justify="center" align="flex-end" mb="-15px">
        {cells}
      </Flex>,
    );
  }

  return (
    <>
      <Box pos="relative" p="sm">
        <div className="line line-p1-start" />
        <div className="line line-p1-end" />

        <div className="line line-p2-start" />
        <div className="line line-p2-end" />

        <div className="line line-p3-start" />
        <div className="line line-p3-end" />

        {rows}
      </Box>
    </>
  );
}

// Memoize the component with explicit prop types for performance.
export default React.memo<HexBoardProps>(HexBoard);
