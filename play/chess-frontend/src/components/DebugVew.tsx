import { useEffect, useRef, useState } from "react";
import { TriHexChessWrapper } from "../../libs/wasm";
import { Board } from "../class/Board.ts";
import { SIZE } from "../lib/constants.ts";
import { NewPiece } from "../class/NewPiece.ts";
import HexagonComponent from "./HexagonComponent.tsx";
import PieceElement from "./PieceElement.tsx";
import { Box, Button, Flex, Text } from "@mantine/core";
import { usePostChessAttackMaps } from "../api/def/default/default.ts";
import { AttackMapsOut } from "../api/def/model";
import { getPieces } from "../utils.ts";

type Props = {
  fen: string;
  attackMap: boolean[];
};

type AttackMapBoardProps = {
  fen: string;
};

const AttackMapBoard = ({ fen, attackMap }: Props) => {
  const state = useRef<TriHexChessWrapper | null>(null);
  const board = new Board(SIZE);
  const [pieces, setPieces] = useState<NewPiece[]>([]);

  useEffect(() => {
    state.current = TriHexChessWrapper.new(fen);
    setPieces(getPieces(state.current, 0));
  }, []);

  return (
    <svg viewBox={board.getViewbox()} width="100%" height={"100%"}>
      <g transform={`rotate(${0})`}>
        {/* Show all hexagons (chessboard background) and algebraic notation at the side */}
        {board.getHexagons().map((hex, index) => (
          <HexagonComponent key={index} hex={hex} rotation={0} disableNumbers highlight={attackMap[hex.memIdx]}></HexagonComponent>
        ))}
      </g>

      {/* Show all chess pieces */}
      {pieces.map((piece) => (
        <PieceElement key={piece.getUniqueID()} piece={piece} rotation={0} />
      ))}
    </svg>
  );
};

const Player = ["W", "G", "B"];
const PlayerNames = ["White", "Gray", "Black"];
const Piece = ["P", "N", "B", "R", "Q", "K"];
const PieceNames = ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"];

const DebugVew = ({ fen }: { fen: string }) => {
  const { mutateAsync } = usePostChessAttackMaps();

  const [data, setData] = useState<AttackMapsOut>(null);

  const getMaps = async () => {
    const res = await mutateAsync({ data: { fen } });
    setData(res);
    console.log(res);
  };

  return (
    <Box
      w="100%"
      h="100vh" // or calc(100vh - header height)
      style={{
        overflowY: "auto",
        overflowX: "hidden",
      }}
    >
      <Button onClick={getMaps} mb="md">
        Generate Attack Maps
      </Button>
      <Flex direction="column" align="center" justify="center" gap="md" w="100%" h="100%">
        {Player.map((p, i) => (
          <Flex direction="column" key={p} align="center" gap="sm" w="100%">
            <Text size="lg" fw="bolder" mb="sm">
              {PlayerNames[i]}
            </Text>

            {/* horizontal scrolling row */}
            <Flex
              gap="sm"
              w="100%"
              style={{
                overflowX: "auto",
                overflowY: "hidden",
                justifyContent: "flex-start", // ✅ align row left
              }}
            >
              {Piece.map((pc, j) => (
                <Box
                  key={`${p}${pc}`}
                  miw={300}
                  h={300}
                  bg="gray.0"
                  style={{
                    border: "1px solid #ccc",
                    position: "relative",
                    flexShrink: 0,
                  }}
                >
                  <Text size="sm" mb="xs" pos="absolute" top={5} left={5} px="4px" style={{ pointerEvents: "none" }}>
                    {PieceNames[j]}
                  </Text>
                  <AttackMapBoard fen={fen} attackMap={data ? data.maps[`${p}${pc}`] : new Array(96).fill(false)} />
                </Box>
              ))}
            </Flex>
          </Flex>
        ))}
      </Flex>
    </Box>
  );
};

export default DebugVew;
