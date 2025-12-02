import { Color, Piece } from "../../libs/wasm";
import { ActionIcon, Flex, Paper } from "@mantine/core";
import { libColorToString, libPieceToString } from "../lib/utils/utils.ts";
import { IconHandClick, IconTrash } from "@tabler/icons-react";
import { Tool } from "./ChessEditorView.tsx";
import React from "react";

type Props = {
  color: Color;
  orientation: "vertical" | "horizontal";
  setTool: React.Dispatch<React.SetStateAction<Tool>>;
  tool: Tool;
};

const pieces: Piece[] = [Piece.Pawn, Piece.Knight, Piece.Bishop, Piece.Rook, Piece.Queen, Piece.King];

const getSvgString = (color: Color, piece: Piece): string => {
  return `${libColorToString(color).toLowerCase()}${libPieceToString(piece).toUpperCase()}`;
};

const sizeSvg = 50;
const selectedColor = "grape"
const selectedVariant = "outline"

const EditorPieceBox = ({ color, orientation, setTool, tool }: Props) => {
  return (
    <Paper withBorder radius="md">
      <Flex direction={orientation === "vertical" ? "column" : "row"} gap="sm" align="center" justify="center" p="sm">
        <ActionIcon
          w={sizeSvg}
          h={sizeSvg}
          variant={tool.mode === "move" ? selectedVariant : "light"}
          color={tool.mode === "move" ? selectedColor : "blue"}
          radius="md"
          onClick={() => setTool({ mode: "move", piece: undefined })}
        >
          <IconHandClick />
        </ActionIcon>

        <ActionIcon
          w={sizeSvg}
          h={sizeSvg}
          variant={tool.mode === "remove" ? selectedVariant : "light"}
          color={tool.mode === "remove" ? selectedColor : "blue"}
          radius="md"
          onClick={() => setTool({ mode: "remove", piece: undefined })}
        >
          <IconTrash />
        </ActionIcon>

        {pieces.map((piece) => (
          <ActionIcon
            key={getSvgString(color, piece)}
            size={sizeSvg}
            variant={tool.mode === "add" && tool.piece?.type === piece && tool.piece?.color === color ? selectedVariant : "light"}
            color={tool.mode === "add" && tool.piece?.type === piece && tool.piece?.color === color ?selectedColor : "blue"}
            radius="md"
          >
            <svg
              width={sizeSvg}
              height={sizeSvg}
              viewBox={`0 0 ${sizeSvg} ${sizeSvg}`}
              xmlns="http://www.w3.org/2000/svg"
              style={{ touchAction: "none", cursor: "grab" }}
              onClick={() => setTool({ mode: "add", piece: { color, type: piece } })}
            >
              <image
                href={`/pieces/${getSvgString(color, piece)}.svg`}
                type="image/svg+xml"
                width={sizeSvg}
                height={sizeSvg}
                // transform={`translate(${s / 8}, ${s / 8})`}
                style={{
                  filter: "grayscale(1)",
                }}
              />
            </svg>
          </ActionIcon>
        ))}
      </Flex>
    </Paper>
  );
};
export default EditorPieceBox;
