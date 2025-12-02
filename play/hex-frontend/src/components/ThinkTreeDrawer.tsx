import type {
  ApiNodeAttributes,
  BoardState,
  ThinkResult,
} from "../api/def/model";
import React, { useState } from "react";
import { usePostHexApplyMoves } from "../api/def/default/default.ts";
import { Button, Drawer, Flex, Text } from "@mantine/core";
import HexBoard from "./HexBoard.tsx";
import Tree, { type TreeNodeDatum } from "react-d3-tree";
import ScoresDisplay from "./ScoresDisplay.tsx";
import { getPlayerColorByIndex, useCenteredTree } from "../misc/util.ts";

import "./../styles/think-tree.css";

export function ThinkTreeDrawer(props: {
  opened: boolean;
  onClose: () => void;
  onHexClick: () => void;
  thinkResult: ThinkResult | null;
}) {
  const [dimensions, translate, containerRef] = useCenteredTree();

  const [hoveredBoardState, setHoveredBoardState] = useState<BoardState | null>(
    null,
  );

  const { mutateAsync: applyMovesApi } = usePostHexApplyMoves();

  const foreignObjectProps = {
    width: 180,
    height: 150,
    x: 20,
    y: 10,
  };

  const nodeClickCallback = async (data: ApiNodeAttributes) => {
    if (data.is_terminal) {
      return;
    }

    const moveIndexes = data.move_indexes;
    if (moveIndexes.length === 0 || !props.thinkResult) {
      return;
    }

    const res = await applyMovesApi({
      data: {
        board: props.thinkResult.root_board,
        moves_policy_indices: moveIndexes,
      },
    });

    setHoveredBoardState(res);
  };

  return (
    <Drawer
      opened={props.opened}
      onClose={props.onClose}
      size="80%"
      h="100%"
      withCloseButton={false}
      styles={{
        inner: { height: "100%" },
        body: { height: "100%" },
      }}
    >
      <Flex h="100%" direction="column" gap="md" p="md">
        <Text>
          This is a tree view of the think process. It shows the decisions made
          by the AI during the thinking process. You can zoom in and out to see
          the details of each decision and the corresponding game state at that
          point in time.
        </Text>

        <div
          id="treeWrapper"
          style={{
            width: "100%",
            flex: 1,
            border: "1px solid #fff",
            borderColor: "#b0b0b0",
            position: "relative",
          }}
          ref={containerRef}
        >
          {hoveredBoardState && (
            <Flex
              pos="absolute"
              top={20}
              left={20}
              style={{
                transform: "scale(0.7) translate(-20%, -20%)",
              }}
            >
              <HexBoard
                board={hoveredBoardState}
                onHexClick={props.onHexClick}
                thinkResult={null}
              />
            </Flex>
          )}

          {props.thinkResult ? (
            <Tree
              zoomable
              dimensions={dimensions}
              translate={translate}
              orientation="vertical"
              separation={{
                siblings: 2,
                nonSiblings: 2,
              }}
              // @ts-ignore
              data={props.thinkResult.tree}
              depthFactor={300}
              renderCustomNodeElement={(rd3tProps) =>
                renderForeignObjectNode({
                  ...rd3tProps,
                  foreignObjectProps,
                  nodeHoverEnter: nodeClickCallback,
                  nodeHoverLeave: () => setHoveredBoardState(null),
                })
              }
            />
          ) : (
            <Text>Awaiting response.</Text>
          )}
        </div>

        <Button onClick={props.onClose} mt="md">
          Close
        </Button>
      </Flex>
    </Drawer>
  );
}

type RenderForeignObjectNodeProps = {
  nodeDatum: TreeNodeDatum;
  toggleNode: () => void;
  foreignObjectProps: React.SVGProps<SVGForeignObjectElement>;
  nodeHoverEnter: (data: ApiNodeAttributes) => Promise<void>;
  nodeHoverLeave: () => void;
};

const renderForeignObjectNode = ({
  nodeDatum,
  toggleNode,
  foreignObjectProps,
  nodeHoverEnter,
  nodeHoverLeave,
}: RenderForeignObjectNodeProps) => {
  const {
    player_to_move,
    visit_count,
    raw_policy,
    sum_values,
    net_value,
    is_terminal,
  } = nodeDatum.attributes as unknown as ApiNodeAttributes;

  return (
    <g className="node-custom-g">
      {!is_terminal ? (
        <circle r={15} fill={getPlayerColorByIndex(player_to_move)}></circle>
      ) : (
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 480 480"
          width="50px"
          height="50px"
          x={-25}
          y={-25}
        >
          <path
            d="M450 210A180 180 0 0 1 270 30a30 30 0 1 0-60 0A180 180 0 0 1 30 210a30 30 0 1 0 0 60 180 180 0 0 1 180 180 30 30 0 1 0 60 0 180 180 0 0 1 180-180 30 30 0 1 0 0-60Z"
            fill={getPlayerColorByIndex(player_to_move)}
            stroke="yellow"
            strokeWidth={20}
          ></path>
        </svg>
      )}

      <text
        x={0}
        y={0}
        textAnchor="middle"
        dominantBaseline="central"
        fill="white !important"
        fontSize="16"
        fontWeight={300}
      >
        {visit_count}
      </text>
      {/* `foreignObject` requires width & height to be explicitly set. */}
      <foreignObject {...foreignObjectProps}>
        <div
          style={{
            border: "1px solid black",
            backgroundColor: "#dedede",
            padding: "6px",
            display: "flex",
            flexDirection: "column",
          }}
          onMouseEnter={() =>
            nodeHoverEnter(nodeDatum.attributes as unknown as ApiNodeAttributes)
          }
          onMouseLeave={nodeHoverLeave}
        >
          <Text fw="bolder" style={{ textAlign: "center" }}>
            {nodeDatum.name}
          </Text>

          {raw_policy && <text>&Pi; prior: {raw_policy.toFixed(2)}</text>}

          {(net_value && (
            <Flex gap="sm" justify="start">
              <Text fs="italic" span size="md">
                v
              </Text>
              <ScoresDisplay scores={net_value} size="xs" />
            </Flex>
          )) || <text>Net value: N/A</text>}

          {(sum_values && (
            <Flex gap="sm" justify="start">
              <Text span size="md">
                &Sigma;
              </Text>
              <ScoresDisplay scores={sum_values} size="xs" />
            </Flex>
          )) || <text>Sum values: N/A</text>}

          {nodeDatum.children && (
            <button style={{ width: "100%" }} onClick={toggleNode}>
              {nodeDatum.__rd3t.collapsed ? "Expand" : "Collapse"}
            </button>
          )}
        </div>
      </foreignObject>
    </g>
  );
};
