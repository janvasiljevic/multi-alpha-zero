import { useState } from "react";

type ArrowProps = {
  from: { x: number; y: number };
  to: { x: number; y: number };
  color: string;
  strokeWidth: number;
  opacity: number;
  text?: string; // Optional text to display
};

// These can be adjusted to change the arrow's appearance
const ARROWHEAD_LENGTH = 40;
const ARROWHEAD_WIDTH = 40;

const Arrow = ({ from, to, color, strokeWidth, opacity, text }: ArrowProps) => {
  const [isHovered, setIsHovered] = useState(false);

  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const len = Math.sqrt(dx * dx + dy * dy);

  if (len === 0) return null;

  const unitDx = dx / len;
  const unitDy = dy / len;

  // --- Calculations for Arrow Shape ---
  const lineEndX = to.x - unitDx * ARROWHEAD_LENGTH;
  const lineEndY = to.y - unitDy * ARROWHEAD_LENGTH;

  const tip = { x: to.x, y: to.y };
  const perpDx = -unitDy;
  const perpDy = unitDx;
  const point1 = tip;
  const point2 = {
    x: lineEndX + perpDx * (ARROWHEAD_WIDTH / 2),
    y: lineEndY + perpDy * (ARROWHEAD_WIDTH / 2),
  };
  const point3 = {
    x: lineEndX - perpDx * (ARROWHEAD_WIDTH / 2),
    y: lineEndY - perpDy * (ARROWHEAD_WIDTH / 2),
  };
  const polygonPoints = `${point1.x},${point1.y} ${point2.x},${point2.y} ${point3.x},${point3.y}`;

  // --- Calculations for Text ---
  // Midpoint of the line segment (not the full arrow)
  const midX = (from.x + lineEndX) / 2;
  const midY = (from.y + lineEndY) / 2;
  // Angle for rotation in degrees
  let angle = (Math.atan2(dy, dx) * 180) / Math.PI;

  const currentOpacity = isHovered ? 1.0 : opacity;

  return (
    <g
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      opacity={currentOpacity}
      style={{ cursor: "pointer", transition: "opacity 0.2s ease-in-out" }}
    >
      {/* Arrow Body */}
      <line
        x1={from.x}
        y1={from.y}
        x2={lineEndX}
        y2={lineEndY}
        stroke={color}
        strokeWidth={strokeWidth}
        strokeLinecap="round"
      />
      {/* Arrow Head */}
      <polygon points={polygonPoints} fill={color} />

      {/* Text in the middle */}
      {text && (
        <text
          x={midX}
          y={midY}
          fill="black" // Contrasting color
          fontSize="14"
          fontWeight="bold"
          textAnchor="middle" // Center horizontally
          dominantBaseline="middle" // Center vertically
          // Rotate text around its own center
          transform={`rotate(${angle} ${midX} ${midY})`}
        >
          {text}
        </text>
      )}
    </g>
  );
};

export default Arrow;