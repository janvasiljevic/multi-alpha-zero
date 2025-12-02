import {CheckMetadata} from "../class/CheckMetadata.ts";

type Props = {
    check: CheckMetadata;
};

const CheckHighlight = ({check}: Props) => {
    return (
        <>
            <defs>
                <radialGradient id="checkGradient" cx="50%" cy="50%" r="50%" fx="50%" fy="50%">
                    <stop offset="50%" style={{stopColor: "transparent", stopOpacity: 1}}/>
                    <stop offset="100%" style={{stopColor: "red", stopOpacity: 0.8}}/>
                </radialGradient>
                <marker
                    id="arrow"
                    viewBox="0 0 10 10"
                    refX="5"
                    refY="5"
                    markerWidth="6"
                    markerHeight="6"
                    orient="auto-start-reverse"
                >
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="red"/>
                </marker>
            </defs>

            <g transform={`translate(${check.kingX}, ${check.kingY})`}>
                <polygon
                    points="50,0 25,-43.5 -25,-43.5 -50,0 -25,43.5 25,43.5"
                    fill="url(#checkGradient)"
                    stroke="black"
                    strokeWidth="2"
                />
            </g>

            <line
                x1={check.startX}
                y1={check.startY}
                x2={check.endX}
                y2={check.endY}
                stroke="red"
                strokeWidth="2"
                opacity={0.8}
                markerEnd="url(#arrow)"
            />
        </>
    );
};

export default CheckHighlight;
