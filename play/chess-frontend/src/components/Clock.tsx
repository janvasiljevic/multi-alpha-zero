import { useEffect, useState, useCallback, useRef, useImperativeHandle, forwardRef } from "react";
import { Color } from "tri-hex-chess";
import { get3PointsCircle } from "../utils";

export type TimeInMs = number;
export type ClockTimes = [TimeInMs, TimeInMs, TimeInMs];

type ClockConfig = {
  defaultTime?: TimeInMs;
  tickInterval?: number;
  increment?: number;
  incrementOnTimeout?: number;
};

type ClockProps = {
  onTimeout: (color: Color) => void;
  currentTurn: Color;
  isClockRunning: boolean;
  rotation: number;
  config?: ClockConfig;
};

// Helper functions
const msToMMSSss = (ms: number) => {
  const totalSeconds = Math.floor(ms / 1_000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  const milliseconds = Math.floor((ms % 1_000) / 10);

  return `${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}.${milliseconds
    .toString()
    .padStart(2, "0")}`;
};

const getTextColor = (ms: number) => {
  if (ms > 20_000) return "black";
  if (ms > 5_000) return "orange";
  return "red";
};

export type ClockHandle = {
  resetClock: (times: ClockTimes) => void;
  setPlayerTime: (player: Color, time: TimeInMs) => void;
  setClockTimes: React.Dispatch<React.SetStateAction<ClockTimes>>;
  addIncrement: (player: Color) => void;
  getCurrentPlayerTime: () => TimeInMs;
  getLastTimes: () => ClockTimes;
};

const points = get3PointsCircle(480);
const playerOrder: Color[] = [Color.White, Color.Gray, Color.Black];

// Main component
const Clock = forwardRef<ClockHandle, ClockProps>(
  ({ onTimeout, currentTurn, isClockRunning, rotation, config = {} }: ClockProps, ref) => {
    const { defaultTime = 3_000, tickInterval = 100, increment = 5_000, incrementOnTimeout = 10_000 } = config;

    const [times, setTimes] = useState<ClockTimes>([defaultTime, defaultTime, defaultTime]);

    const lastTimes = useRef<ClockTimes>(times);

    // Control functions
    const resetClock = useCallback((times: ClockTimes) => {
      setTimes(times);
    }, []);

    const setPlayerTime = useCallback((player: Color, time: TimeInMs) => {
      setTimes((prev) => {
        const newTimes = [...prev] as ClockTimes;
        newTimes[player] = time;
        return newTimes;
      });
    }, []);

    const addIncrement = useCallback(
      (player: Color) => {
        setTimes((prev) => {
          const newTimes = [...prev] as ClockTimes;
          newTimes[player] += increment;
          return newTimes;
        });
      },
      [increment]
    );

    // Get current values without exposing state directly
    const getCurrentPlayerTime = useCallback(() => {
      return times[currentTurn];
    }, [times, currentTurn]);

    const getLastTimes = useCallback(() => {
      return lastTimes.current;
    }, []);

    // Expose control via ref
    useImperativeHandle(ref, () => ({
      resetClock,
      setPlayerTime,
      setClockTimes: setTimes,
      addIncrement,
      getCurrentPlayerTime,
      getLastTimes,
    }));

    useEffect(() => {
      lastTimes.current = times;
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [currentTurn]);

    useEffect(() => {
      if (!isClockRunning) return;

      const interval = setInterval(() => {
        setTimes((prevTimes) => {
          const newTimes = [...prevTimes] as ClockTimes;
          newTimes[currentTurn] = Math.max(0, newTimes[currentTurn] - tickInterval);

          if (newTimes[currentTurn] === 0) {
            newTimes[currentTurn] = incrementOnTimeout;
            // This is super weird, but prevents React from complaining that a components is setting
            // state during render.
            setTimeout(() => onTimeout(currentTurn), 0);
          }

          return newTimes;
        });
      }, tickInterval);

      return () => clearInterval(interval);
    }, [isClockRunning, tickInterval, currentTurn, onTimeout, incrementOnTimeout, increment]);

    // Render
    return (
      <>
        {points.map(([x, y], i) => (
          <g transform={`translate(${x}, ${y})`} key={i}>
            <text
              x="0"
              y="0"
              fill={getTextColor(times[playerOrder[i]])}
              fontSize="26"
              dominantBaseline="central"
              textAnchor="middle"
              transform={`rotate(${-rotation})`}
              fontWeight="bold"
              style={{ userSelect: "none" }}
              fontFamily="IBM Plex Mono"
            >
              {msToMMSSss(times[playerOrder[i]])}
            </text>
          </g>
        ))}
      </>
    );
  }
);

export default Clock;
