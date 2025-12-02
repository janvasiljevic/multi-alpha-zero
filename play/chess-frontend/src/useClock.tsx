import { useEffect, useState, useCallback, useRef } from "react";
import { Color } from "tri-hex-chess";

export type TimeInMs = number;

export type ClockTimes = [TimeInMs, TimeInMs, TimeInMs];

type ClockConfig = {
  /** Default time for each player (ms) */
  defaultTime?: TimeInMs;
  /** Update interval for the clock (ms) */
  tickInterval?: number;

  increment?: number;

  incrementOnTimeout?: number;
};

/** Props for the clock hook */
type ClockProps = {
  /** Function called when a player's time runs out */
  onTimeout: (color: Color) => void;
  /** Current player's turn */
  currentTurn: Color;
  /** Optional configuration */
  config?: ClockConfig;

  isClockRunning: boolean;
};

const DEFAULT_TIME = 1000 * 60 * 5; // 5 minutes
const DEFAULT_TICK_INTERVAL = 100; // 50ms
const DEFAULT_INCREMENT_ON_TIMEOUT = 10 * 1000; // 10 seconds

const useClock = ({ onTimeout, currentTurn, config = {}, isClockRunning }: ClockProps) => {
  const {
    defaultTime = DEFAULT_TIME,
    tickInterval = DEFAULT_TICK_INTERVAL,
    incrementOnTimeout = DEFAULT_INCREMENT_ON_TIMEOUT,
  } = config;

  const [times, setTimes] = useState<ClockTimes>([defaultTime, defaultTime, defaultTime]);

  const lastTimes = useRef<ClockTimes>(times);

  const resetClock = useCallback(() => setTimes([defaultTime, defaultTime, defaultTime]), [defaultTime]);

  const setPlayerTime = useCallback((player: Color, time: TimeInMs) => {
    setTimes((prev) => {
      const newTimes = [...prev] as ClockTimes;
      newTimes[player] = time;
      return newTimes;
    });
  }, []);

  useEffect(() => {
    lastTimes.current = times;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentTurn]);

  // Clock tick effect
  useEffect(() => {
    if (!isClockRunning) return;

    const interval = setInterval(() => {
      setTimes((prevTimes) => {
        const newTimes = [...prevTimes] as ClockTimes;
        newTimes[currentTurn] = Math.max(0, newTimes[currentTurn] - tickInterval);

        // Check for timeout
        if (newTimes[currentTurn] === 0) {
          newTimes[currentTurn] = 0; // Ensure time doesn't go negative
          newTimes[currentTurn] = incrementOnTimeout; // Add increment on timeout
          onTimeout(currentTurn);
        }

        return newTimes;
      });
    }, tickInterval);

    return () => clearInterval(interval);
  }, [isClockRunning, tickInterval, currentTurn, onTimeout, incrementOnTimeout]);

  const addIncrement = useCallback((player: Color, increment: number) => {
    setTimes((prev) => {
      const newTimes = [...prev] as ClockTimes;
      newTimes[player] += increment;
      return newTimes;
    });
  }, []);

  return {
    times,
    resetClock,
    setPlayerTime,
    setClockTimes: setTimes,
    currentPlayerTime: times[currentTurn],
    lastTimes: lastTimes.current,
    addIncrement,
  };
};

export default useClock;
