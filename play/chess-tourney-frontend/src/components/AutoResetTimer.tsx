import { useEffect, useState } from "react";
import { Progress } from "@mantine/core";

type AutoResetTimerProps = {
  duration: number;    // total seconds
  active: boolean;     // whether timer should run
  resetKey?: any;      // change this to externally reset the timer
};

export default function AutoResetTimer({ duration, active, resetKey }: AutoResetTimerProps) {
  const [elapsed, setElapsed] = useState(0);
  const [flashOn, setFlashOn] = useState(false);

  const expired = elapsed >= duration;
  const pct = (elapsed / duration) * 100;

  // --- External reset ---
  useEffect(() => {
    setElapsed(0);
    setFlashOn(false);
  }, [resetKey]);

  // --- Timer progression ---
  useEffect(() => {
    if (!active) return;

    const interval = setInterval(() => {
      setElapsed(prev => Math.min(prev + 0.1, duration));
    }, 100);

    return () => clearInterval(interval);
  }, [active, duration]);

  // --- Flash red after expiration ---
  useEffect(() => {
    if (!expired) return;

    const flash = setInterval(() => {
      setFlashOn(f => !f);
    }, 600);

    return () => clearInterval(flash);
  }, [expired]);

  // --- Color selection ---
  const color = expired ? (flashOn ? "red.7" : "red.4") : "blue";

  return (
    <Progress
      w="100%"
      size={4}
      value={pct}
      color={color}      // <-- now color is applied
      striped={expired}  // optional: adds subtle animation during flashing
    />
  );
}
