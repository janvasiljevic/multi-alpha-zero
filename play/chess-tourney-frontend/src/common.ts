import { GameStatus } from "./api/model";

export const isGameOver = (status: GameStatus) => {
  return (
    status === GameStatus.FinishedDraw ||
    status === GameStatus.FinishedSemiDraw ||
    status === GameStatus.FinishedWin
  );
};
