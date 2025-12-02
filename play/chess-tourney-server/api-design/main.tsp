import "@typespec/http";
import "@typespec/openapi";

using Http;
using OpenAPI;

@doc("API for a custom three-player chess variant game server. Authentication is required via a JWT Bearer token for protected endpoints.")
@server("http://localhost:3000")
@service
namespace ThreePlayerChessService;

@error
@doc("Bad Request")
model BadRequestError {
  @statusCode
  code: 400;

  message: string;
}

@error
@doc("Unauthorized access")
model UnauthorizedError {
  @statusCode
  code: 401;

  message: string;
}

@error
@doc("Forbidden")
model ForbiddenError {
  @statusCode
  code: 403;

  message: string;
}

@error
@doc("Not Found")
model NotFoundError {
  @statusCode
  code: 404;

  message: string;
}

@error
@doc("Conflict")
model ConflictError {
  @statusCode
  code: 409;

  message: string;
}

@error
@doc("ISE")
model InternalError {
  @statusCode
  code: 500;

  message: string;
}

alias ApiError =
  | BadRequestError
  | UnauthorizedError
  | ForbiddenError
  | NotFoundError
  | ConflictError
  | InternalError;

enum UserType {
  Regular,
  Bot,
  Admin,
}

model User {
  id: int64;
  username: string;
  type: UserType;
}

model RegisterPayload {
  username: string;

  @secret
  password: string;

  @minValue(100)
  @maxValue(3000)
  lichess_rating?: uint32;

  @minValue(100)
  @maxValue(3000)
  chess_com_rating?: uint32;

  @minValue(100)
  @maxValue(3000)
  fide_rating?: uint32;

  @minValue(0)
  @maxValue(10)
  experience_with_chess?: uint32;
}

model UpdateProfilePayload {
  @minValue(100)
  @maxValue(3000)
  lichess_rating?: uint32;

  @minValue(100)
  @maxValue(3000)
  chess_com_rating?: uint32;

  @minValue(100)
  @maxValue(3000)
  fide_rating?: uint32;

  @minValue(0)
  @maxValue(10)
  experience_with_chess?: uint32;
}

model LoginPayload {
  username: string;

  @secret
  password: string;
}

model LoginResponse {
  @statusCode
  code: 200;

  @doc("The JWT token to be used for authenticating subsequent requests.")
  token: string;

  user: User;
}

model UserCreated {
  @statusCode
  code: 201;

  @doc("The JWT token to be used for authenticating subsequent requests.")
  token: string;

  user: User;
}

model MeUser {
  ...User;

  @minValue(100)
  @maxValue(3000)
  lichess_rating?: uint32;

  @minValue(100)
  @maxValue(3000)
  chess_com_rating?: uint32;

  @minValue(100)
  @maxValue(3000)
  fide_rating?: uint32;

  @minValue(0)
  @maxValue(10)
  experience_with_chess?: uint32;
}

@route("/api/auth")
@tag("Authentication")
namespace Auth {
  @post
  @route("/register")
  @summary("Register a new user")
  op register(@body user: RegisterPayload): UserCreated | ApiError;

  @post
  @route("/login")
  @summary("Log in to get an authentication token")
  op login(@body creds: LoginPayload): LoginResponse | ApiError;

  @patch
  @route("/me")
  @summary("Update the current authenticated user's profile")
  @useAuth(BearerAuth)
  op updateProfile(@body profile: UpdateProfilePayload): MeUser | ApiError;

  @get
  @route("/me")
  @summary("Get the current authenticated user's profile")
  @useAuth(BearerAuth)
  op me(): MeUser | ApiError;
}

enum GameRelation {
  winner,
  loser,
  draw,
}

model Player {
  id: int64;
  username: string;
  relation: GameRelation | null;
  isOwner: boolean;
  isConnectedToGame: boolean;
}

enum GameStatus {
  Waiting,
  InProgress,
  FinishedWin,
  FinishedDraw,
  FinishedSemiDraw,
}

model PlayerUpdate {
  white: Player | null;
  grey: Player | null;
  black: Player | null;
}

model GameState {
  gameId: int64;
  name: string;
  fen: string;
  ownerId: int64;
  ownerUsername: string;
  players: PlayerUpdate | null;
  status: GameStatus;
  players_masked: boolean;
  material_masked: boolean;
  training_mode: boolean;
  suggested_move_time_seconds: int32 | null;
  tournamentId: int64 | null;
}

enum PromotionPiece {
  Queen,
  Rook,
  Bishop,
  Knight,
}

model MakeMovePayload {
  @minValue(0)
  @maxValue(96)
  fromIndex: int32;

  @minValue(0)
  @maxValue(96)
  toIndex: int32;

  promotion?: PromotionPiece;
}

model CreateGame {
  name: string;
  suggested_move_time_seconds?: int32;
  names_masked: boolean;
  material_masked: boolean;
  tournamentId?: int64;
  training_mode: boolean;
}

enum PlayerColor {
  White,
  Grey,
  Black,
}

model JoinGamePayload {
  color: PlayerColor;
}

model WsPlayerJoined {
  username: string;
  color: PlayerColor;
  players: PlayerUpdate;
}

model WsPlayerLeft {
  username: string;
  color: PlayerColor;
  players: PlayerUpdate;
}

model WsGameStarted {
  game: GameState;
}

model WsGameEnded {
  game: GameState;
}

model Position {
  q: int32;
  r: int32;
  s: int32;
  i: int32;
}

model Move {
  from: Position;
  to: Position;
  move_uci: string;
}

model WsMoveMade {
  move: Move;
  newFen: string;
  newTurn: PlayerColor;
}

model WsOnJoinMessage {
  game: GameState;
}

model WsPlayerUpdate {
  players: PlayerUpdate;
}

@discriminated
union WsEvent {
  joined: WsPlayerJoined,
  left: WsPlayerLeft,
  started: WsGameStarted,
  ended: WsGameEnded,
  moveMade: WsMoveMade,
  onJoin: WsOnJoinMessage,
  playerUpdate: WsPlayerUpdate,
}

@doc("Only available for admins")
model FinishGameIn {
  white: GameRelation;
  grey: GameRelation;
  black: GameRelation;
}

enum FilterGames {
  Waiting,
  InProgress,
  Finished,
}

model GameFilter {
  @query
  status?: FilterGames;

  @query
  tournamentId?: int64;
}

@route("/api/games")
@tag("Game Management")
@useAuth(BearerAuth)
namespace Games {
  @post
  @route("/")
  @summary("Creates a new game lobby")
  @doc("Creates a new game and adds the creator as the first player.")
  op createGame(@body game: CreateGame): GameState | ApiError;

  @get
  @route("/")
  @summary("Lists available games to join")
  @doc("Returns a list of games that are in the 'waiting' state and not full.")
  op listGames(...GameFilter): GameState[] | ApiError;

  @post
  @route("/{gameId}/join")
  @summary("Join an existing game")
  op joinGame(
    @path gameId: int64,
    @body join: JoinGamePayload,
  ): void | ApiError;

  @post
  @route("/{gameId}/leave")
  @summary("Leave a game")
  op leaveGame(@path gameId: int64): void | ApiError;

  @post
  @route("/{gameId}/start")
  @summary("Start a game, if all player slots are filled and you are the creator")
  op startGame(@path gameId: int64): void | ApiError;

  @get
  @route("/{gameId}/state")
  @summary("Get the full state of a specific game")
  op getGameState(@path gameId: int64): GameState | ApiError;

  @post
  @route("/{gameId}/move")
  @summary("Make a move in a game")
  op makeMove(
    @path gameId: int64,
    @body move: MakeMovePayload,
  ): void | ApiError;

  @get
  @route("/{gameId}/events")
  @doc("""
    Used only for type generation.
    Actual subscription to game events is done via WebSocket at /ws
    """)
  op subscribeToGameEvents(@path gameId: int64): WsEvent | ApiError;

  @post
  @route("/{gameId}/admin/finish")
  @summary("Finish a game in a specific outcome (Admin only)")
  op finishGameIn(
    @path gameId: int64,
    @body result: FinishGameIn,
  ): void | ApiError;
}

@route("/api/users")
@tag("User Management")
@useAuth(BearerAuth)
namespace Users {
  @get
  @route("")
  @summary("Get a list of all users")
  op listUsers(): User[] | ApiError;

  @get
  @route("/bots")
  @summary("Get a list of all bot users")
  op listBotUsers(): User[] | ApiError;

  @delete
  @route("/{userId}/games/{gameId}")
  @summary("Remove a user from a specific game (Admin only)")
  op removeUserFromGame(
    @path userId: int64,
    @path gameId: int64,
  ): void | ApiError;
}

model AssignBotPayload {
  gameId: int64;
  botId: int64;
  color: PlayerColor;
}

model Bot {
  @visibility(Lifecycle.Read)
  user_id: int64;

  @visibility(Lifecycle.Read)
  bot_id: int64;

  @minLength(3)
  username: string;

  model_key?: string;

  @minValue(1)
  playouts_per_move: int32;

  @minValue(0.0)
  @maxValue(20.0)
  exploration_factor: float32;

  @minValue(0.0)
  @maxValue(5.0)
  virtual_loss: float32;

  @minValue(-3.0)
  @maxValue(3.0)
  contempt: float32;

  @secret
  @visibility(Lifecycle.Create)
  @minLength(1)
  password: string;
}

model UpdateBot is Update<Bot>;
model CreateBot is Create<Bot>;
model ReadBot is Read<Bot>;

@route("/api/bots")
@tag("Bot Management")
@useAuth(BearerAuth)
namespace Bots {
  @post
  @route("/assign")
  @summary("Assign a bot to a game - Admin / Owner only")
  op assignBot(@body assignment: AssignBotPayload): void | ApiError;

  @post
  @route("/create")
  @summary("Create a new bot user - Admin only")
  op create(@body bot: CreateBot): ReadBot | ApiError;

  @patch
  @route("/update/{botId}")
  @summary("Update an existing bot's configuration - Admin")
  op update(@path botId: int64, @body bot: UpdateBot): ReadBot | ApiError;

  @delete
  @route("/delete/{botId}")
  @summary("Delete a bot user (changes the user type back to Regular) - Admin only")
  op delete(@path botId: int64): void | ApiError;

  @get
  @route("/keys")
  @summary("Get available bot model keys")
  op modelKeys(): string[] | ApiError;

  @get
  @route("/")
  @summary("Get list of all bots")
  op listBots(): ReadBot[] | ApiError;
}

model LeaderboardEntry {
  userId: int64;
  isBot: boolean;
  username: string;
  gamesPlayed: int32;
  wins: int32;
  winRate: float32;
  losses: int32;
  lossRate: float32;
  draws: int32;
}

enum LeaderboardSortBy {
  Wins,
  GamesPlayed,
  WinRate,
  LossRate,
  Losses,
  Draws,
}

@route("/api/leaderboard")
@tag("Leaderboard")
@useAuth(BearerAuth)
namespace Leaderboard {
  @get
  @route("/")
  @summary("Get the leaderboard")
  op getLeaderboard(
    @query includeBots: boolean,
    @query sortBy: LeaderboardSortBy,
    @query tournamentId?: int64,
  ): LeaderboardEntry[] | ApiError;
}

model HistoryItem {
  turn_counter: int32;
  fen: string;
  move_uci: string;
  color: PlayerColor;
}

model GameHistory {
  history: HistoryItem[];
  game: GameState;
  maxTurns: int32;
}

@route("/api/history")
@tag("Game History")
@useAuth(BearerAuth)
namespace History {
  @get
  @route("/{gameId}")
  @summary("Get the move history of a specific game")
  op getGameHistory(@path gameId: int64): GameHistory | ApiError;
}

model Tournament {
  @visibility(Lifecycle.Read)
  id: int64;

  @visibility(Lifecycle.Read, Lifecycle.Create)
  name: string;
}

model CreateTournament is Create<Tournament>;
model ReadTournament is Read<Tournament>;

@route("/api/tournaments")
@tag("Tournament Management")
@useAuth(BearerAuth)
namespace Tournaments {
  @post
  @route("/create")
  @summary("Create a new tournament - Admin only")
  op createTournament(
    @body tournament: CreateTournament,
  ): ReadTournament | ApiError;

  @get
  @route("/")
  @summary("Get list of all tournaments")
  op listTournaments(): ReadTournament[] | ApiError;
}
