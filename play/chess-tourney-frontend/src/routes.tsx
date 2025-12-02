// routes.tsx
import React from "react";
import { TanStackRouterDevtools } from "@tanstack/react-router-devtools";
import {
  createRootRouteWithContext,
  createRoute,
  createRouter,
  Outlet,
  redirect,
} from "@tanstack/react-router";
import { GamesListPage } from "./routes/GamesListPage.tsx";
import { RegisterPage } from "./routes/RegisterPage.tsx";
import { LoginPage } from "./routes/LoginPage.tsx";
import { Flex, Image, Text } from "@mantine/core";
import { CustomLink } from "./components/CustomLink.tsx";
import GameDetailPage from "./routes/GamePlayPage.tsx";
import { useAuthStore } from "./authStore.ts";
import { BotsListPage } from "./routes/BotsListPage.tsx";
import { UserType } from "./api/model";
import { LeaderBoardPage } from "./routes/LeaderBoardPage.tsx";
import GameHistoryPage from "./routes/GameHistoryPage.tsx";
import { ProfilePage } from "./routes/ProfilePage.tsx";

const LogoutPage = () => {
  const auth = useAuthStore();
  React.useEffect(() => auth.logout(), []);
  return <div>Logging out...</div>;
};

const rootRoute = createRootRouteWithContext()({
  component: () => (
    <>
      <TanStackRouterDevtools />
      <Outlet />
    </>
  ),
  notFoundComponent: () => (
    <Flex
      w="100vw"
      h="100vh"
      align="center"
      justify="center"
      direction="column"
    >
      <Image w="100px" src="/images.jpeg"></Image>
      <Text fw="bold">
        404 - You just blundered the URL. Magnus is disappointed.
      </Text>

      <CustomLink label="Go to Login Page" to={loginRoute.id} />
    </Flex>
  ),
});

const publicRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/public",
  component: () => <Outlet />,
});

export const loginRoute = createRoute({
  getParentRoute: () => publicRoute,
  path: "/login",
  component: LoginPage,
});

export const registerRoute = createRoute({
  getParentRoute: () => publicRoute,
  path: "/register",
  component: RegisterPage,
});

export const gamesRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/games",
  component: GamesListPage,
  beforeLoad: () => {
    const auth = useAuthStore.getState();

    if (!auth.isAuthenticated) {
      throw redirect({ to: loginRoute.id, statusCode: 307 });
    }
  },
});

export const profileRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/profile",
  component: ProfilePage,
  beforeLoad: () => {
    const auth = useAuthStore.getState();

    if (!auth.isAuthenticated) {
      throw redirect({ to: loginRoute.id, statusCode: 307 });
    }
  },
});

export const botsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/bots",
  component: BotsListPage,
  beforeLoad: () => {
    const auth = useAuthStore.getState();

    if (!auth.isAuthenticated) {
      throw redirect({ to: loginRoute.id, statusCode: 307 });
    }

    if (auth.user?.type != UserType.Admin) {
      throw redirect({ to: gamesRoute.id, statusCode: 307 });
    }
  },
});
export const leaderBoardRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/leaderboard",
  component: LeaderBoardPage,
  beforeLoad: () => {
    const auth = useAuthStore.getState();

    if (!auth.isAuthenticated) {
      throw redirect({ to: loginRoute.id, statusCode: 307 });
    }
  },
});

export const gameDetailRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/game/$id",
  component: GameDetailPage,
  beforeLoad: () => {
    const auth = useAuthStore.getState();

    if (!auth.isAuthenticated) {
      throw redirect({ to: loginRoute.id, statusCode: 307 });
    }
  },
});

export const gameHistoryRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/history/$id",
  component: GameHistoryPage,
  beforeLoad: () => {
    const auth = useAuthStore.getState();

    if (!auth.isAuthenticated) {
      throw redirect({ to: loginRoute.id, statusCode: 307 });
    }
  },
});



const logoutRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/logout",
  component: LogoutPage,
  beforeLoad: () => {
    const auth = useAuthStore.getState();

    if (!auth.isAuthenticated) {
      throw redirect({ to: loginRoute.id, statusCode: 307 });
    }
  },
});

const notFoundRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "*",
  component: () => (
    <Flex
      w="100vw"
      h="100vh"
      align="center"
      justify="center"
      direction="column"
    >
      <h1>404 - Page Not Found</h1>
      <CustomLink label="Go to Login Page" to={loginRoute.id} />
    </Flex>
  ),
});

// Build tree
const routeTree = rootRoute.addChildren([
  publicRoute.addChildren([loginRoute, registerRoute]),
  gamesRoute,
  gameDetailRoute,
  botsRoute,
  leaderBoardRoute,
  gameHistoryRoute,
  profileRoute,
  logoutRoute,
  notFoundRoute,
]);

export const router = createRouter({
  routeTree,
});
