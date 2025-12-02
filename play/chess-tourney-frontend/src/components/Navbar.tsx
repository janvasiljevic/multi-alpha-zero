import {
  ActionIcon,
  Button,
  Divider,
  Flex,
  Text,
  ThemeIcon,
} from "@mantine/core";
import {
  IconChessQueen,
  IconList,
  IconLogout,
  IconRobot,
  IconTrophy,
} from "@tabler/icons-react";
import { useNavigate } from "@tanstack/react-router";
import {
  botsRoute,
  gamesRoute,
  leaderBoardRoute,
  loginRoute,
  profileRoute,
} from "../routes.tsx";
import { useAuthStore } from "../authStore.ts";

const Navbar = () => {
  const auth = useAuthStore();

  const navigate = useNavigate();

  const logoutAndRedirect = async () => {
    auth.logout();
    await navigate({ to: loginRoute.id });
  };

  const goToGamesList = async () => {
    await navigate({ to: gamesRoute.id });
  };

  const goToBotsList = async () => {
    await navigate({ to: botsRoute.id });
  };

  const goToLeaderBoard = async () => {
    await navigate({ to: leaderBoardRoute.id });
  };

  const goToProfile = async () => {
    await navigate({ to: profileRoute.id });
  }

  return (
    <Flex
      w="100%"
      h={60}
      bg="gray.1"
      px={20}
      align="center"
      justify="space-between"
    >
      <Flex gap="sm" align="center">
        <ThemeIcon size="lg" radius="md" color="blue" variant="light">
          <IconChessQueen />
        </ThemeIcon>
        <Text fw="bold">Tourney</Text>
      </Flex>

      <Flex gap="sm" align="center">
        <Button variant="subtle" size="compact-sm" onClick={goToProfile}>
          {auth.user?.username || "Guest"}
        </Button>

        <ActionIcon
          onClick={goToGamesList}
          title="Games List"
          variant="light"
          size="lg"
          color="blue"
        >
          <IconList />
        </ActionIcon>

        {auth.user?.type != "Regular" && (
          <Flex gap="sm">
            <ActionIcon
              onClick={goToBotsList}
              title="Bots List"
              variant="light"
              size="lg"
              color="orange"
            >
              <IconRobot />
            </ActionIcon>
          </Flex>
        )}
        <ActionIcon
          onClick={goToLeaderBoard}
          title="Leader Board"
          variant="light"
          size="lg"
          color="yellow"
        >
          <IconTrophy />
        </ActionIcon>

        <Divider orientation="vertical" mx="sm" h={30} />
        <ActionIcon
          onClick={logoutAndRedirect}
          title="Logout"
          color="red"
          variant="light"
          size="lg"
        >
          <IconLogout />
        </ActionIcon>
      </Flex>
    </Flex>
  );
};
export default Navbar;
