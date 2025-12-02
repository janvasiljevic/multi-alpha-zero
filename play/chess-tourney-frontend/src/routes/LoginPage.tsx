import React from "react";
import { useAuthLogin } from "../api/authentication/authentication.ts";
import { useNavigate } from "@tanstack/react-router";
import { useForm } from "@mantine/form";
import { showNotification } from "@mantine/notifications";
import {
  Button,
  Container,
  Flex,
  Paper,
  PasswordInput,
  Stack,
  TextInput,
  Title,
} from "@mantine/core";
import { CustomLink } from "../components/CustomLink.tsx";
import { gamesRoute, loginRoute, registerRoute } from "../routes.tsx";
import { useAuthStore } from "../authStore.ts";
import classes from "./Background.module.css";

export const LoginPage = () => {
  const [loading, setLoading] = React.useState(false);
  const { mutateAsync } = useAuthLogin();

  const navigate = useNavigate({ from: loginRoute.id });
  const auth = useAuthStore();

  const form = useForm({
    initialValues: {
      username: "",
      password: "",
    },
    validate: {
      username: (value) =>
        value.trim().length === 0 ? "Username is required" : null,
      password: (value) =>
        value.trim().length === 0 ? "Password is required" : null,
    },
  });

  const handleLogin = async (values: typeof form.values) => {
    setLoading(true);
    try {
      const data = await mutateAsync({ data: values });

      auth.login(data.token, data.user);

      showNotification({
        title: "Login Successful",
        message: "You have been logged in successfully.",
        color: "green",
      });

      navigate({ to: gamesRoute.id });
    } catch (error: any) {
      showNotification({
        title: "Login Failed",
        message:
          error.response?.data?.message || "An error occurred during login.",
        color: "red",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Flex
      align={"center"}
      justify="center"
      h="100%"
      mih="100vh"
      w="100vw"
      direction="column"
      className={classes.bg}
    >
      <Container size={520} w="100%" py="xl">
        <Flex align="center" justify="center" direction="column" h="100%">
          <Paper p="xl" radius="md" withBorder w="100%">
            <Title order={2} ta="center" mb="lg">
              Login
            </Title>
            <form onSubmit={form.onSubmit(handleLogin)}>
              <Stack>
                <TextInput
                  label="Username"
                  placeholder="Enter your username"
                  {...form.getInputProps("username")}
                />

                <PasswordInput
                  label="Password"
                  placeholder="Enter your password"
                  {...form.getInputProps("password")}
                />

                <Button type="submit" loading={loading} fullWidth>
                  Login
                </Button>
              </Stack>
            </form>
          </Paper>

          <CustomLink
            label={"Don't have an account? Register"}
            to={registerRoute.id}
            mt="lg"
          />
        </Flex>
      </Container>
    </Flex>
  );
};
