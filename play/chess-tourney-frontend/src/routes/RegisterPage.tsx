import React from "react";
import { useAuthRegister } from "../api/authentication/authentication.ts";
import { useNavigate } from "@tanstack/react-router";
import { useForm } from "@mantine/form";
import { showNotification } from "@mantine/notifications";
import {
  Button,
  Container,
  Flex,
  NumberInput,
  Paper,
  PasswordInput,
  Stack,
  Text,
  TextInput,
  Title,
} from "@mantine/core";
import { CustomLink } from "../components/CustomLink.tsx";
import { gamesRoute, loginRoute, registerRoute } from "../routes.tsx";
import { useAuthStore } from "../authStore.ts";
import type { RegisterPayload } from "../api/model";

import classes from "./Background.module.css";

type CreateUser = RegisterPayload & {
  confirmPassword: string;
};

export const RegisterPage = () => {
  const auth = useAuthStore();
  const [loading, setLoading] = React.useState(false);
  const { mutateAsync } = useAuthRegister(); // assuming you have a register hook

  const navigate = useNavigate({ from: registerRoute.id });

  const form = useForm<CreateUser>({
    initialValues: {
      username: "",
      password: "",
      confirmPassword: "",
      // Optional stuff
      chess_com_rating: undefined,
      experience_with_chess: undefined,
      fide_rating: undefined,
      lichess_rating: undefined,
    },
    transformValues: (values) => {
      const normalize = (v: unknown) => (typeof v === "number" ? v : undefined);

      return {
        username: values.username,
        password: values.password,
        confirmPassword: values.confirmPassword,
        fide_rating: normalize(values.fide_rating),
        lichess_rating: normalize(values.lichess_rating),
        chess_com_rating: normalize(values.chess_com_rating),
        experience_with_chess: normalize(values.experience_with_chess),
      };
    },
    validate: {
      username: (value) =>
        value.trim().length === 0 ? "Username is required" : null,
      password: (value) =>
        value.trim().length === 0 ? "Password is required" : null,
      confirmPassword: (value, values) =>
        value !== values.password ? "Passwords must match" : null,
      fide_rating: (value) =>
        typeof value === "number" && (value < 100 || value > 3000)
          ? "FIDE rating must be between 100 and 3000"
          : null,
      lichess_rating: (value) =>
        typeof value === "number" && (value < 100 || value > 3000)
          ? "Lichess rating must be between 100 and 3000"
          : null,
      chess_com_rating: (value) =>
        typeof value === "number" && (value < 100 || value > 3000)
          ? "Chess.com rating must be between 100 and 3000"
          : null,
    },
  });

  const handleRegister = async (values: typeof form.values) => {
    setLoading(true);
    try {
      await mutateAsync(
        { data: values },
        {
          onError: (error) => {
            showNotification({
              title: "Registration Failed",
              message:
                error.response?.data?.message ||
                "An error occurred during registration.",
              color: "red",
            });
          },
          onSuccess: (data) => {
            auth.login(data.token, data.user);
            showNotification({
              title: "Registration Successful",
              message: "You have been registered and logged in.",
              color: "green",
            });
            navigate({ to: gamesRoute.id });
          },
        },
      );
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
              Register
            </Title>
            <form onSubmit={form.onSubmit(handleRegister)}>
              <Stack>
                <TextInput
                  required
                  label="Username"
                  placeholder="Enter your username"
                  {...form.getInputProps("username")}
                />
                <PasswordInput
                  required
                  label="Password"
                  placeholder="Enter your password"
                  {...form.getInputProps("password")}
                />
                <PasswordInput
                  required
                  label="Confirm Password"
                  placeholder="Confirm your password"
                  {...form.getInputProps("confirmPassword")}
                />
                <Text fw="bold" mt="md">
                  Optional Information
                </Text>
                <Text>
                  Not visible to other users, purely for statistical purposes.
                  Can be updated later in profile settings.
                </Text>
                <NumberInput
                  label="FIDE Rating"
                  placeholder="Enter your FIDE rating (optional)"
                  {...form.getInputProps("fide_rating")}
                />
                <NumberInput
                  label="Lichess Rating"
                  placeholder="Enter your Lichess rating (optional)"
                  {...form.getInputProps("lichess_rating")}
                />
                <NumberInput
                  label="Chess.com Rating"
                  placeholder="Enter your Chess.com rating (optional)"
                  {...form.getInputProps("chess_com_rating")}
                />
                <NumberInput
                  label="Experience with Chess (1-10)"
                  placeholder="Rate your experience with chess (optional)"
                  min={1}
                  max={10}
                  {...form.getInputProps("experience_with_chess")}
                />
                <Button type="submit" loading={loading} fullWidth>
                  Register
                </Button>
              </Stack>
            </form>
          </Paper>
          <CustomLink
            to={loginRoute.id}
            label={"Already have an account? Login"}
            mt="lg"
          />
        </Flex>
      </Container>
    </Flex>
  );
};
