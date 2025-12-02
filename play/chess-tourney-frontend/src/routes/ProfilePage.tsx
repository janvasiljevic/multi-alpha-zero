import React, { useEffect } from "react";
import {
  useAuthMe,
  useAuthUpdateProfile,
} from "../api/authentication/authentication.ts";
import { useForm } from "@mantine/form";
import { showNotification } from "@mantine/notifications";
import {
  Badge,
  Button,
  Container,
  Flex,
  LoadingOverlay,
  NumberInput,
  Paper,
  Stack,
  Text,
  TextInput,
  Title,
} from "@mantine/core";
import type { UpdateProfilePayload } from "../api/model";
import Navbar from "../components/Navbar.tsx";

export const ProfilePage = () => {
  const [loading, setLoading] = React.useState(false);
  const { mutateAsync } = useAuthUpdateProfile();

  const { data, isLoading } = useAuthMe();

  const form = useForm<UpdateProfilePayload>({
    initialValues: {
      chess_com_rating: undefined,
      experience_with_chess: undefined,
      fide_rating: undefined,
      lichess_rating: undefined,
    },

    transformValues: (values) => {
      const normalize = (v: unknown) => (typeof v === "number" ? v : undefined);

      return {
        fide_rating: normalize(values.fide_rating),
        lichess_rating: normalize(values.lichess_rating),
        chess_com_rating: normalize(values.chess_com_rating),
        experience_with_chess: normalize(values.experience_with_chess),
      };
    },

    validate: {
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

  useEffect(() => {
    if (data) {
      form.setValues({
        chess_com_rating: data.chess_com_rating || undefined,
        experience_with_chess: data.experience_with_chess || undefined,
        fide_rating: data.fide_rating || undefined,
        lichess_rating: data.lichess_rating || undefined,
      });
    }
  }, [data]);

  const handleUpdate = async (values: typeof form.values) => {
    setLoading(true);
    try {
      await mutateAsync(
        { data: values },
        {
          onError: (error) => {
            showNotification({
              title: "Update Failed",
              message:
                error.response?.data?.message ||
                "An error occurred during  profile update.",
              color: "red",
            });
          },
          onSuccess: () => {
            showNotification({
              title: "Profile Updated",
              message: "Your profile has been updated successfully.",
              color: "green",
            });
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
      mih="100vh"
      w="100vw"
      direction="column"
      bg="gray.0"
    >
      <Navbar />
      <Container size={520} w="100%" py="xl">
        <Flex align="center" justify="center" direction="column" h="100%">
          <Paper p="xl" radius="md" withBorder w="100%">
            <LoadingOverlay visible={isLoading} />
            <Title order={2} ta="center" mb="lg">
              User profile
            </Title>
            <form onSubmit={form.onSubmit(handleUpdate)}>
              <Stack>
                <TextInput
                  required
                  label="Username"
                  placeholder="Enter your username"
                  value={data?.username || ""}
                  disabled
                />
                <Badge color="orange">{data?.type}</Badge>
                <Text fw="bold" mt="md">
                  Optional Information
                </Text>
                <Text>
                  Not visible to other users, purely for statistical purposes.
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
                  Update
                </Button>
              </Stack>
            </form>
          </Paper>
        </Flex>
      </Container>
    </Flex>
  );
};
