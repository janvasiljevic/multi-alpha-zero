import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Notifications, showNotification } from "@mantine/notifications";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { AxiosError } from "axios";
import { IconBug } from "@tabler/icons-react";
import { MantineProvider } from "@mantine/core";
import { RouterProvider } from "@tanstack/react-router";
import { router } from "./routes.tsx";

import "@mantine/core/styles.css";
import "@mantine/notifications/styles.css";

export const queryClient = new QueryClient({
  defaultOptions: {
    mutations: {
      onSettled(_, error) {
        const axError = error as AxiosError;

        if (!axError) return;
        if (!axError.response) return;

        if (axError.response?.status === 403) {
          showNotification({
            title: "Unauthorized",
            message: "You are not authorized to perform this action.",
            icon: <IconBug />,
            id: "unauthorized",
          });
        }
      },
    },
    queries: {
      retry: (failureCount, error) => {
        const axError = error as AxiosError;

        // Dont retry on the following errors
        switch (axError.response?.status) {
          case 401: // Unauthorized
          case 403: // Forbidden
          case 404: // Not found
            return false;
        }

        return failureCount < 2;
      },
    },
  },
});


createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <MantineProvider>
        <Notifications />
        <RouterProvider router={router} />
      </MantineProvider>
    </QueryClientProvider>
  </StrictMode>,
);
