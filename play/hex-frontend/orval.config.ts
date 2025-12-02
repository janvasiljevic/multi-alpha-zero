import { defineConfig } from "orval";

export default defineConfig({
  be: {
    input: {
      target: "http://127.0.0.1:3000/docs/private/api.json",
      validation: false,
    },
    output: {
      target: "./src/api/def/index.ts",
      schemas: "src/api/def/model",
      mode: "tags-split",
      client: "react-query",
      override: {
        mutator: {
          path: "./src/api/mutator/custom-instance.ts",
          name: "customInstance",
        },
      },
    },
    hooks: {
      afterAllFilesWrite: "prettier --write",
    },
  },
});
