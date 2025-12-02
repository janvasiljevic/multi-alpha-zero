import { defineConfig } from 'orval';

export default defineConfig({
    api: {
        input: './../chess-tourney-server/api/schema/openapi.json',
        output: {
            target: './src/api/index.ts',
            schemas: 'src/api/model',
            mode: 'tags-split',
            client: 'react-query',
            override: {
                mutator: {
                    path: './src/api/mutator/custom-instance.ts',
                    name: 'customInstance',
                },
            },
        },
        hooks: {
            afterAllFilesWrite: 'prettier --write',
        },
    },
});
