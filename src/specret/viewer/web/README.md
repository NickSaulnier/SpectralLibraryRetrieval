# specret viewer (React)

The source of `specret view`'s web app (PLAN_VIEWER.md). You need Node only to change the
frontend: the build in `../static/app/` is committed, and FastAPI serves it at `/`. The original
single-page viewer stays at `/legacy` until the React app has replaced it.

```
pnpm install        # or npm install
pnpm run dev        # Vite dev server; proxies /api to `specret view` on 127.0.0.1:8765
pnpm test           # Vitest
pnpm run api        # regenerate src/api/schema.d.ts from FastAPI's OpenAPI schema
pnpm run build      # type-check, build into ../static/app, record the source hash
```

- **Always rebuild before committing.** Commit `../static/app/` together with any source change. `tests/test_phase3.py` fails when the committed build doesn't match these sources: the build writes a hash of them to `../static/app/source-hash.txt`.
- **Types:** route paths are typed from the generated `src/api/schema.d.ts`. Response bodies are typed by hand in `src/api/types.ts` until the endpoints have Pydantic models.
- **On this machine,** the global `npm` shim is broken (an nvm-windows symlink mixing Node 21 and 24), so run pnpm directly:

  ```
  node "C:/Program Files/nodejs/node_modules/pnpm/bin/pnpm.cjs" run build
  ```
