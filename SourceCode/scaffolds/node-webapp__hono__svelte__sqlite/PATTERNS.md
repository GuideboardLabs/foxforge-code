# Stack patterns — Hono + Svelte + SQLite

## Directory structure

```
src/
  server/
    index.ts         # Hono app, route registration
    routes/
      users.ts       # Hono route handlers per resource
      items.ts
    db/
      schema.ts      # Drizzle schema definitions
      index.ts       # Drizzle client singleton
    middleware/
      auth.ts        # Hono middleware
  lib/               # Shared utilities
  routes/            # SvelteKit page routes
    +page.svelte
    +layout.svelte
    api/             # SvelteKit API routes (if not using separate Hono server)
  components/        # Reusable Svelte components
static/
drizzle.config.ts    # Drizzle ORM config
package.json
```

## Hono route pattern

```typescript
// src/server/routes/users.ts
import { Hono } from "hono";
import { db } from "../db";
import { users } from "../db/schema";
import { eq } from "drizzle-orm";

const app = new Hono();

app.get("/", async (c) => {
  const all = await db.select().from(users);
  return c.json(all);
});

app.post("/", async (c) => {
  const body = await c.req.json();
  const [user] = await db.insert(users).values(body).returning();
  return c.json(user, 201);
});

export default app;
```

## Drizzle schema — source of truth

```typescript
// src/server/db/schema.ts
import { sqliteTable, text, integer } from "drizzle-orm/sqlite-core";

export const users = sqliteTable("users", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  username: text("username").notNull().unique(),
  email: text("email").notNull().unique(),
  createdAt: text("created_at").notNull().$defaultFn(() => new Date().toISOString()),
});

export const milestones = sqliteTable("milestones", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  userId: integer("user_id").notNull().references(() => users.id),
  name: text("name").notNull(),
  completed: integer("completed", { mode: "boolean" }).notNull().default(false),
});
```

After schema changes: `npx drizzle-kit generate` then `npx drizzle-kit migrate`.

## Drizzle client singleton

```typescript
// src/server/db/index.ts
import Database from "better-sqlite3";
import { drizzle } from "drizzle-orm/better-sqlite3";
import * as schema from "./schema";

const sqlite = new Database("app.db");
export const db = drizzle(sqlite, { schema });
```

## Svelte component pattern

```svelte
<!-- src/components/MilestoneList.svelte -->
<script lang="ts">
  export let milestones: { id: number; name: string; completed: boolean }[];
</script>

<ul>
  {#each milestones as m (m.id)}
    <li class:done={m.completed}>{m.name}</li>
  {/each}
</ul>
```

Prefer typed `export let` props over stores for simple data flow.

## Naming conventions

- Route files: `resource.ts` in `server/routes/`
- Schema tables: camelCase export, snake_case DB name — `export const trainingLogs = sqliteTable("training_logs", ...)`
- Svelte components: PascalCase — `MilestoneCard.svelte`
- Hono app entrypoint: `index.ts`, exports `app`

## Common mistakes to avoid

- Do NOT use `new Database()` in every route — use the singleton from `db/index.ts`
- Do NOT modify generated Drizzle migration files — re-generate instead
- Do NOT mix SvelteKit API routes and Hono routes — pick one pattern and stick to it
- Do NOT store secrets in `src/` — use `.env` (gitignored)
