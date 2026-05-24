# ANPR Web (Next.js 15)

Premium dashboard for the Colombian ANPR system. Talks to the FastAPI
backend over REST and renders detections in real time.

## Local development

```bash
cd apps/web
npm install --legacy-peer-deps
cp .env.example .env.local
# edit .env.local: NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev
```

Open <http://localhost:3000>.

## Routes

| Path | Description |
|------|-------------|
| `/` | Landing page (hero, capability grid, pipeline diagram) |
| `/dashboard` | Upload / webcam / sample picker + live detection canvas |
| `/history` | Filterable, exportable detection log |

## Type generation

The frontend uses hand-written types in [lib/types.ts](lib/types.ts) by
default — they mirror the FastAPI Pydantic schemas. To regenerate from
the live `/api/openapi.json`:

```bash
# Backend must be running locally
npm run gen:types
```

This writes `lib/api-types.ts` for advanced typed clients; the simple
`api` wrapper in [lib/api.ts](lib/api.ts) sticks to the hand-written
types so the frontend builds without a network round-trip.

## Visual language

- Dark-by-default. Theme variables live in [app/globals.css](app/globals.css).
- Bbox colors mirror the original Streamlit overlay:
  - **blue** vehicles · **green** plates · **orange** brand logos
  (see [lib/draw.ts](lib/draw.ts))
- Framer Motion handles hero entrance, panel transitions, and the
  detection card stagger.
- shadcn/ui primitives in [components/ui/](components/ui/).
