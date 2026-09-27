# SAM 3 Visual Match

React Native Web interface for iterative same-image grounding. Draw a positive
box to find similar objects, add another positive box for a missed variation, or
draw a negative box to suppress a false-positive family.

Start the [backend](../backend/README.md), then run the Expo web app in another
terminal from the repository root:

```powershell
cd frontend
npm ci
npm run web
```

Open `http://localhost:8081`. The first image upload loads the local checkpoint
and can take longer than later prompt updates. Set `EXPO_PUBLIC_API_URL` when the
API is not running at `http://127.0.0.1:8000`.

Run `npm run typecheck` to check TypeScript and `npm run build:web` to export
the web build into `dist/`.
