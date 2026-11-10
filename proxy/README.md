# Meeting Notes summary proxy

Cloudflare Worker that writes the notes for the app's "Cloud" provider. It holds the Anthropic API
key, wraps the transcript in a fixed prompt, and enforces the free tier (5 meetings per device).
Only transcript text reaches it, never audio; it stores counters only, never transcripts.

```
cd proxy && npm install
npx wrangler kv namespace create KV        # paste the id into wrangler.toml
npx wrangler secret put ANTHROPIC_API_KEY
npm test                                    # worker tests (fake KV, fake Anthropic)
npm run deploy
```

Then in the app's `.env`: `CLOUD_API_URL=https://<your-worker>.workers.dev` (and `CLOUD_INVITE_CODE=...`
if you set `INVITE_CODES` in `wrangler.toml` to restrict the beta).

Change the notes prompt in `summarizer.py`, then run `python3 proxy/gen_prompts.py` to sync it here.
KV counters are not atomic, so limits are approximate. That is fine for a small beta; move the
counters to a Durable Object before charging money against them.
