export type Turn = { prompt: string; reply: string }

export type ChatRequest = { type: 'generate'; turns: Turn[] } | { type: 'stop' }

export type ChatUpdate =
  | { type: 'ready' }
  | { type: 'reply'; text: string }
  | { type: 'done' }
  | { type: 'error'; phase: 'load' | 'generate'; message: string }
