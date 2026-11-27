import { SentencePieceProcessor } from '@sctg/sentencepiece-js'
import * as ort from 'onnxruntime-web/wasm'
import runtimeModule from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url'
import runtimeBinary from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url'
import type { ChatRequest, ChatUpdate, Turn } from './chat'

const contextLength = 512
const replyLimit = 48
const eos = 2
const tokenizer = new SentencePieceProcessor()
let stopped = false

function send(update: ChatUpdate) {
  self.postMessage(update)
}

async function loadModel() {
  ort.env.wasm.numThreads = 1
  ort.env.wasm.wasmPaths = { mjs: runtimeModule, wasm: runtimeBinary }
  const assets = import.meta.env.BASE_URL + 'model/'
  const [session, response] = await Promise.all([
    ort.InferenceSession.create(assets + 'titusai.onnx', { executionProviders: ['wasm'] }),
    fetch(assets + 'tokenizer.model'),
  ])
  if (!response.ok) throw new Error('The tokenizer file could not be loaded.')
  const bytes = new Uint8Array(await response.arrayBuffer())
  await tokenizer.loadFromB64StringModel(btoa(Array.from(bytes, (byte) => String.fromCharCode(byte)).join('')))
  send({ type: 'ready' })
  return session
}

function promptIds(turns: Turn[]) {
  const assistant = tokenizer.encodeIds('Assistant:')
  const user = tokenizer.encodeIds('User:')
  const budget = contextLength - replyLimit - assistant.length - 1
  let prompt = tokenizer.encodeIds('User: ' + turns[turns.length - 1].prompt)
  if (prompt.length > budget) prompt = [...user, ...prompt.slice(-(budget - user.length))]

  for (const turn of turns.slice(0, -1).reverse()) {
    if (!turn.reply) continue
    const previous = [
      ...tokenizer.encodeIds('User: ' + turn.prompt),
      ...assistant,
      ...tokenizer.encodeIds(turn.reply),
      eos,
    ]
    if (previous.length + prompt.length > budget) break
    prompt = [...previous, ...prompt]
  }
  return [1, ...prompt, ...assistant]
}

function nextToken(logits: Float32Array, allowEnd: boolean) {
  const candidates = Array.from(logits, (score, id) => ({ score, id }))
    .filter(({ id }) => id > 3 || (allowEnd && id === eos))
    .sort((first, second) => second.score - first.score)
    .slice(0, 24)
  const peak = candidates[0].score
  const weights = candidates.map(({ score }) => Math.exp((score - peak) / 0.75))
  let choice = Math.random() * weights.reduce((total, weight) => total + weight, 0)
  for (const [index, weight] of weights.entries()) {
    choice -= weight
    if (choice <= 0) return candidates[index].id
  }
  return candidates[0].id
}

async function generate(session: ort.InferenceSession, turns: Turn[]) {
  const input = promptIds(turns)
  const reply: number[] = []
  for (let index = 0; index < replyLimit && !stopped; index++) {
    const ids = BigInt64Array.from(input, BigInt)
    const { logits } = await session.run({ input_ids: new ort.Tensor('int64', ids, [1, ids.length]) })
    if (stopped) break
    const token = nextToken(logits.data as Float32Array, reply.length >= 2)
    if (token === eos) break
    reply.push(token)
    input.push(token)
    const decoded = tokenizer.decodeIds(Int32Array.from(reply)) as string
    const boundary = decoded.search(/\b(?:User|Assistant):/i)
    send({ type: 'reply', text: (boundary < 0 ? decoded : decoded.slice(0, boundary)).trim() })
    if (boundary >= 0) break
    await new Promise((resolve) => setTimeout(resolve, 0))
  }
  send({ type: 'done' })
}

const session = loadModel().catch((error: Error) => {
  send({ type: 'error', phase: 'load', message: error.message })
})

self.onmessage = async ({ data }: MessageEvent<ChatRequest>) => {
  if (data.type === 'stop') {
    stopped = true
    return
  }
  const model = await session
  if (!model) return
  stopped = false
  try {
    await generate(model, data.turns)
  } catch (error) {
    send({ type: 'error', phase: 'generate', message: String(error) })
  }
}
