import { useEffect, useRef, useState } from 'react'
import type { FormEvent, KeyboardEvent } from 'react'
import { createRoot } from 'react-dom/client'
import {
  ArrowUp, ArrowUpRight, Asterisk, Braces, Cpu, Database, Layers3, LockKeyhole,
  MessageCircle, Plus, Sparkles, Square,
} from 'lucide-react'
import type { ChatRequest, ChatUpdate, Turn } from './chat'
import '@fontsource-variable/dm-sans'
import '@fontsource/ibm-plex-mono/500.css'
import './style.css'

const stats = [
  { value: '17M', label: 'Parameters', Icon: Layers3 },
  { value: '300M', label: 'Pretraining tokens', Icon: Database },
  { value: '512', label: 'Context tokens', Icon: Braces },
  { value: 'INT8', label: 'Browser weights', Icon: Cpu },
]

const suggestions = [
  { text: 'Hey, how’s it going?', Icon: MessageCircle },
  { text: 'Tell me a fun fact', Icon: Sparkles },
  { text: 'Recommend a movie', Icon: ArrowUpRight },
]

function Mark({ className = '' }: { className?: string }) {
  return <span className={'mark ' + className} aria-hidden="true"><Asterisk size="65%" strokeWidth={2.6} /></span>
}

function ModelInfo() {
  return (
    <aside className="grid min-w-0 gap-4" aria-label="About the model">
      <section className="hero relative isolate overflow-hidden rounded-[24px]">
        <img className="hero-art" src={import.meta.env.BASE_URL + 'model-art.webp'} alt="" />
        <div className="hero-copy relative">
          <p className="eyebrow mb-4 flex items-center gap-2.5"><span className="hero-pip" />TITUSAI MODEL</p>
          <h1>Small model.<br /><em>Real conversation.</em></h1>
          <p className="hero-description mt-4">
            A tiny language model trained from scratch for short, everyday conversations.
          </p>
        </div>
      </section>

      <section className="specs panel rounded-[24px]" aria-labelledby="specs-heading">
        <div className="px-5 pb-3.5 pt-5 sm:px-6">
          <h2 id="specs-heading" className="text-[20px] font-bold tracking-[-.04em]">Inside the model</h2>
          <p className="mt-1 text-[13px] text-muted">Six transformer layers and an 8,000-token vocabulary.</p>
        </div>
        <dl className="mx-5 grid grid-cols-2 gap-2.5 sm:mx-6">
          {stats.map(({ value, label, Icon }) => (
            <div className="stat" key={label}>
              <Icon size={16} aria-hidden="true" />
              <dd>{value}</dd>
              <dt>{label}</dt>
            </div>
          ))}
        </dl>
        <div className="sources mx-5 mt-4 flex flex-wrap items-center gap-x-2 gap-y-2 py-3.5 sm:mx-6">
          <span className="mr-auto text-[12px] font-medium text-muted">Chat training data</span>
          <a href="https://huggingface.co/datasets/allenai/soda" target="_blank" rel="noreferrer">SODA <ArrowUpRight size={12} /></a>
          <a href="https://datashare.ed.ac.uk/items/7daabcd9-bb5f-4106-adc3-45e831c5577c" target="_blank" rel="noreferrer">MaiChat <ArrowUpRight size={12} /></a>
        </div>
      </section>
    </aside>
  )
}

function Chat() {
  const [turns, setTurns] = useState<Turn[]>([])
  const [draft, setDraft] = useState('')
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading')
  const [generating, setGenerating] = useState(false)
  const [issue, setIssue] = useState('')
  const [attempt, setAttempt] = useState(0)
  const worker = useRef<Worker | null>(null)
  const textarea = useRef<HTMLTextAreaElement>(null)
  const log = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const process = new Worker(new URL('./model.worker.ts', import.meta.url), { type: 'module' })
    worker.current = process
    process.onmessage = ({ data }: MessageEvent<ChatUpdate>) => {
      switch (data.type) {
        case 'ready':
          setStatus('ready')
          break
        case 'reply':
          setTurns((previous) => previous.map((turn, index) =>
            index === previous.length - 1 ? { ...turn, reply: data.text } : turn
          ))
          break
        case 'done':
          setGenerating(false)
          break
        case 'error':
          if (data.phase === 'load') setStatus('error')
          else setIssue('The model stopped before it could finish. Try sending another message.')
          setGenerating(false)
          break
      }
    }
    process.onerror = () => {
      setStatus('error')
      setGenerating(false)
    }
    return () => process.terminate()
  }, [attempt])

  useEffect(() => {
    if (log.current) log.current.scrollTop = log.current.scrollHeight
  }, [turns, issue])

  function post(request: ChatRequest) {
    worker.current?.postMessage(request)
  }

  function send(text: string) {
    const prompt = text.trim()
    if (status !== 'ready' || generating || !prompt) return
    const next = [...turns, { prompt, reply: '' }]
    setTurns(next)
    setDraft('')
    setIssue('')
    setGenerating(true)
    post({ type: 'generate', turns: next })
  }

  function newChat() {
    if (generating) post({ type: 'stop' })
    setTurns([])
    setDraft('')
    setIssue('')
    textarea.current?.focus()
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    send(draft)
  }

  function handleKey(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault()
      send(draft)
    }
  }

  return (
    <section className="chat panel flex min-w-0 flex-col overflow-hidden rounded-[24px]" aria-labelledby="chat-heading">
      <header className="chat-header flex shrink-0 items-center justify-between gap-3 px-5 sm:px-6">
        <div className="flex min-w-0 items-center gap-3">
          <Mark className="header-mark shrink-0" />
          <div>
            <h2 id="chat-heading" className="text-[18px] font-bold tracking-[-.035em]">Chat with TitusAI</h2>
            <p className="model-status mt-0.5 flex items-center gap-1.5 text-[12px]" role="status">
              <span className={'status-dot ' + status} />
              {status === 'ready' ? 'Model ready' : status === 'loading' ? 'Loading model…' : 'Model unavailable'}
            </p>
          </div>
        </div>
        <button className="new-chat shrink-0" type="button" onClick={newChat}>
          <Plus size={16} /> <span>New chat</span>
        </button>
      </header>

      <div ref={log} className="chat-log flex min-h-0 flex-1 flex-col overflow-y-auto" role="log" aria-live="polite">
        {turns.length === 0 ? (
          <div className="empty-chat m-auto flex w-full flex-col items-center px-4 py-6 text-center">
            <div className="empty-orbit"><Mark /></div>
            <h3 className="mt-5 text-[31px] font-bold leading-tight tracking-[-.06em]">Say hello.</h3>
            {status === 'error' ? (
              <>
                <p className="mt-2 max-w-70 text-sm text-muted">The browser model couldn’t load. Check the model export and try again.</p>
                <button className="new-chat mt-4" onClick={() => { setStatus('loading'); setAttempt(attempt + 1) }}>Retry model load</button>
              </>
            ) : (
              <>
                <p className="mt-2 text-sm text-muted">Pick a prompt or start with your own.</p>
                <div className="mt-6 flex flex-wrap justify-center gap-2">
                  {suggestions.map(({ text, Icon }) => (
                    <button type="button" className="suggestion" key={text} disabled={status !== 'ready'} onClick={() => send(text)}>
                      <Icon size={14} /> {text}
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>
        ) : (
          <div className="space-y-6 px-5 py-6 sm:px-6">
            {turns.map((turn, index) => (
              <div className="space-y-5" key={index}>
                <div className="flex justify-end"><p className="message user-message">{turn.prompt}</p></div>
                {(turn.reply || (generating && index === turns.length - 1)) && (
                  <div className="flex items-start gap-2.5">
                    <Mark className="reply-mark shrink-0" />
                    {turn.reply ? <p className="message model-message">{turn.reply}</p> : (
                      <div className="typing message model-message" aria-label="TitusAI is typing"><span /><span /><span /></div>
                    )}
                  </div>
                )}
              </div>
            ))}
            {issue && <p className="text-sm text-red-300" role="alert">{issue}</p>}
          </div>
        )}
      </div>

      <footer className="chat-footer shrink-0 px-4 pb-4 sm:px-5">
        <form onSubmit={submit} className="composer flex items-end gap-2">
          <textarea
            ref={textarea} rows={2} aria-label="Message TitusAI" placeholder="Message TitusAI…"
            value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={handleKey}
          />
          {generating ? (
            <button type="button" className="send-button" onClick={() => post({ type: 'stop' })} aria-label="Stop response">
              <Square size={15} fill="currentColor" />
            </button>
          ) : (
            <button type="submit" className="send-button" disabled={status !== 'ready' || !draft.trim()} aria-label="Send message">
              <ArrowUp size={21} strokeWidth={2.2} />
            </button>
          )}
        </form>
        <div className="mt-3 flex flex-wrap items-center justify-between gap-x-4 gap-y-2 px-0.5 text-[11px] text-muted">
          <span className="flex items-center gap-1.5"><LockKeyhole size={12} /> Chats stay on this device</span>
          <span className="hidden sm:inline"><kbd>Enter</kbd> to send <span className="mx-1">·</span> <kbd>Shift + Enter</kbd> for new line</span>
        </div>
      </footer>
    </section>
  )
}

function App() {
  return (
    <div className="mx-auto max-w-[1336px] px-4 sm:px-7 lg:px-10">
      <header className="site-header flex items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <Mark />
          <span className="text-[23px] font-extrabold tracking-[-.075em]">Titus<span className="text-brand">AI</span></span>
        </div>
        <div className="runtime flex items-center gap-2">
          <Cpu size={15} aria-hidden="true" />
          <span>On-device inference</span>
        </div>
      </header>
      <main className="grid items-stretch gap-5 pb-7 pt-5 lg:grid-cols-[minmax(0,.4fr)_minmax(0,.6fr)] lg:gap-6 lg:pt-7">
        <ModelInfo />
        <Chat />
      </main>
    </div>
  )
}

createRoot(document.getElementById('root')!).render(<App />)
