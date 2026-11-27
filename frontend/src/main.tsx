import { useEffect, useRef, useState } from 'react'
import type { FormEvent, KeyboardEvent } from 'react'
import { createRoot } from 'react-dom/client'
import {
  ArrowUpRight, Braces, Copy, Cpu, Database, Ellipsis, Github, Layers3,
  LockKeyhole, MessageCircle, Sparkles, Square, SquarePen,
} from 'lucide-react'
import type { ChatRequest, ChatUpdate, Turn } from './chat'
import '@fontsource-variable/dm-sans'
import '@fontsource/ibm-plex-mono/latin-500.css'
import '@fontsource/instrument-serif/latin-400.css'
import '@fontsource/instrument-serif/latin-400-italic.css'
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

const assets = import.meta.env.BASE_URL
const clock = new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' })

type ChatTurn = Turn & { sentAt: number; repliedAt?: number }

function Avatar({ className = '' }: { className?: string }) {
  return <img className={'avatar ' + className} src={assets + 'titus-avatar.webp'} alt="" />
}

function Timestamp({ value }: { value: number }) {
  const time = new Date(value)
  return <time className="timestamp" dateTime={time.toISOString()}>{clock.format(time)}</time>
}

function PaperPlane() {
  return (
    <svg width="21" height="21" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M21.15 2.91a.9.9 0 0 1 1.11 1.17l-6.9 17.4a.87.87 0 0 1-1.58.08l-3.48-5.95 6.47-6.47a.65.65 0 0 0-.92-.92l-6.47 6.47-6.04-3.31a.87.87 0 0 1 .06-1.56l17.75-6.91Z"/>
    </svg>
  )
}

function ModelIntro() {
  return (
    <section className="intro min-w-0" aria-labelledby="intro-heading">
      <p className="eyebrow"><span />A TINY LANGUAGE MODEL</p>
      <h1 id="intro-heading">Small model.<em>Real conversation.</em></h1>
      <p className="intro-description">
        A tiny language model trained from scratch for short, everyday conversations.
      </p>
      <p className="runtime"><Cpu size={17} /> Runs in your browser</p>
    </section>
  )
}

function ModelSpecifications() {
  return (
    <section className="specs min-w-0" aria-labelledby="specs-heading">
      <h2 id="specs-heading">Model specifications</h2>
      <dl className="grid grid-cols-2 gap-2.5 sm:grid-cols-4 lg:grid-cols-2 xl:grid-cols-4">
        {stats.map(({ value, label, Icon }) => (
          <div className="stat" key={label}>
            <Icon size={21} strokeWidth={1.65} aria-hidden="true" />
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
      <div className="sources mt-4 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
        <span className="text-muted">Chat training data</span>
        <a href="https://huggingface.co/datasets/allenai/soda" target="_blank" rel="noreferrer">SODA <ArrowUpRight size={12} /></a>
        <a href="https://datashare.ed.ac.uk/items/7daabcd9-bb5f-4106-adc3-45e831c5577c" target="_blank" rel="noreferrer">MaiChat <ArrowUpRight size={12} /></a>
      </div>
    </section>
  )
}

function Chat() {
  const [turns, setTurns] = useState<ChatTurn[]>([])
  const [draft, setDraft] = useState('')
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading')
  const [generating, setGenerating] = useState(false)
  const [issue, setIssue] = useState('')
  const [notice, setNotice] = useState('')
  const [menuOpen, setMenuOpen] = useState(false)
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
          const repliedAt = Date.now()
          setTurns((previous) => previous.map((turn, index) =>
            index === previous.length - 1 ? { ...turn, reply: data.text, repliedAt: turn.repliedAt ?? repliedAt } : turn
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
    const next = [...turns, { prompt, reply: '', sentAt: Date.now() }]
    setTurns(next)
    setDraft('')
    setIssue('')
    setNotice('')
    setGenerating(true)
    post({ type: 'generate', turns: next })
  }

  function newChat() {
    if (generating) post({ type: 'stop' })
    setTurns([])
    setDraft('')
    setIssue('')
    setNotice('')
    setMenuOpen(false)
    textarea.current?.focus()
  }

  async function copyConversation() {
    const transcript = turns.flatMap(({ prompt, reply }) =>
      reply ? ['You: ' + prompt, 'TitusAI: ' + reply] : ['You: ' + prompt]
    ).join('\n\n')
    setMenuOpen(false)
    try {
      await navigator.clipboard.writeText(transcript)
      setNotice('Conversation copied')
    } catch {
      setNotice('Couldn’t copy the conversation')
    }
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
    <section className="chat flex min-w-0 flex-col overflow-hidden" aria-labelledby="chat-heading">
      <header className="chat-header flex shrink-0 items-center justify-between gap-3 px-5 sm:px-6">
        <div className="flex min-w-0 items-center gap-3">
          <Avatar className="header-avatar" />
          <div>
            <h2 id="chat-heading" className="text-[17px] font-bold tracking-[-.025em] sm:text-[19px]">Chat with TitusAI</h2>
            <p className="model-status mt-0.5 flex items-center gap-1.5 text-[12px]" role="status">
              <span className={'status-dot ' + status} />
              {status === 'ready' ? 'Model ready' : status === 'loading' ? 'Loading model…' : 'Model unavailable'}
            </p>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <button className="icon-button" type="button" onClick={newChat} aria-label="New chat" title="New chat">
            <SquarePen size={18} strokeWidth={1.7} />
          </button>
          <div className="relative" onBlur={(event) => {
            if (!event.currentTarget.contains(event.relatedTarget)) setMenuOpen(false)
          }} onKeyDown={(event) => {
            if (event.key === 'Escape') {
              setMenuOpen(false)
              event.currentTarget.querySelector('button')?.focus()
            }
          }}>
            <button
              className="icon-button" type="button" aria-label="More chat actions" title="More chat actions"
              aria-expanded={menuOpen} onClick={() => setMenuOpen(!menuOpen)}
            >
              <Ellipsis size={20} />
            </button>
            {menuOpen && (
              <div className="chat-menu">
                <button type="button" disabled={turns.length === 0} onClick={copyConversation}>
                  <Copy size={15} /> Copy conversation
                </button>
              </div>
            )}
          </div>
        </div>
      </header>

      <div ref={log} className="chat-log flex min-h-0 flex-1 flex-col overflow-y-auto" role="log" aria-live="polite">
        {turns.length === 0 ? (
          <div className="empty-chat m-auto flex w-full flex-col items-center px-4 py-6 text-center">
            <div className="avatar-frame"><Avatar className="empty-avatar" /></div>
            <h3 className="empty-title mt-4">Say hello.</h3>
            {status === 'error' ? (
              <>
                <p className="mt-2 max-w-70 text-sm text-muted">The browser model couldn’t load. Check the model export and try again.</p>
                <button className="suggestion mt-4" onClick={() => { setStatus('loading'); setAttempt(attempt + 1) }}>Retry model load</button>
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
          <div className="space-y-5 px-5 py-5 sm:px-6">
            {turns.map((turn, index) => (
              <div className="space-y-4" key={index}>
                <div className="flex justify-end">
                  <div className="message-stack user-stack">
                    <p className="message user-message">{turn.prompt}</p>
                    <Timestamp value={turn.sentAt} />
                  </div>
                </div>
                {(turn.reply || (generating && index === turns.length - 1)) && (
                  <div className="flex items-start gap-2.5">
                    <Avatar className="reply-avatar" />
                    <div className="message-stack min-w-0">
                      {turn.reply ? (
                        <>
                          <p className="message model-message">{turn.reply}</p>
                          {turn.repliedAt !== undefined && <Timestamp value={turn.repliedAt} />}
                        </>
                      ) : (
                        <div className="typing message model-message" aria-label="TitusAI is typing"><span /><span /><span /></div>
                      )}
                    </div>
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
            ref={textarea} rows={1} aria-label="Message TitusAI" placeholder="Message TitusAI…"
            value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={handleKey}
          />
          {generating ? (
            <button type="button" className="send-button" onClick={() => post({ type: 'stop' })} aria-label="Stop response">
              <Square size={15} fill="currentColor" />
            </button>
          ) : (
            <button type="submit" className="send-button" disabled={status !== 'ready' || !draft.trim()} aria-label="Send message">
              <PaperPlane />
            </button>
          )}
        </form>
        <div className="mt-3 flex flex-wrap items-center justify-between gap-x-4 gap-y-2 px-0.5 text-[11px] text-muted">
          <span className="flex items-center gap-1.5" aria-live="polite"><LockKeyhole size={12} /> {notice || 'Chats stay on this device'}</span>
          <span className="hidden sm:inline"><kbd>Enter</kbd> to send <span className="mx-1">·</span> <kbd>Shift + Enter</kbd> for new line</span>
        </div>
      </footer>
    </section>
  )
}

function App() {
  return (
    <>
      <div className="scenic-background" aria-hidden="true" />
      <div className="site-shell relative z-10 mx-auto max-w-[1440px] px-4 sm:px-8 lg:px-12">
        <header className="site-header flex items-center justify-between gap-3">
          <div className="wordmark flex items-center gap-2.5">
            <img src={assets + 'titus-mark.svg'} width="44" height="44" alt="" />
            <span>Titus<span className="text-brand">AI</span></span>
          </div>
          <a className="repo-link" href="https://github.com/jaredturck/TitusAI" target="_blank" rel="noreferrer" aria-label="TitusAI on GitHub">
            <Github size={18} /> <span>GitHub</span> <ArrowUpRight size={13} />
          </a>
        </header>
        <main className="workspace">
          <ModelIntro />
          <Chat />
          <ModelSpecifications />
        </main>
      </div>
    </>
  )
}

createRoot(document.getElementById('root')!).render(<App />)
