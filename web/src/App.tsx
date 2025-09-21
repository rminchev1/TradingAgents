import React, { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeSanitize from 'rehype-sanitize'

type Message = {
  id: string
  role: 'user' | 'assistant'
  content: string
  agent?: string
}

const apiBase = () => {
  // If Vite proxy is configured, relative "/api" works; otherwise use env
  const env = (import.meta as any).env?.VITE_API_URL as string | undefined
  return env ? env.replace(/\/$/, '') : ''
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [convId, setConvId] = useState<string | null>(null)
  const [streaming, setStreaming] = useState(false)
  const esRef = useRef<EventSource | null>(null)
  const listRef = useRef<HTMLDivElement | null>(null)

  // Auto-scroll to bottom
  useEffect(() => {
    const el = listRef.current
    if (el) {
      el.scrollTop = el.scrollHeight
    }
  }, [messages])

  const startSSE = (q: string) => {
    if (esRef.current) {
      esRef.current.close()
      esRef.current = null
    }
    setStreaming(true)
    const params = new URLSearchParams({ q })
    if (convId) params.set('conv_id', convId)
    const url = `${apiBase()}/api/chat-stream?${params.toString()}`.replace('//api', '/api')
    const es = new EventSource(url)
    esRef.current = es

    es.onmessage = (ev) => {
      if (!ev.data) return
      try {
        const payload = JSON.parse(ev.data)
        if (payload.type === 'start') {
          if (payload.conv_id) setConvId(payload.conv_id)
          return
        }
        if (payload.type === 'message') {
          const agent = payload.agent || undefined
          const text = payload.content || ''
          setMessages((m) => m.concat({ id: crypto.randomUUID(), role: 'assistant', content: text, agent }))
          return
        }
        if (
          payload.type === 'market_report' ||
          payload.type === 'sentiment_report' ||
          payload.type === 'news_report' ||
          payload.type === 'fundamentals_report' ||
          payload.type === 'investment_plan' ||
          payload.type === 'trader_investment_plan'
        ) {
          const agent = payload.agent || undefined
          const text = String(payload.value || '')
          setMessages((m) => m.concat({ id: crypto.randomUUID(), role: 'assistant', content: text, agent }))
          return
        }
        if (payload.type === 'done') {
          const agent = payload.agent || 'Risk Manager'
          const decision = payload.decision_processed || payload.decision || ''
          const text = `Final decision: ${decision}`
          setMessages((m) => m.concat({ id: crypto.randomUUID(), role: 'assistant', content: text, agent }))
          return
        }
        if (payload.type === 'error') {
          setMessages((m) => m.concat({ id: crypto.randomUUID(), role: 'assistant', content: `Error: ${payload.message}` }))
          return
        }
      } catch (e) {
        // ignore parse errors
      }
    }

    es.addEventListener('end', () => {
      setStreaming(false)
      es.close()
      if (esRef.current === es) esRef.current = null
    })

    es.onerror = () => {
      setStreaming(false)
      es.close()
      if (esRef.current === es) esRef.current = null
    }
  }

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const q = input.trim()
    if (!q) return
    setMessages((m) => m.concat({ id: crypto.randomUUID(), role: 'user', content: q }))
    setInput('')
    startSSE(q)
  }

  const stop = () => {
    if (esRef.current) {
      esRef.current.close()
      esRef.current = null
      setStreaming(false)
    }
  }

  return (
    <div className="app">
      <header className="app-header">TradingAgents — Fund Representative</header>
      <div className="chat" ref={listRef}>
        {messages.map((m) => (
          <div key={m.id} className={`msg ${m.role}`}>
            <div className="meta">
              {m.role === 'user' ? 'You' : m.agent || 'Assistant'}
            </div>
            <div className="bubble">
              {m.role === 'assistant' ? (
                <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeSanitize]}>
                  {m.content}
                </ReactMarkdown>
              ) : (
                <span>{m.content}</span>
              )}
            </div>
          </div>
        ))}
      </div>

      <form className="composer" onSubmit={onSubmit}>
        <input
          type="text"
          placeholder={streaming ? 'Streaming…' : 'Ask about a stock, goals, period…'}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={streaming}
        />
        <button type="submit" disabled={streaming || !input.trim()}>
          Send
        </button>
        <button type="button" onClick={stop} disabled={!streaming} className="ghost">
          Stop
        </button>
      </form>
    </div>
  )
}
 
