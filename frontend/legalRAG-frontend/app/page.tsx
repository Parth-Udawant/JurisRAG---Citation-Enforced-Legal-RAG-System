"use client";

import { useMemo, useState } from "react";
import { AnswerContent } from "@/components/AnswerContent";
import { ChatMessage, UserMessage, type UserMessageData } from "@/components/ChatMessage";
import { ChevronDownIcon, InfoIcon, MenuIcon, PlusIcon, SendIcon, SparklesIcon, XIcon } from "@/components/Icons";
import { SourceCard } from "@/components/SourceCard";
import { sendChat } from "@/lib/api";
import type { ActKey, ChatResponse, Citation } from "@/types/api";

interface Turn { user: UserMessageData; response?: ChatResponse; }

const ACTS: { key: ActKey; label: string }[] = [
  { key: "all", label: "All Acts" },
  { key: "bns", label: "Bharatiya Nyaya Sanhita, 2023" },
  { key: "ica", label: "Indian Contract Act, 1872" },
];

function timeNow() { return new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" }).format(new Date()); }
function newId() { return typeof crypto !== "undefined" && crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`; }

export default function Home() {
  const [act, setAct] = useState<ActKey>("all");
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedSource, setSelectedSource] = useState<string | null>(null);
  const [mobileOpen, setMobileOpen] = useState(false);

  const selected = useMemo(() => {
    for (const turn of turns) {
      const found = turn.response?.sources.find(s => s.source_id === selectedSource);
      if (found) return found;
    }
    return null;
  }, [turns, selectedSource]);

  async function submit() {
    const q = question.trim();
    if (!q || loading) return;
    const user: UserMessageData = { id: newId(), question: q, time: timeNow() };
    setQuestion(""); setError(null); setLoading(true);
    const pending: Turn = { user };
    setTurns(prev => [...prev, pending]);
    try {
      const result = await sendChat(q, act, conversationId);
      setConversationId(prev => prev ?? result.conversation_id ?? newId());
      setTurns(prev => prev.map(t => t.user.id === user.id ? { ...t, response: result } : t));
    } catch (e) {
      const message = e instanceof Error ? e.message : "Something went wrong while contacting LexiRAG.";
      setError(message);
      setTurns(prev => prev.filter(t => t.user.id !== user.id));
      setQuestion(q);
    } finally { setLoading(false); }
  }

  function newChat() {
    setConversationId(null); setTurns([]); setQuestion(""); setError(null); setSelectedSource(null); setMobileOpen(false);
  }

  function handleCitation(citation: Citation) {
    setSelectedSource(citation.citation_id);
    requestAnimationFrame(() => document.getElementById(`source-${citation.citation_id}`)?.scrollIntoView({ behavior: "smooth", block: "center" }));
  }

  return (
    <main className="app-shell">
      <div className="ambient ambient-one" /><div className="ambient ambient-two" /><div className="ambient ambient-three" />
      <aside className={`sidebar ${mobileOpen ? "sidebar-open" : ""}`}>
        <button className="new-chat-button" onClick={newChat}><span className="plus-circle"><PlusIcon size={19} /></span><span>New Chat</span><kbd>⌘ N</kbd></button>
        <div className="sidebar-spacer" />
        <div className="brand-footer"><strong>LexiRAG</strong><span>v1.0</span><i /> <span>Made in India 🇮🇳</span></div>
      </aside>
      {mobileOpen && <button className="mobile-backdrop" aria-label="Close menu" onClick={() => setMobileOpen(false)} />}
      <section className="chat-panel">
        <header className="topbar">
          <button className="mobile-menu" onClick={() => setMobileOpen(true)} aria-label="Open menu"><MenuIcon size={22} /></button>
          <div className="brand-mark"><SparklesIcon size={30} /></div>
          <div className="topbar-copy"><h1>Chat with LexiRAG</h1><p>Get accurate, sourced answers from Indian law.</p></div>
          <div className="act-select-wrap">
            <span>Source</span>
            <div className="select-shell">
              <select value={act} onChange={e => setAct(e.target.value as ActKey)} disabled={loading} aria-label="Legal source filter">
                {ACTS.map(item => <option key={item.key} value={item.key}>{item.label}</option>)}
              </select>
              <ChevronDownIcon size={15} />
            </div>
          </div>
        </header>

        <div className="conversation" onClick={() => setSelectedSource(null)}>
          {turns.length === 0 ? (
            <div className="empty-state">
              <div className="empty-orb"><SparklesIcon size={34} /></div>
              <h2>Ask about the supplied legal corpus</h2>
              <p>Ask about the Bharatiya Nyaya Sanhita, 2023 or the Indian Contract Act, 1872.</p>
              <div className="suggestions">
                <button onClick={() => setQuestion("What is the punishment for murder under the BNS?")}>Punishment for murder</button>
                <button onClick={() => setQuestion("What is free consent under the Indian Contract Act?")}>Free consent</button>
                <button onClick={() => setQuestion("What constitutes criminal intimidation under the BNS?")}>Criminal intimidation</button>
              </div>
            </div>
          ) : turns.map(turn => (
            <div className="turn" key={turn.user.id}>
              <UserMessage message={turn.user} />
              {turn.response && <ChatMessage response={turn.response} onCitation={handleCitation} />}
            </div>
          ))}
          {loading && <div className="loading-card"><span className="loading-dots"><i /><i /><i /></span><span>Searching the legal corpus and preparing a grounded answer…</span></div>}
          {error && <div className="error-banner"><InfoIcon size={18} /><span>{error}</span><button onClick={() => setError(null)}><XIcon size={16} /></button></div>}
        </div>

        <div className="composer-area">
          <div className="composer">
            <textarea value={question} onChange={e => setQuestion(e.target.value)} onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } }} placeholder="Ask a follow-up question…" rows={1} disabled={loading} aria-label="Question" />
            <button className="send-button" onClick={submit} disabled={!question.trim() || loading} aria-label="Send question"><SendIcon size={24} /></button>
          </div>
          <p className="legal-note">LexiRAG provides informational answers grounded in the supplied legal documents and is not a substitute for professional legal advice.</p>
        </div>
      </section>

      {selected && (
        <aside className="source-drawer" onClick={e => e.stopPropagation()}>
          <div className="drawer-header"><div><span>Source</span><h2>{selected.citation}</h2></div><button onClick={() => setSelectedSource(null)} aria-label="Close source"><XIcon size={19} /></button></div>
          <SourceCard source={selected} highlighted />
        </aside>
      )}
    </main>
  );
}
