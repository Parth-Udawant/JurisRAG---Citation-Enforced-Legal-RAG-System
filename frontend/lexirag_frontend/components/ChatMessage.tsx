"use client";

import type { ChatResponse, Citation } from "@/types/api";
import { AnswerContent } from "@/components/AnswerContent";
import { CheckIcon, InfoIcon, ShieldIcon, UserIcon, WarningIcon } from "@/components/Icons";
import { SourceCard } from "@/components/SourceCard";

export interface UserMessageData { id: string; question: string; time: string; }

export function UserMessage({ message }: { message: UserMessageData }) {
  return (
    <div className="message-row user-row">
      <div className="avatar user-avatar"><UserIcon size={22} /></div>
      <div className="user-bubble">
        <div>{message.question}</div>
        <time>{message.time}</time>
      </div>
    </div>
  );
}

export function ChatMessage({ response, onCitation }: { response: ChatResponse; onCitation: (citation: Citation) => void }) {
  const citationCount = response.citations.length;
  return (
    <div className="assistant-card">
      {response.insufficient_evidence ? (
        <>
          <div className="answer-status-row">
            <span className="status-pill insufficient"><WarningIcon size={15} /> Insufficient evidence</span>
          </div>
          <AnswerContent answer={response.answer} citations={[]} onCitation={onCitation} />
          <div className="insufficient-state">
            <div className="status-icon warning"><WarningIcon size={22} /></div>
            <div>
              <div className="status-title">Not grounded in the supplied corpus</div>
              <div className="status-text">The retrieved legal documents do not contain sufficient evidence to answer this question. No source citations are shown because they were not used to support the response.</div>
            </div>
          </div>
        </>
      ) : (
        <>
          <div className="answer-status-row">
            <span className="status-pill grounded"><CheckIcon size={15} /> Grounded</span>
            <span className="status-pill">{citationCount} {citationCount === 1 ? "citation" : "citations"}</span>
          </div>
          <AnswerContent answer={response.answer} citations={response.citations} onCitation={onCitation} />
          <div className="source-list">
            {response.sources.map(source => <SourceCard key={source.source_id} source={source} />)}
          </div>
          <div className="grounded-footer">
            <ShieldIcon size={21} />
            <span>This answer is grounded in {response.sources.length} source{response.sources.length === 1 ? "" : "s"} from the supplied legal corpus.</span>
            <InfoIcon size={17} />
          </div>
        </>
      )}
      <div className="answer-disclaimer">{response.disclaimer}</div>
    </div>
  );
}

// Backward-compatible alias for callers that use the assistant-specific name.
export const AssistantMessage = ChatMessage;
