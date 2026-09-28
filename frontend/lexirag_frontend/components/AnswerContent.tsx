"use client";

import type React from "react";
import type { Citation } from "@/types/api";

function renderInline(text: string, citations: Citation[], onCitation: (citation: Citation) => void) {
  const regex = /\[(BNS|ICA)\s+§([0-9A-Za-z-]+)\]/g;
  const parts: React.ReactNode[] = [];
  let last = 0;
  let match: RegExpExecArray | null;
  let key = 0;
  while ((match = regex.exec(text)) !== null) {
    if (match.index > last) parts.push(<span key={`t-${key++}`}>{text.slice(last, match.index)}</span>);
    const label = `[${match[1]} §${match[2]}]`;
    const citation = citations.find(c => c.citation === label);
    parts.push(
      citation ? (
        <button key={`c-${key++}`} className="citation-link" onClick={() => onCitation(citation)} title={`Open ${label}`}>
          {label}
        </button>
      ) : <span key={`u-${key++}`} className="citation-link-disabled">{label}</span>
    );
    last = regex.lastIndex;
  }
  if (last < text.length) parts.push(<span key={`t-${key++}`}>{text.slice(last)}</span>);
  return parts;
}

export function AnswerContent({ answer, citations, onCitation }: { answer: string; citations: Citation[]; onCitation: (citation: Citation) => void }) {
  const blocks = answer.split(/\n\s*\n/);
  return (
    <div className="answer-content">
      {blocks.map((block, i) => {
        const lines = block.split("\n");
        return (
          <p key={i}>
            {lines.map((line, j) => {
              const clean = line.replace(/^[-*]\s+/, "");
              const isBullet = /^[-*]\s+/.test(line);
              const formatted = clean.split(/(\*\*[^*]+\*\*)/g).map((part, k) => {
                if (part.startsWith("**") && part.endsWith("**")) {
                  return <strong key={k}>{renderInline(part.slice(2, -2), citations, onCitation)}</strong>;
                }
                return <span key={k}>{renderInline(part, citations, onCitation)}</span>;
              });
              return <span key={j} className={isBullet ? "answer-bullet" : "answer-line"}>{formatted}{j < lines.length - 1 && <br />}</span>;
            })}
          </p>
        );
      })}
    </div>
  );
}
