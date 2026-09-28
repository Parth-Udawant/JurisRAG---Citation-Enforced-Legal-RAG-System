"use client";

import { useState } from "react";
import { BookIcon, ChevronDownIcon } from "@/components/Icons";
import type { Source } from "@/types/api";

function shortAct(act: string) {
  if (act.includes("Bharatiya Nyaya")) return "BNS";
  if (act.includes("Contract Act")) return "ICA";
  return act;
}

export function SourceCard({ source, highlighted = false }: { source: Source; highlighted?: boolean }) {
  const [open, setOpen] = useState(highlighted);
  return (
    <article className={`source-card ${highlighted ? "source-card-highlight" : ""}`} id={`source-${source.source_id}`}>
      <button className="source-card-head" onClick={() => setOpen(v => !v)} aria-expanded={open}>
        <span className="source-icon"><BookIcon size={19} /></span>
        <span className="source-title-wrap">
          <span className="source-title">{source.section_number ? `Section ${source.section_number}` : source.citation}</span>
          <span className="source-subtitle">{shortAct(source.act)}{source.section_title ? ` · ${source.section_title}` : ""}</span>
        </span>
        <ChevronDownIcon size={18} />
      </button>
      {open && (
        <div className="source-card-body">
          <div className="source-meta">
            {source.chapter_number && <span>Chapter {source.chapter_number}</span>}
            {source.subsection && <span>{source.subsection}</span>}
            {source.content_type && <span>{source.content_type}</span>}
          </div>
          <p className="source-text">{source.content}</p>
          <div className="source-footer">{source.citation_text}</div>
        </div>
      )}
    </article>
  );
}
