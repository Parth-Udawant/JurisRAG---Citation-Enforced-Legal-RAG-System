export type ActKey = "all" | "bns" | "ica";

export interface Citation {
  citation: string;
  citation_id: string;
  act: string;
  section_number: string | null;
  section_title: string | null;
}

export interface Source {
  source_id: string;
  citation: string;
  act: string;
  chapter_number: string | null;
  chapter_title: string | null;
  section_number: string | null;
  section_title: string | null;
  subsection: string | null;
  content_type: string | null;
  content: string;
  citation_text: string;
}

export interface RetrievalInfo {
  strategy: string;
  candidate_k: number;
  final_k: number;
  act_filter: string | null;
  returned_sources: number;
}

export interface ChatResponse {
  request_id: string;
  conversation_id: string | null;
  answer: string;
  insufficient_evidence: boolean;
  citations: Citation[];
  sources: Source[];
  retrieval: RetrievalInfo;
  disclaimer: string;
}

export interface ChatRequest {
  question: string;
  act?: ActKey;
  conversation_id?: string | null;
}
