export type Source = {
  document: string;
  page: number;
  chunk_id: string;
  rerank_score: number | null;
  text: string;
};

export type PlanStep = {
  tool: string;
  input: string;
  purpose: string;
  ok: boolean;
  output: string;
};

export type Replan = {
  step: number;
  reason: string;
  old_input: string;
  new_input: string;
  recovered: boolean;
};

export type Trace = {
  route: string | null;
  route_reason: string | null;
  used: boolean;
  queries: string[];
  rewritten_query: string | null;
  hyde_used: boolean;
  hyde_document: string | null;
  fused: number;
  reranked: number;
  compressed: boolean;
  llm_calls: number;
  seconds?: number;
  agent?: boolean;
  plan?: { tool: string; input: string; purpose: string }[];
  steps?: PlanStep[];
  replanned?: Replan[];
  tool_notes?: string[];
  chunks?: Source[];
};

export type Payload = {
  question: string;
  answer: string;
  answered: boolean;
  sources: Source[];
  retrieval?: Trace;
  conversation_id?: string;
};

export type Message = {
  role: "user" | "assistant";
  content: string;
  payload?: Payload;
  created_at?: string;
  /** set while the answer is still streaming in */
  pending?: boolean;
};

export type Conversation = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: number;
};

export type Settings = {
  model: string;
  utility_model: string;
  embedding_model: string;
  reranker: string;
  agent: boolean;
  query_rewrite: boolean;
  multi_query: boolean;
  hyde: boolean;
  compression: boolean;
  compression_mode: string;
};

export type Health = {
  status: string;
  chunks: number;
  ingested: boolean;
};

export type StreamEvent =
  | { type: "start"; conversation_id: string }
  | { type: "retrieval"; trace: Trace; tool_notes: string[]; chunks: Source[] }
  | { type: "token"; text: string }
  | { type: "done"; payload: Payload }
  | { type: "error"; message: string };
