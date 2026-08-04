"use client";

import type { Trace as TraceData } from "@/lib/types";

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="trace-row">
      <div className="trace-label">{label}</div>
      <div className="trace-value">{children}</div>
    </div>
  );
}

/**
 * The pipeline made a series of decisions to produce this answer. Showing
 * them is what makes a wrong answer debuggable instead of mysterious.
 */
export default function Trace({ trace }: { trace: TraceData }) {
  const steps = trace.steps ?? [];
  const replanned = trace.replanned ?? [];

  return (
    <details className="trace">
      <summary>
        <span className={`pill ${trace.used ? "ok" : "warn"}`}>
          {trace.route ?? "unknown"}
        </span>
        {trace.agent ? <span className="pill">agent</span> : null}
        {trace.hyde_used ? <span className="pill">hyde</span> : null}
        {trace.compressed ? <span className="pill">compressed</span> : null}
        <span>{trace.llm_calls} llm calls</span>
        {typeof trace.seconds === "number" ? (
          <span>· {trace.seconds}s</span>
        ) : null}
        {trace.reranked ? <span>· {trace.reranked} chunks</span> : null}
      </summary>

      <div className="trace-body">
        {trace.route_reason ? <Row label="Why">{trace.route_reason}</Row> : null}

        {steps.length > 0 ? (
          <Row label="Plan">
            {steps.map((step, index) => (
              <div key={index}>
                <span className={`pill ${step.ok ? "ok" : "bad"}`}>
                  {step.ok ? "ok" : "failed"}
                </span>
                <span className="mono">
                  {step.tool}({step.input})
                </span>
                {step.purpose ? (
                  <div className="excerpt">{step.purpose}</div>
                ) : null}
              </div>
            ))}
          </Row>
        ) : null}

        {replanned.length > 0 ? (
          <Row label="Replan">
            {replanned.map((entry, index) => (
              <div key={index}>
                <span className={`pill ${entry.recovered ? "ok" : "bad"}`}>
                  step {entry.step}
                </span>
                {entry.reason}
                <div className="excerpt mono">
                  {entry.old_input} → {entry.new_input}
                </div>
              </div>
            ))}
          </Row>
        ) : null}

        {trace.queries?.length ? (
          <Row label="Queries">
            {trace.queries.map((query, index) => (
              <span className="pill mono" key={index}>
                {query}
              </span>
            ))}
          </Row>
        ) : null}

        {trace.tool_notes?.length ? (
          <Row label="Tools">
            {trace.tool_notes.map((note, index) => (
              <div className="mono" key={index}>
                {note}
              </div>
            ))}
          </Row>
        ) : null}

        {trace.hyde_document ? (
          <Row label="HyDE">
            <div className="excerpt">{trace.hyde_document}</div>
          </Row>
        ) : null}

        {trace.chunks?.length ? (
          <Row label="Retrieved">
            {trace.chunks.map((chunk, index) => (
              <details key={index}>
                <summary className="mono">
                  [{index + 1}] {chunk.document} p.{chunk.page}
                  {typeof chunk.rerank_score === "number"
                    ? ` · ${chunk.rerank_score.toFixed(2)}`
                    : ""}
                </summary>
                <div className="excerpt">{chunk.text}</div>
              </details>
            ))}
          </Row>
        ) : null}
      </div>
    </details>
  );
}
