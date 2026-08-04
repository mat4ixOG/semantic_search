"use client";

import type { Message as MessageData } from "@/lib/types";
import Trace from "./Trace";

export default function Message({ message }: { message: MessageData }) {
  if (message.role === "user") {
    return (
      <div className="message user">
        <div className="bubble">{message.content}</div>
      </div>
    );
  }

  const trace = message.payload?.retrieval;
  const sources = message.payload?.sources ?? [];

  // Deduplicated, because several chunks routinely come off the same page.
  const citations = Array.from(
    new Set(sources.map((source) => `${source.document}|${source.page}`)),
  ).map((key) => {
    const [document, page] = key.split("|");
    return { document, page };
  });

  return (
    <div className="message assistant">
      <div className="bubble">
        {message.content}
        {message.pending ? <span className="caret" /> : null}
      </div>

      {citations.length > 0 ? (
        <div className="sources">
          {citations.map((citation, index) => (
            <span className="source" key={index}>
              <strong>{citation.document}</strong> p.{citation.page}
            </span>
          ))}
        </div>
      ) : null}

      {trace ? <Trace trace={trace} /> : null}
    </div>
  );
}
