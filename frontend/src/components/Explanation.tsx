// The explanation comes back from the language model as light Markdown: paragraphs, **bold**, and numbered
// or bulleted lists. Render just those, as React elements (no HTML injection).

function inline(text: string) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith("**") && part.endsWith("**") ? (
      <strong key={i} className="font-semibold text-ink">
        {part.slice(2, -2)}
      </strong>
    ) : (
      part
    ),
  );
}

export function Explanation({ text }: { text: string }) {
  const blocks = text.trim().split(/\n\s*\n/);
  return (
    <div className="mt-3 space-y-4 leading-relaxed text-muted">
      {blocks.map((block, k) => {
        const lines = block.split("\n").map((l) => l.trim()).filter(Boolean);
        // A Markdown table: | a | b | rows, with a |---|---| separator under the header.
        if (lines.length >= 2 && lines.every((l) => l.startsWith("|"))) {
          const cells = (l: string) => l.replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
          const rows = lines.filter((l) => !/^\|[\s:|-]+\|?$/.test(l)).map(cells);
          const [head, ...body] = rows;
          return (
            <div key={k} className="overflow-x-auto">
              <table className="w-full border-collapse text-left text-sm">
                <thead>
                  <tr className="border-b border-ink">
                    {head.map((c, i) => (
                      <th key={i} className="py-2 pr-4 font-medium text-ink">
                        {inline(c)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {body.map((r, i) => (
                    <tr key={i} className="border-b border-rule align-top">
                      {r.map((c, j) => (
                        <td key={j} className="py-2 pr-4">
                          {inline(c.replace(/<br\s*\/?>/gi, " "))}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        }
        const numbered = lines.every((l) => /^\d+[.)]\s/.test(l));
        const bulleted = lines.every((l) => /^[-*•]\s/.test(l));
        if (numbered || bulleted) {
          const List = numbered ? "ol" : "ul";
          return (
            <List key={k} className={`space-y-2 pl-5 ${numbered ? "list-decimal" : "list-disc"} marker:text-faint`}>
              {lines.map((l, i) => (
                <li key={i}>{inline(l.replace(/^(\d+[.)]|[-*•])\s+/, ""))}</li>
              ))}
            </List>
          );
        }
        if (lines.length === 1 && /^\*\*[^*]+\*\*:?$/.test(lines[0])) {
          return (
            <p key={k} className="font-medium text-ink">
              {lines[0].replace(/\*\*/g, "")}
            </p>
          );
        }
        if (/^#{1,4}\s/.test(lines[0] ?? "")) {
          return (
            <p key={k} className="font-medium text-ink">
              {inline(lines.join(" ").replace(/^#{1,4}\s+/, ""))}
            </p>
          );
        }
        return <p key={k}>{inline(lines.join(" "))}</p>;
      })}
    </div>
  );
}
