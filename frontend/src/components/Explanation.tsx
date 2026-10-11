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
