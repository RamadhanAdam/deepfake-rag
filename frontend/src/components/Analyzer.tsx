import { useCallback, useEffect, useRef, useState } from "react";
import { analyse, wake, type Verdict } from "../api";
import { Explanation } from "./Explanation";

type Stage = "idle" | "waking" | "analysing" | "done" | "error";

const ACCEPTED = ["image/jpeg", "image/png"];
// Some browsers and systems leave file.type empty, so the extension counts too.
const isImage = (f: File) => ACCEPTED.includes(f.type) || /\.(jpe?g|png)$/i.test(f.name);

function formatBytes(n: number) {
  return n < 1024 * 1024 ? `${Math.round(n / 1024)} KB` : `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export function Analyzer() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  const [stage, setStage] = useState<Stage>("idle");
  const [result, setResult] = useState<Verdict | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const input = useRef<HTMLInputElement>(null);

  const take = useCallback((f: File | undefined | null) => {
    if (!f) return;
    if (!isImage(f)) {
      setError("That file isn't a JPG or PNG.");
      return;
    }
    setFile(f);
    setPreview((old) => {
      if (old) URL.revokeObjectURL(old);
      return URL.createObjectURL(f);
    });
    setSize(null);
    setResult(null);
    setError(null);
    setStage("idle");
  }, []);

  // Pasting an image anywhere on the page loads it.
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const item = [...(e.clipboardData?.files ?? [])].find(isImage);
      if (item) take(item);
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [take]);

  // A running clock while the server works, so a cold start doesn't look like a hang.
  useEffect(() => {
    if (stage !== "waking" && stage !== "analysing") return;
    const start = Date.now();
    setElapsed(0);
    const t = setInterval(() => setElapsed(Math.floor((Date.now() - start) / 1000)), 500);
    return () => clearInterval(t);
  }, [stage === "waking" || stage === "analysing"]); // eslint-disable-line react-hooks/exhaustive-deps

  const run = async () => {
    if (!file) return;
    setError(null);
    setResult(null);
    setStage("waking");
    await wake();
    setStage("analysing");
    try {
      setResult(await analyse(file));
      setStage("done");
    } catch (e) {
      setError(e instanceof Error && e.message !== "Failed to fetch" ? e.message : "Couldn't reach the server. It may still be waking up; try again in a minute.");
      setStage("error");
    }
  };

  const reset = () => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setStage("idle");
    if (input.current) input.current.value = "";
  };

  const busy = stage === "waking" || stage === "analysing";

  return (
    <section className="mx-auto grid max-w-6xl gap-8 px-5 pb-20 pt-8 sm:px-8 lg:grid-cols-[1.15fr_1fr] lg:grid-rows-[auto_1fr] lg:gap-x-14 lg:gap-y-0 lg:pt-16">
      {/* intro */}
      <div className="lg:col-start-2 lg:row-start-1 lg:pt-2">
        <h1 className="font-serif text-5xl leading-[1.02] tracking-tight sm:text-6xl">
          Is this face <em>real?</em>
        </h1>
        <p className="mt-5 max-w-md text-lg leading-relaxed text-muted">
          Upload a portrait and get a verdict, with an explanation drawn from deepfake-forensics research.
        </p>
        <p className="mt-3 max-w-md text-sm leading-relaxed text-faint">
          Works best on one front-facing face. It was trained on GAN-generated faces, so images from newer generators can fool it.
        </p>
      </div>

      {/* the specimen well */}
      <div className="lg:col-start-1 lg:row-span-2 lg:row-start-1">
        <div
          role="button"
          tabIndex={0}
          aria-label={preview ? "Choose a different photo" : "Choose a photo to analyse"}
          onClick={() => !busy && input.current?.click()}
          onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && !busy && input.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            take(e.dataTransfer.files[0]);
          }}
          className={`grid-paper relative flex aspect-[4/5] w-full cursor-pointer items-center justify-center overflow-hidden rounded-sm border bg-sheet outline-none transition-colors focus-visible:ring-2 focus-visible:ring-ink sm:aspect-[5/5] ${
            dragging ? "border-ink" : "border-rule hover:border-faint"
          }`}
        >
          {/* registration marks */}
          {["left-3 top-3 border-l border-t", "right-3 top-3 border-r border-t", "bottom-3 left-3 border-b border-l", "bottom-3 right-3 border-b border-r"].map((c) => (
            <span key={c} className={`pointer-events-none absolute z-10 h-5 w-5 border-ink ${c}`} />
          ))}

          {preview ? (
            <>
              <img
                src={preview}
                alt="The photo being analysed"
                onLoad={(e) => setSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
                className="h-full w-full object-contain p-8"
              />
              {busy && (
                <div className="pointer-events-none absolute inset-0 overflow-hidden">
                  <div className="sweep h-24 w-full bg-gradient-to-b from-transparent via-ink/15 to-transparent" />
                </div>
              )}
            </>
          ) : (
            <div className="px-8 text-center">
              <p className="font-serif text-4xl leading-tight sm:text-5xl">Drop a face photo here</p>
              <p className="mt-4 text-muted">or click to choose one. Pasting works too.</p>
              <p className="mt-6 font-mono text-xs uppercase tracking-wider text-faint">JPG or PNG · one face, front-on</p>
            </div>
          )}
          <input ref={input} type="file" accept=".jpg,.jpeg,.png" className="hidden" onChange={(e) => take(e.target.files?.[0])} />
        </div>

        {file && (
          <div className="mt-3 flex flex-wrap items-center justify-between gap-x-6 gap-y-1 font-mono text-xs text-faint">
            <span className="truncate">{file.name}</span>
            <span>
              {size ? `${size.w} × ${size.h} px · ` : ""}
              {formatBytes(file.size)}
            </span>
          </div>
        )}
      </div>

      {/* controls and the report */}
      <div className="lg:col-start-2 lg:row-start-2">
        <div className="flex flex-wrap gap-3 lg:mt-8">
          <button
            type="button"
            onClick={file ? run : () => input.current?.click()}
            disabled={busy}
            className="rounded-sm bg-ink px-6 py-3 font-medium text-paper transition hover:opacity-85 disabled:cursor-wait disabled:opacity-60"
          >
            {busy ? "Working…" : file ? (result ? "Analyse again" : "Analyse photo") : "Choose a photo"}
          </button>
          {file && !busy && (
            <button type="button" onClick={reset} className="rounded-sm border border-rule px-6 py-3 font-medium transition hover:border-ink">
              Clear
            </button>
          )}
        </div>

        {busy && (
          <ol className="mt-8 space-y-3 border-l border-rule pl-5 font-mono text-sm">
            <Step state={stage === "waking" ? "now" : "done"}>Waking the server{stage === "waking" && elapsed > 4 ? " (it sleeps when idle; up to a minute)" : ""}</Step>
            <Step state={stage === "analysing" ? "now" : "next"}>Classifying, then retrieving from the papers</Step>
            <li className="text-faint">{elapsed}s</li>
          </ol>
        )}

        {error && <p className="mt-6 border-l-2 border-fake pl-4 text-fake">{error}</p>}

        {result && <Report result={result} />}
      </div>
    </section>
  );
}

function Step({ state, children }: { state: "done" | "now" | "next"; children: React.ReactNode }) {
  return (
    <li className={`flex items-baseline gap-3 ${state === "next" ? "text-faint" : ""}`}>
      <span className={`inline-block h-2 w-2 shrink-0 rounded-full ${state === "done" ? "bg-ink" : state === "now" ? "animate-pulse bg-ink" : "border border-faint"}`} />
      {children}
    </li>
  );
}

function Report({ result }: { result: Verdict }) {
  const fake = result.label === "fake";
  const tone = fake ? "text-fake" : "text-real";
  // Position on a single scale: 0 = certainly real, 100 = certainly fake.
  const towardFake = fake ? result.confidence : 100 - result.confidence;
  return (
    <div className="mt-10 border-t border-ink pt-6">
      <div className="flex items-end justify-between gap-6">
        <div>
          <p className="font-mono text-xs uppercase tracking-wider text-faint">Verdict</p>
          <p className={`mt-1 font-serif text-5xl ${tone}`}>{fake ? "Likely fake" : "Likely real"}</p>
        </div>
        <p className="font-mono text-3xl tabular-nums">{result.confidence.toFixed(1)}%</p>
      </div>

      <div className="mt-6" aria-hidden="true">
        <div className="relative h-8">
          <div className="absolute inset-x-0 top-4 h-px bg-rule" />
          {Array.from({ length: 11 }, (_, i) => (
            <span key={i} className="absolute top-2 h-4 w-px bg-rule" style={{ left: `${i * 10}%` }} />
          ))}
          <span className={`absolute top-0 h-8 w-0.5 ${fake ? "bg-fake" : "bg-real"}`} style={{ left: `calc(${towardFake}% - 1px)` }} />
        </div>
        <div className="mt-1 flex justify-between font-mono text-xs text-faint">
          <span>real</span>
          <span>fake</span>
        </div>
      </div>

      <div className="mt-8">
        <p className="font-mono text-xs uppercase tracking-wider text-faint">Why</p>
        <Explanation text={result.explanation} />
        <p className="mt-6 text-sm text-faint">The explanation comes from the research for this verdict; it doesn't point at parts of your photo.</p>
      </div>
    </div>
  );
}
