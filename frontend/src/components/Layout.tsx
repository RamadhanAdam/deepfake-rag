const REPO = "https://github.com/RamadhanAdam/deepfake-rag";

export function Header() {
  return (
    <header className="mx-auto flex w-full max-w-6xl items-center justify-between gap-6 px-5 py-5 sm:px-8">
      <a href="/" className="flex items-center gap-2.5 font-mono text-sm font-medium">
        <svg viewBox="0 0 32 32" className="h-6 w-6" aria-hidden="true">
          <path d="M6 6h6M6 6v6M26 6h-6M26 6v6M6 26h6M6 26v-6M26 26h-6M26 26v-6" stroke="currentColor" strokeWidth="2" fill="none" />
          <circle cx="16" cy="16" r="3.5" className="fill-fake" />
        </svg>
        deepfake-rag
      </a>
      <a href={REPO} target="_blank" rel="noreferrer" className="text-sm text-muted hover:text-ink">
        Code ↗
      </a>
    </header>
  );
}

export function Footer() {
  return (
    <footer className="mx-auto flex w-full max-w-6xl flex-wrap items-center justify-between gap-4 px-5 py-8 text-sm text-faint sm:px-8">
      <p>
        Built by{" "}
        <a href="https://ramadhanadam.github.io" className="text-ink underline decoration-rule underline-offset-4 hover:decoration-ink">
          Ramadhan Zome
        </a>
      </p>
      <a href={REPO} target="_blank" rel="noreferrer" className="hover:text-ink">
        How it works, on GitHub
      </a>
    </footer>
  );
}
