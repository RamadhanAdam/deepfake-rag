import { Analyzer } from "./components/Analyzer";
import { Footer, Header } from "./components/Layout";

export default function App() {
  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <main className="flex-1">
        <Analyzer />
      </main>
      <Footer />
    </div>
  );
}
