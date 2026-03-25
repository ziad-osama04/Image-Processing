/**
 * App: Root application component.
 *
 * Phase 1: Demo page with multiple ViewportComponent instances
 * to verify independence. Creates session on mount.
 */

import { useCallback, useEffect, useState } from 'react';
import { ViewportComponent } from './components/ViewportComponent';
import { createSession } from './services/imageApi';
import './App.css';

function App() {
  const [sessionId, setSessionId] = useState<string>('');
  const [sessionError, setSessionError] = useState<string | null>(null);

  // Create session on mount
  useEffect(() => {
    createSession()
      .then((res) => setSessionId(res.session_id))
      .catch((err) => setSessionError(err.message));
  }, []);

  const handleImageLoaded = useCallback((slot: number, filename: string) => {
    console.log(`Image loaded in slot ${slot}: ${filename}`);
  }, []);

  if (sessionError) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-red-400 bg-red-900/30 rounded-xl p-6 max-w-md text-center">
          <h2 className="text-lg font-semibold mb-2">Connection Error</h2>
          <p className="text-sm">
            Could not connect to backend. Make sure the FastAPI server is
            running on port 8000.
          </p>
          <p className="text-xs mt-2 text-red-300">{sessionError}</p>
        </div>
      </div>
    );
  }

  if (!sessionId) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="flex items-center gap-3 text-[var(--text-secondary)]">
          <div className="w-5 h-5 border-2 border-[var(--accent-blue)] border-t-transparent rounded-full animate-spin" />
          <span>Connecting to server...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen p-4">
      {/* Header */}
      <header className="mb-6 text-center">
        <h1 className="text-2xl font-bold bg-gradient-to-r from-[var(--accent-blue)] to-[var(--accent-purple)] bg-clip-text text-transparent">
          FT Mixer & Emphasizer
        </h1>
        <p className="text-sm text-[var(--text-secondary)] mt-1">
          Fourier Transform Component Mixer & Properties Emphasizer
        </p>
      </header>

      {/* Viewport Grid — 4 inputs */}
      <section className="mb-6">
        <h2 className="text-sm font-semibold text-[var(--text-secondary)] mb-3 uppercase tracking-wider">
          Input Viewports
        </h2>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {[0, 1, 2, 3].map((slot) => (
            <ViewportComponent
              key={slot}
              sessionId={sessionId}
              slot={slot}
              label={`Input ${slot + 1}`}
              allowUpload={true}
              onImageLoaded={handleImageLoaded}
            />
          ))}
        </div>
      </section>

      {/* Output Viewports — 2 outputs */}
      <section>
        <h2 className="text-sm font-semibold text-[var(--text-secondary)] mb-3 uppercase tracking-wider">
          Output Viewports
        </h2>
        <div className="grid grid-cols-2 gap-4 max-w-2xl mx-auto">
          {[4, 5].map((slot) => (
            <ViewportComponent
              key={slot}
              sessionId={sessionId}
              slot={slot}
              label={`Output ${slot - 3}`}
              allowUpload={false}
            />
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="mt-8 text-center text-xs text-[var(--text-secondary)]">
        Session: <code className="text-[var(--accent-blue)]">{sessionId.slice(0, 8)}...</code>
      </footer>
    </div>
  );
}

export default App;
