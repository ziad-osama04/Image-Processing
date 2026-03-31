import React, { createContext, useCallback, useEffect, useState } from 'react';
import { createSession as apiCreateSession } from '../services/imageApi';

interface SessionContextType {
  sessionId: string | null;
  loading: boolean;
  error: string | null;
  createSession: () => Promise<void>;
  emphasizerImage: any | null;
  setEmphasizerImage: (img: any | null) => void;
  emphasizerPixels: number[] | null;
  setEmphasizerPixels: (arr: number[] | null) => void;
}

export const SessionContext = createContext<SessionContextType | undefined>(undefined);

export const SessionProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [emphasizerImage, setEmphasizerImage] = useState<any | null>(null);
  const [emphasizerPixels, setEmphasizerPixels] = useState<number[] | null>(null);

  const createSession = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      // This now auto-falls back to mock if backend is offline
      const res = await apiCreateSession();
      setSessionId(res.session_id);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to create session');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    createSession();
  }, [createSession]);

  return (
    <SessionContext.Provider value={{ 
      sessionId, loading, error, createSession,
      emphasizerImage, setEmphasizerImage,
      emphasizerPixels, setEmphasizerPixels
    }}>
      {children}
    </SessionContext.Provider>
  );
};
