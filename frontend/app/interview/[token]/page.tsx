"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useParams } from "next/navigation";
import Avatar, { AvatarHandle } from "./Avatar";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface TranscriptEntry {
  role: "user" | "ai";
  text: string;
  timestamp: number;
}

interface InterviewContext {
  candidate_name: string;
  job_title: string;
  job_description: string;
  keywords: string[];
  mandatory_questions: string[];
  cv_text: string;
  max_interview_minutes: number;
}

// ── Audio helpers (mic encoding) ─────────────────────────

function floatTo16BitPCM(float32Array: Float32Array): Uint8Array {
  const buffer = new ArrayBuffer(float32Array.length * 2);
  const view = new DataView(buffer);
  for (let i = 0; i < float32Array.length; i++) {
    let s = Math.max(-1, Math.min(1, float32Array[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Uint8Array(buffer);
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  return btoa(binary);
}

function downsampleTo24k(float32Buffer: Float32Array, inputSampleRate: number): Float32Array {
  if (inputSampleRate === 24000) return float32Buffer;
  const ratio = inputSampleRate / 24000;
  const newLength = Math.round(float32Buffer.length / ratio);
  const result = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;

  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
    let accum = 0;
    let count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < float32Buffer.length; i++) {
      accum += float32Buffer[i];
      count++;
    }
    result[offsetResult] = count > 0 ? accum / count : 0;
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }

  return result;
}

// ──────────────────────────────────────────────────────────

export default function InterviewPage() {
  const params = useParams();
  const token = params.token as string;

  const [phase, setPhase] = useState<"loading" | "ready" | "live" | "ending" | "done" | "error">("loading");
  const [context, setContext] = useState<InterviewContext | null>(null);
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  const [error, setError] = useState("");
  const [elapsed, setElapsed] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const micSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const avatarRef = useRef<AvatarHandle>(null);
  const aiSpeakingRef = useRef(false);
  const transcriptEndRef = useRef<HTMLDivElement>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number>(0);
  const speakingRef = useRef(false);
  const speechEndTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const transcriptRef = useRef<TranscriptEntry[]>([]);

  // Keep transcriptRef in sync
  useEffect(() => {
    transcriptRef.current = transcript;
  }, [transcript]);

  // Scroll transcript to bottom
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [transcript]);

  // Load interview context
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API}/api/interview-context/${token}`);
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || "Invalid interview link");
        }
        const data = await res.json();
        setContext(data);
        setPhase("ready");
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : "Failed to load interview");
        setPhase("error");
      }
    })();
  }, [token]);

  // ── Playback controls (delegated to Avatar) ──

  const stopPlaybackNow = useCallback(() => {
    avatarRef.current?.stopAudio();
  }, []);

  // ── Start interview ──

  const startInterview = async () => {
    if (!context) return;
    setPhase("live");
    startTimeRef.current = Date.now();
    timerRef.current = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTimeRef.current) / 1000));
    }, 1000);

    try {
      // Build WebSocket URL to backend
      const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const apiHost = new URL(API).host;
      const wsUrl = `${wsProtocol}//${apiHost}/ws/${token}`;

      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log("WebSocket connected to backend");
      };

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);

        if (msg.type === "status") {
          console.log("Status:", msg.message);
          return;
        }

        if (msg.type === "session") {
          console.log("Session ID:", msg.sessionId);
          return;
        }

        if (msg.type === "text") {
          // AI transcript text
          setTranscript(prev => {
            const last = prev[prev.length - 1];
            if (last && last.role === "ai") {
              return [...prev.slice(0, -1), { ...last, text: last.text + " " + msg.text }];
            }
            return [...prev, { role: "ai", text: msg.text, timestamp: Date.now() }];
          });
          return;
        }

        if (msg.type === "audio") {
          avatarRef.current?.feedAudio(msg.data, msg.mimeType);
          return;
        }

        if (msg.type === "turnComplete") {
          // Force new transcript entry for next speaker
          setTranscript(prev => [...prev]);
          return;
        }

        if (msg.type === "interviewComplete") {
          // Ignore premature signals — must be at least 90s into the interview
          const elapsedSec = (Date.now() - startTimeRef.current) / 1000;
          if (elapsedSec < 90) {
            console.warn("Ignoring premature interviewComplete signal (only", Math.round(elapsedSec), "s elapsed)");
            return;
          }
          console.log("Interview complete signal received, reason:", msg.reason);
          // Small delay to let final audio finish playing
          setTimeout(() => {
            endInterview();
          }, 2000);
          return;
        }

        if (msg.type === "sessionDropped") {
          console.warn("Gemini session dropped:", msg.reason);
          stopPlaybackNow();
          setError("The interview session was interrupted by the server. Please refresh and try again.");
          setPhase("error");
          return;
        }

        if (msg.type === "error") {
          console.error("Server error:", msg.error);
        }
      };

      ws.onclose = (event) => {
        console.log("WebSocket closed", event.code, event.reason);
        // If the interview is still live when the WS closes, it's an unexpected drop
        setPhase(prev => {
          if (prev === "live") {
            setError("Connection lost. Please refresh the page to restart the interview.");
            return "error";
          }
          return prev;
        });
      };

      ws.onerror = (event) => {
        console.warn("WebSocket error", event);
      };

      // Initialize Avatar audio (playback context) and mic audio context
      await avatarRef.current?.startStream();
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      audioCtxRef.current = audioCtx;

      // Get microphone
      const micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          channelCount: 1,
        },
      });
      micStreamRef.current = micStream;

      // Set up ScriptProcessor with speech detection (from vocal-part)
      const micSource = audioCtx.createMediaStreamSource(micStream);
      micSourceRef.current = micSource;
      const processor = audioCtx.createScriptProcessor(16384, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (event: AudioProcessingEvent) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

        // While AI is outputting audio, suppress mic entirely to prevent self-interruption via echo
        if (aiSpeakingRef.current) {
          if (speechEndTimerRef.current) {
            clearTimeout(speechEndTimerRef.current);
            speechEndTimerRef.current = null;
          }
          return;
        }

        const input = event.inputBuffer.getChannelData(0);

        // RMS-based speech detection
        let rms = 0;
        for (let i = 0; i < input.length; i++) rms += input[i] * input[i];
        rms = Math.sqrt(rms / input.length);

        const speaking = rms > 0.032;
        if (speaking) {
          // Cancel any pending speechEnd debounce — user is still talking
          if (speechEndTimerRef.current) {
            clearTimeout(speechEndTimerRef.current);
            speechEndTimerRef.current = null;
          }
          if (!speakingRef.current) {
            speakingRef.current = true;
            stopPlaybackNow(); // Interrupt AI audio when user speaks
            wsRef.current.send(JSON.stringify({ type: "speechStart" }));
          }
        } else if (speakingRef.current && !speechEndTimerRef.current) {
          // Debounce: wait 1.5s of silence before committing speechEnd
          // This prevents false triggers during brief mid-sentence pauses
          speechEndTimerRef.current = setTimeout(() => {
            speechEndTimerRef.current = null;
            speakingRef.current = false;
            if (wsRef.current?.readyState === WebSocket.OPEN) {
              wsRef.current.send(JSON.stringify({ type: "speechEnd" }));
            }
          }, 2200);
        }

        // Downsample and send audio
        const downsampled = downsampleTo24k(input, audioCtx.sampleRate);
        const pcmBytes = floatTo16BitPCM(downsampled);
        const b64 = bytesToBase64(pcmBytes);

        wsRef.current.send(JSON.stringify({
          type: "audio",
          mimeType: "audio/pcm;rate=24000",
          data: b64,
        }));
      };

      micSource.connect(processor);
      processor.connect(audioCtx.destination);

    } catch (err: unknown) {
      console.error("Start interview error:", err);
      setError(err instanceof Error ? err.message : "Failed to start interview");
      setPhase("error");
    }
  };

  // ── End interview ──

  const endInterview = async () => {
    setPhase("ending");
    if (timerRef.current) clearInterval(timerRef.current);

    // Stop playback
    stopPlaybackNow();

    // Stop audio processing
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current.onaudioprocess = null;
      processorRef.current = null;
    }
    if (micSourceRef.current) {
      micSourceRef.current.disconnect();
      micSourceRef.current = null;
    }
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach(t => t.stop());
      micStreamRef.current = null;
    }
    if (audioCtxRef.current) {
      try { audioCtxRef.current.close(); } catch { /* ignore */ }
      audioCtxRef.current = null;
    }

    // Close WebSocket (this triggers server-side scoring)
    if (wsRef.current && wsRef.current.readyState <= WebSocket.OPEN) {
      wsRef.current.close();
    }
    wsRef.current = null;

    setPhase("done");
  };

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  if (phase === "loading") {
    return (
      <div className="flex items-center justify-center min-h-[calc(100vh-57px)]">
        <div className="text-slate-500">Loading interview...</div>
      </div>
    );
  }

  if (phase === "error") {
    return (
      <div className="flex items-center justify-center min-h-[calc(100vh-57px)]">
        <div className="bg-red-50 border border-red-200 rounded-xl p-6 max-w-md text-center">
          <h2 className="text-xl font-bold text-red-800 mb-2">Error</h2>
          <p className="text-red-700">{error}</p>
        </div>
      </div>
    );
  }

  if (phase === "done") {
    return (
      <div className="flex items-center justify-center min-h-[calc(100vh-57px)]">
        <div className="bg-green-50 border border-green-200 rounded-xl p-8 max-w-md text-center">
          <div className="text-4xl mb-3">&#10003;</div>
          <h2 className="text-xl font-bold text-green-800 mb-2">Interview Complete!</h2>
          <p className="text-green-700">
            Thank you for your time. Your interview has been recorded and scored.
            You will receive an email with the results.
          </p>
          <p className="text-sm text-green-600 mt-3">Duration: {formatTime(elapsed)}</p>
        </div>
      </div>
    );
  }

  if (phase === "ready") {
    return (
      <div className="flex items-center justify-center min-h-[calc(100vh-57px)] px-4">
        <div className="max-w-lg text-center">
          <h1 className="text-3xl font-bold text-slate-900 mb-2">Voice Interview</h1>
          <p className="text-slate-600 mb-1">Position: <strong>{context?.job_title}</strong></p>
          <p className="text-slate-600 mb-6">Welcome, <strong>{context?.candidate_name}</strong></p>

          <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 mb-6 text-left text-sm text-blue-800">
            <p className="font-semibold mb-2">Before you begin:</p>
            <ul className="list-disc list-inside space-y-1">
              <li>Find a quiet environment</li>
              <li>Allow microphone access when prompted</li>
              <li>Speak naturally - the AI will detect when you start and stop talking</li>
              <li>You can interrupt the interviewer by speaking</li>
              <li>The interview takes about {context?.max_interview_minutes} minutes</li>
            </ul>
          </div>

          <button
            onClick={startInterview}
            className="bg-blue-600 text-white px-10 py-4 rounded-xl font-bold text-lg hover:bg-blue-700 transition-colors"
          >
            Start Interview
          </button>
        </div>
      </div>
    );
  }

  // Live phase
  return (
    <div className="flex flex-col h-[calc(100vh-57px)] bg-slate-900">
      {/* Top bar */}
      <div className="bg-slate-800 px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="font-semibold text-slate-200 text-sm">
            Interview
          </span>
          <span className="text-slate-400 text-sm">{formatTime(elapsed)}</span>
        </div>
        <button
          onClick={endInterview}
          disabled={phase === "ending"}
          className="bg-red-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-red-700 disabled:opacity-50 text-sm"
        >
          {phase === "ending" ? "Ending..." : "End Interview"}
        </button>
      </div>

      {/* Avatar — full remaining height */}
      <div className="flex-1 flex items-center justify-center">
        <Avatar
          ref={avatarRef}
          className="w-full h-full"
          onTalkingChange={(talking) => { aiSpeakingRef.current = talking; }}
        />
      </div>
    </div>
  );
}
