"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useParams } from "next/navigation";

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

// ── Audio helpers (from vocal-part) ──────────────────────

function int16ToFloat32(int16: Int16Array): Float32Array {
  const f32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    f32[i] = int16[i] / 32768;
  }
  return f32;
}

function base64ToInt16(base64: string): Int16Array {
  const binary = atob(base64);
  const len = binary.length;
  const buffer = new ArrayBuffer(len);
  const bytes = new Uint8Array(buffer);
  for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
  return new Int16Array(buffer);
}

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

function downsampleTo16k(float32Buffer: Float32Array, inputSampleRate: number): Float32Array {
  if (inputSampleRate === 16000) return float32Buffer;
  const ratio = inputSampleRate / 16000;
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
  const [isSpeaking, setIsSpeaking] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const micSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const playbackTimeRef = useRef(0);
  const activeSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const transcriptEndRef = useRef<HTMLDivElement>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number>(0);
  const speakingRef = useRef(false);
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

  // ── Playback controls (from vocal-part) ──

  const stopPlaybackNow = useCallback(() => {
    for (const src of activeSourcesRef.current) {
      try { src.stop(0); } catch { /* ignore */ }
    }
    activeSourcesRef.current.clear();
    if (audioCtxRef.current) {
      playbackTimeRef.current = audioCtxRef.current.currentTime;
    }
  }, []);

  const playPcmBase64 = useCallback((base64: string, mimeType: string = "audio/pcm;rate=24000") => {
    const audioCtx = audioCtxRef.current;
    if (!audioCtx) return;

    const sampleRateMatch = /rate=(\d+)/.exec(mimeType);
    const sampleRate = sampleRateMatch ? Number(sampleRateMatch[1]) : 24000;

    const pcm16 = base64ToInt16(base64);
    const audioData = int16ToFloat32(pcm16);

    const buffer = audioCtx.createBuffer(1, audioData.length, sampleRate);
    buffer.copyToChannel(new Float32Array(audioData.buffer as ArrayBuffer), 0);

    const source = audioCtx.createBufferSource();
    source.buffer = buffer;
    source.connect(audioCtx.destination);

    if (playbackTimeRef.current < audioCtx.currentTime) {
      playbackTimeRef.current = audioCtx.currentTime;
    }
    source.start(playbackTimeRef.current);
    playbackTimeRef.current += buffer.duration;

    activeSourcesRef.current.add(source);
    source.onended = () => activeSourcesRef.current.delete(source);
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
          playPcmBase64(msg.data, msg.mimeType);
          return;
        }

        if (msg.type === "turnComplete") {
          // Force new transcript entry for next speaker
          setTranscript(prev => [...prev]);
          return;
        }

        if (msg.type === "error") {
          console.error("Server error:", msg.error);
        }
      };

      ws.onclose = () => {
        console.log("WebSocket closed");
      };

      ws.onerror = () => {
        console.error("WebSocket error");
      };

      // Set up audio context
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      audioCtxRef.current = audioCtx;
      playbackTimeRef.current = audioCtx.currentTime;

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

        const input = event.inputBuffer.getChannelData(0);

        // RMS-based speech detection
        let rms = 0;
        for (let i = 0; i < input.length; i++) rms += input[i] * input[i];
        rms = Math.sqrt(rms / input.length);

        const speaking = rms > 0.045;
        if (speaking && !speakingRef.current) {
          speakingRef.current = true;
          setIsSpeaking(true);
          stopPlaybackNow(); // Interrupt AI audio when user speaks
          wsRef.current.send(JSON.stringify({ type: "speechStart" }));
        } else if (!speaking && speakingRef.current) {
          speakingRef.current = false;
          setIsSpeaking(false);
          wsRef.current.send(JSON.stringify({ type: "speechEnd" }));
        }

        // Downsample and send audio
        const downsampled = downsampleTo16k(input, audioCtx.sampleRate);
        const pcmBytes = floatTo16BitPCM(downsampled);
        const b64 = bytesToBase64(pcmBytes);

        wsRef.current.send(JSON.stringify({
          type: "audio",
          mimeType: "audio/pcm;rate=16000",
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
    <div className="flex flex-col h-[calc(100vh-57px)]">
      {/* Top bar */}
      <div className="bg-white border-b border-slate-200 px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className={`w-3 h-3 rounded-full ${isSpeaking ? "bg-green-500 animate-pulse" : "bg-red-500 animate-pulse"}`} />
          <span className="font-semibold text-slate-900">
            {isSpeaking ? "You are speaking..." : "Interview in Progress"}
          </span>
          <span className="text-slate-500 text-sm">{formatTime(elapsed)}</span>
        </div>
        <button
          onClick={endInterview}
          disabled={phase === "ending"}
          className="bg-red-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-red-700 disabled:opacity-50 text-sm"
        >
          {phase === "ending" ? "Ending..." : "End Interview"}
        </button>
      </div>

      {/* Transcript */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-2xl mx-auto space-y-4">
          {transcript.length === 0 && (
            <p className="text-center text-slate-400 mt-8">
              The AI interviewer will start speaking shortly...
            </p>
          )}
          {transcript.map((entry, i) => (
            <div
              key={i}
              className={`flex ${entry.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                  entry.role === "user"
                    ? "bg-blue-600 text-white"
                    : "bg-white border border-slate-200 text-slate-800"
                }`}
              >
                <p className="text-xs font-medium mb-1 opacity-70">
                  {entry.role === "user" ? "You" : "Sarah (AI)"}
                </p>
                <p className="text-sm">{entry.text}</p>
              </div>
            </div>
          ))}
          <div ref={transcriptEndRef} />
        </div>
      </div>
    </div>
  );
}
