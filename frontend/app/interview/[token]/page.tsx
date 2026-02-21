"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useParams } from "next/navigation";
import { GoogleGenAI, Modality } from "@google/genai";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const LIVE_MODEL = process.env.NEXT_PUBLIC_LIVE_MODEL || "gemini-2.5-flash-native-audio-preview-12-2025";

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

// AudioWorklet processor code as a blob URL
const WORKLET_CODE = `
class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = new Float32Array(0);
  }
  process(inputs) {
    const input = inputs[0];
    if (input.length > 0) {
      const channelData = input[0];
      // Convert float32 to int16 PCM
      const int16 = new Int16Array(channelData.length);
      for (let i = 0; i < channelData.length; i++) {
        const s = Math.max(-1, Math.min(1, channelData[i]));
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      this.port.postMessage(int16.buffer, [int16.buffer]);
    }
    return true;
  }
}
registerProcessor('pcm-processor', PCMProcessor);
`;

export default function InterviewPage() {
  const params = useParams();
  const token = params.token as string;

  const [phase, setPhase] = useState<"loading" | "ready" | "live" | "ending" | "done" | "error">("loading");
  const [context, setContext] = useState<InterviewContext | null>(null);
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  const [error, setError] = useState("");
  const [elapsed, setElapsed] = useState(0);

  const sessionRef = useRef<any>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const playbackQueueRef = useRef<Int16Array[]>([]);
  const isPlayingRef = useRef(false);
  const transcriptEndRef = useRef<HTMLDivElement>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number>(0);

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

  // Audio playback (24kHz PCM)
  const playAudioChunk = useCallback((pcmData: Int16Array) => {
    if (!audioContextRef.current) return;
    playbackQueueRef.current.push(pcmData);
    if (!isPlayingRef.current) {
      drainPlaybackQueue();
    }
  }, []);

  const drainPlaybackQueue = useCallback(() => {
    if (!audioContextRef.current || playbackQueueRef.current.length === 0) {
      isPlayingRef.current = false;
      return;
    }
    isPlayingRef.current = true;
    const pcm = playbackQueueRef.current.shift()!;

    const float32 = new Float32Array(pcm.length);
    for (let i = 0; i < pcm.length; i++) {
      float32[i] = pcm[i] / 32768;
    }

    const buffer = audioContextRef.current.createBuffer(1, float32.length, 24000);
    buffer.getChannelData(0).set(float32);

    const source = audioContextRef.current.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContextRef.current.destination);
    source.onended = () => drainPlaybackQueue();
    source.start();
  }, []);

  const clearPlayback = useCallback(() => {
    playbackQueueRef.current = [];
    isPlayingRef.current = false;
  }, []);

  // Build system instruction
  const buildSystemInstruction = (ctx: InterviewContext) => {
    return `You are an experienced HR recruiter named Sarah conducting a short phone screen for the position of ${ctx.job_title}.

JOB DESCRIPTION:
${ctx.job_description}

MANDATORY QUESTIONS (you MUST ask all of these, in any order):
${ctx.mandatory_questions.map((q, i) => `${i + 1}. ${q}`).join("\n")}

KEYWORDS TO PROBE FOR:
${ctx.keywords.join(", ")}

CANDIDATE CONTEXT:
Name: ${ctx.candidate_name}
CV Summary (truncated):
${ctx.cv_text}

CONVERSATION RULES:
- Greet the candidate warmly by name and introduce yourself as Sarah from HappyHR.
- Ask ONE question at a time.
- Keep your responses concise and natural - this should feel like a real phone conversation.
- Listen carefully and ask relevant follow-up questions when appropriate.
- Probe for evidence of the listed keywords through natural conversation.
- After all mandatory questions are covered, ask if the candidate has any questions about the role.
- End the interview gracefully when: (a) all mandatory questions are done AND (b) the candidate has no more questions, OR after ${ctx.max_interview_minutes} minutes.
- When ending, thank the candidate and let them know they'll hear back soon.

SAFETY/FAIRNESS RULES:
- Do NOT infer or ask about protected attributes (race, religion, health, disability, age, gender, sexual orientation, marital status, etc.).
- Evaluate ONLY job-relevant skills and experience.
- Be equally warm and professional with all candidates.`;
  };

  // Start interview
  const startInterview = async () => {
    if (!context) return;
    setPhase("live");
    startTimeRef.current = Date.now();
    timerRef.current = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTimeRef.current) / 1000));
    }, 1000);

    try {
      // Get ephemeral token
      const tokenRes = await fetch(`${API}/api/live-token`, { method: "POST" });
      const tokenData = await tokenRes.json();
      const apiKey = tokenData.token;

      // Set up audio context
      audioContextRef.current = new AudioContext({ sampleRate: 24000 });

      // Get mic stream
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000,
        },
      });
      streamRef.current = stream;

      // Connect to Gemini Live
      const ai = new GoogleGenAI({ apiKey });
      const systemInstruction = buildSystemInstruction(context);

      const session = await ai.live.connect({
        model: LIVE_MODEL,
        config: {
          responseModalities: [Modality.AUDIO],
          inputAudioTranscription: {},
          outputAudioTranscription: {},
          systemInstruction: systemInstruction,
        },
        callbacks: {
          onopen: () => {
            console.log("Live session opened");
          },
          onmessage: (msg: any) => {
            // Handle audio output
            if (msg.serverContent?.modelTurn?.parts) {
              for (const part of msg.serverContent.modelTurn.parts) {
                if (part.inlineData?.mimeType?.startsWith("audio/pcm")) {
                  const raw = atob(part.inlineData.data);
                  const bytes = new Uint8Array(raw.length);
                  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
                  const int16 = new Int16Array(bytes.buffer);
                  playAudioChunk(int16);
                }
              }
            }

            // Handle interruption
            if (msg.serverContent?.interrupted) {
              clearPlayback();
            }

            // Handle input transcription
            if (msg.serverContent?.inputTranscription?.text) {
              const text = msg.serverContent.inputTranscription.text;
              setTranscript(prev => {
                const last = prev[prev.length - 1];
                if (last && last.role === "user") {
                  return [...prev.slice(0, -1), { ...last, text: last.text + text }];
                }
                return [...prev, { role: "user", text, timestamp: Date.now() }];
              });
            }

            // Handle output transcription
            if (msg.serverContent?.outputTranscription?.text) {
              const text = msg.serverContent.outputTranscription.text;
              setTranscript(prev => {
                const last = prev[prev.length - 1];
                if (last && last.role === "ai") {
                  return [...prev.slice(0, -1), { ...last, text: last.text + text }];
                }
                return [...prev, { role: "ai", text, timestamp: Date.now() }];
              });
            }

            // Handle turn complete - start new entry for next speaker
            if (msg.serverContent?.turnComplete) {
              // Turn is done, next text will create a new entry
            }
          },
          onerror: (err: any) => {
            console.error("Live session error:", err);
          },
          onclose: () => {
            console.log("Live session closed");
          },
        },
      });

      sessionRef.current = session;

      // Set up audio worklet for mic capture
      const workletBlob = new Blob([WORKLET_CODE], { type: "application/javascript" });
      const workletUrl = URL.createObjectURL(workletBlob);
      const micContext = new AudioContext({ sampleRate: 16000 });
      await micContext.audioWorklet.addModule(workletUrl);
      URL.revokeObjectURL(workletUrl);

      const source = micContext.createMediaStreamSource(stream);
      const workletNode = new AudioWorkletNode(micContext, "pcm-processor");
      workletNodeRef.current = workletNode;

      workletNode.port.onmessage = (event) => {
        if (sessionRef.current) {
          const b64 = arrayBufferToBase64(event.data);
          sessionRef.current.sendRealtimeInput({
            data: b64,
            mimeType: "audio/pcm;rate=16000",
          });
        }
      };

      source.connect(workletNode);
      workletNode.connect(micContext.destination); // needed for worklet to process

    } catch (err: unknown) {
      console.error("Start interview error:", err);
      setError(err instanceof Error ? err.message : "Failed to start interview");
      setPhase("error");
    }
  };

  // End interview
  const endInterview = async () => {
    setPhase("ending");
    if (timerRef.current) clearInterval(timerRef.current);

    // Close live session
    try {
      if (sessionRef.current) {
        sessionRef.current.close();
        sessionRef.current = null;
      }
    } catch {}

    // Stop mic
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
    }

    // Close audio context
    if (audioContextRef.current) {
      try { audioContextRef.current.close(); } catch {}
    }

    // Build transcript string
    const transcriptText = transcript
      .map(e => `${e.role === "user" ? "Candidate" : "Interviewer"}: ${e.text}`)
      .join("\n\n");

    // Send for scoring
    try {
      const res = await fetch(`${API}/api/interview-complete/${token}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transcript: transcriptText }),
      });
      if (res.ok) {
        const data = await res.json();
        setPhase("done");
      } else {
        setPhase("done");
      }
    } catch {
      setPhase("done");
    }
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
              <li>Speak naturally - no push-to-talk needed</li>
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
          <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
          <span className="font-semibold text-slate-900">Interview in Progress</span>
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

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}
