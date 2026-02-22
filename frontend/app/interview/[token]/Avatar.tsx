"use client";

import {
  useRef,
  useImperativeHandle,
  forwardRef,
  useState,
  useCallback,
  useEffect,
} from "react";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

export interface AvatarHandle {
  /** Feed a base64-encoded PCM audio chunk for playback */
  feedAudio: (base64: string, mimeType?: string) => void;
  /** Stop all audio playback immediately (e.g. user interruption) */
  stopAudio: () => void;
  /** Initialize the audio context (call once on user gesture) */
  startStream: () => Promise<void>;
  /** Always true — no heavy loading needed anymore */
  isReady: () => boolean;
}

interface AvatarProps {
  className?: string;
  onTalkingChange?: (talking: boolean) => void;
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

const PCM_SAMPLE_RATE = 24000;

function base64ToFloat32(base64: string): Float32Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const int16 = new Int16Array(bytes.buffer);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    float32[i] = int16[i] / 32768.0;
  }
  return float32;
}

/* ------------------------------------------------------------------ */
/*  CSS fallback avatar (shown when video files are not present)      */
/* ------------------------------------------------------------------ */

function FallbackAvatar({ isTalking }: { isTalking: boolean }) {
  return (
    <>
      <style>{`
        @keyframes sarah-pulse {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.04); }
        }
        .sarah-talking {
          animation: sarah-pulse 0.5s ease-in-out infinite;
        }
      `}</style>
      <div className="w-full h-full flex items-center justify-center">
        <div className="w-[400px] h-[400px] rounded-full overflow-hidden">
          <img
            src="/avatars/sarah.png"
            alt="Sarah AI Interviewer"
            className={`w-full h-full object-cover transition-transform duration-300 ${isTalking ? "sarah-talking" : ""}`}
          />
        </div>
      </div>
    </>
  );
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

const Avatar = forwardRef<AvatarHandle, AvatarProps>(
  function Avatar({ className, onTalkingChange }, ref) {
    const idleVideoRef = useRef<HTMLVideoElement>(null);
    const talkingVideoRef = useRef<HTMLVideoElement>(null);
    const audioCtxRef = useRef<AudioContext | null>(null);
    const nextStartTimeRef = useRef<number>(0);
    const activeSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());

    const [isTalking, setIsTalking] = useState(false);
    // 'checking' → 'video' | 'fallback'
    const [videoMode, setVideoMode] = useState<"checking" | "video" | "fallback">("checking");

    /* ---- Notify parent of talking state changes ---- */
    useEffect(() => {
      onTalkingChange?.(isTalking);
    }, [isTalking, onTalkingChange]);

    /* ---- Detect if video files exist ---- */
    useEffect(() => {
      let resolved = false;

      const resolve = (mode: "video" | "fallback") => {
        if (!resolved) {
          resolved = true;
          setVideoMode(mode);
        }
      };

      fetch("/avatars/sarah-idle.mp4", { method: "HEAD" })
        .then((res) => resolve(res.ok ? "video" : "fallback"))
        .catch(() => resolve("fallback"));

      // Safety timeout: fall back to CSS avatar after 2 s if fetch hangs
      setTimeout(() => resolve("fallback"), 2000);
    }, []);

    /* ---- Sync video playback with talking state ---- */
    useEffect(() => {
      if (videoMode !== "video") return;

      if (isTalking) {
        idleVideoRef.current?.pause();
        talkingVideoRef.current?.play().catch(() => {});
      } else {
        talkingVideoRef.current?.pause();
        idleVideoRef.current?.play().catch(() => {});
      }
    }, [isTalking, videoMode]);

    /* ---- Imperative API ---- */

    const startStream = useCallback(async () => {
      if (!audioCtxRef.current) {
        audioCtxRef.current = new AudioContext();
      }
      if (audioCtxRef.current.state === "suspended") {
        await audioCtxRef.current.resume();
      }
      nextStartTimeRef.current = 0;
    }, []);

    const feedAudio = useCallback((base64: string) => {
      // Lazily create the AudioContext on first audio chunk (Avatar may not have
      // been rendered yet when startStream() was called from startInterview).
      if (!audioCtxRef.current) {
        audioCtxRef.current = new AudioContext();
        nextStartTimeRef.current = 0;
      }
      const ctx = audioCtxRef.current;
      // Resume if the browser suspended it due to autoplay policy.
      if (ctx.state === "suspended") {
        ctx.resume().catch(() => {});
      }

      const float32 = base64ToFloat32(base64);

      const buffer = ctx.createBuffer(1, float32.length, PCM_SAMPLE_RATE);
      buffer.getChannelData(0).set(float32);

      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);

      const now = ctx.currentTime;
      const startTime = nextStartTimeRef.current > now ? nextStartTimeRef.current : now;
      source.start(startTime);
      nextStartTimeRef.current = startTime + buffer.duration;

      activeSourcesRef.current.add(source);
      setIsTalking(true);

      source.onended = () => {
        activeSourcesRef.current.delete(source);
        if (activeSourcesRef.current.size === 0) {
          setIsTalking(false);
          nextStartTimeRef.current = 0;
        }
      };
    }, []);

    const stopAudio = useCallback(() => {
      const sources = [...activeSourcesRef.current];
      activeSourcesRef.current.clear();
      sources.forEach((source) => {
        source.onended = null;
        try {
          source.stop();
        } catch {
          /* already ended */
        }
      });
      nextStartTimeRef.current = 0;
      setIsTalking(false);
    }, []);

    const isReady = useCallback(() => true, []);

    useImperativeHandle(
      ref,
      () => ({ feedAudio, stopAudio, startStream, isReady }),
      [feedAudio, stopAudio, startStream, isReady]
    );

    /* ---- Render ---- */
    return (
      <div className={`relative ${className ?? ""}`} style={{ minHeight: 300 }}>
        {/* Video mode */}
        {videoMode === "video" && (
          <div className="absolute inset-0">
            <video
              ref={idleVideoRef}
              src="/avatars/sarah-idle.mp4"
              loop
              muted
              playsInline
              autoPlay
              className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-200 ${
                isTalking ? "opacity-0" : "opacity-100"
              }`}
            />
            <video
              ref={talkingVideoRef}
              src="/avatars/sarah-talking.mp4"
              loop
              muted
              playsInline
              className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-200 ${
                isTalking ? "opacity-100" : "opacity-0"
              }`}
            />
          </div>
        )}

        {/* CSS fallback */}
        {videoMode === "fallback" && <FallbackAvatar isTalking={isTalking} />}

        {/* Loading placeholder */}
        {videoMode === "checking" && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-100">
            <div className="text-slate-400 text-sm">Loading...</div>
          </div>
        )}
      </div>
    );
  }
);

export default Avatar;
