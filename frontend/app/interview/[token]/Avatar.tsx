"use client";

import {
  useRef,
  useEffect,
  useImperativeHandle,
  forwardRef,
  useState,
  useCallback,
} from "react";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

export interface AvatarHandle {
  /** Feed a base64-encoded PCM audio chunk for playback + lip sync */
  feedAudio: (base64: string, mimeType?: string) => void;
  /** Stop all audio playback immediately (e.g. user interruption) */
  stopAudio: () => void;
  /** Initialize the streaming session (call once before feedAudio) */
  startStream: () => Promise<void>;
  /** Returns true once TalkingHead has fully loaded the avatar */
  isReady: () => boolean;
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function base64ToInt16(base64: string): Int16Array {
  const binary = atob(base64);
  const len = binary.length;
  const buffer = new ArrayBuffer(len);
  const bytes = new Uint8Array(buffer);
  for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
  return new Int16Array(buffer);
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

const Avatar = forwardRef<AvatarHandle, { className?: string }>(
  function Avatar({ className }, ref) {
    const containerRef = useRef<HTMLDivElement>(null);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const headRef = useRef<any>(null);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const headAudioRef = useRef<any>(null);
    const animFrameRef = useRef<number>(0);
    const lastTimeRef = useRef<number>(0);
    const mountedRef = useRef(true);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const failedRef = useRef(false);

    /* ---- Initialise TalkingHead only (HeadAudio set up in startStream) ---- */
    useEffect(() => {
      mountedRef.current = true;

      // Suppress Next.js dev overlay for TalkingHead errors
      const onError = (e: ErrorEvent) => {
        if (e.message?.includes("TalkingHead") || e.message?.includes("Blend shapes") || e.message?.includes("three")) {
          e.preventDefault();
        }
      };
      const onRejection = (e: PromiseRejectionEvent) => {
        const msg = e.reason?.message || String(e.reason);
        if (msg.includes("TalkingHead") || msg.includes("Blend shapes") || msg.includes("three")) {
          e.preventDefault();
        }
      };
      window.addEventListener("error", onError);
      window.addEventListener("unhandledrejection", onRejection);

      const init = async () => {
        try {
          const thUrl = "/modules/talkinghead.mjs";
          const thMod = await import(/* webpackIgnore: true */ thUrl);
          const TalkingHead = thMod.TalkingHead;
          if (!mountedRef.current || !containerRef.current) return;

          let head: ReturnType<typeof TalkingHead>;
          try {
            head = new TalkingHead(containerRef.current, {
              pcmSampleRate: 24000,
              lipsyncModules: ["en"],
              lipsyncLang: "en",
              cameraView: "upper",
              modelFPS: 30,
              cameraRotateEnable: false,
              cameraPanEnable: false,
              cameraZoomEnable: false,
              lightAmbientIntensity: 2,
              lightDirectIntensity: 30,
              avatarMood: "neutral",
              avatarIdleEyeContact: 0.6,
              avatarIdleHeadMove: 0.6,
              avatarSpeakingEyeContact: 0.8,
              avatarSpeakingHeadMove: 0.4,
            });
          } catch (constructErr) {
            console.warn("TalkingHead constructor failed:", constructErr);
            if (mountedRef.current) {
              failedRef.current = true;
              setError("Avatar engine failed to initialize");
              setLoading(false);
            }
            return;
          }

          try {
            await head.showAvatar(
              {
                url: "/avatars/sarah.glb",
                body: "F",
                avatarMood: "neutral",
                lipsyncLang: "en",
              },
              (ev: ProgressEvent) => {
                if (ev.lengthComputable) {
                  const pct = Math.round((ev.loaded / ev.total) * 100);
                  console.log(`Avatar loading: ${pct}%`);
                }
              }
            );
          } catch (avatarErr) {
            console.warn("showAvatar failed:", avatarErr);
            if (mountedRef.current) {
              failedRef.current = true;
              setError("Avatar model failed to load");
              setLoading(false);
            }
            return;
          }

          headRef.current = head;

          // Animation frame loop — drives HeadAudio viseme smoothing once HeadAudio is ready
          const tick = (time: number) => {
            if (!mountedRef.current) return;
            const dt = lastTimeRef.current ? time - lastTimeRef.current : 16;
            lastTimeRef.current = time;
            headAudioRef.current?.update(dt);
            animFrameRef.current = requestAnimationFrame(tick);
          };
          animFrameRef.current = requestAnimationFrame(tick);

          if (mountedRef.current) setLoading(false);
        } catch (err) {
          console.error("Avatar init error:", err);
          if (mountedRef.current) {
            failedRef.current = true;
            setError(err instanceof Error ? err.message : "Failed to load avatar");
            setLoading(false);
          }
        }
      };

      init();

      return () => {
        mountedRef.current = false;
        window.removeEventListener("error", onError);
        window.removeEventListener("unhandledrejection", onRejection);
        if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
        try { headAudioRef.current?.stop(); } catch { /* */ }
        try { headRef.current?.stop(); } catch { /* */ }
      };
    }, []);

    /* ---- Imperative API exposed to parent ---- */

    const startStream = useCallback(async () => {
      const head = headRef.current;
      if (!head) return;

      // Stop existing HeadAudio before re-initialising (e.g. interview restart)
      if (headAudioRef.current) {
        try { headAudioRef.current.stop(); } catch { /* */ }
        headAudioRef.current = null;
      }

      // streamStart may call initAudioGraph(24000) which recreates head.audioCtx
      // and all audio nodes — so HeadAudio MUST be set up AFTER this call.
      await head.streamStart({
        sampleRate: 24000,
        lipsyncLang: "en",
      });

      // ---- Set up HeadAudio with the (possibly new) AudioContext ----
      try {
        const audioCtx = head.audioCtx as AudioContext;

        // The worklet module must be added to the current AudioContext
        await audioCtx.audioWorklet.addModule("/modules/headworklet.mjs");

        const haUrl = "/modules/headaudio.mjs";
        const haMod = await import(/* webpackIgnore: true */ haUrl);
        const HeadAudio = haMod.HeadAudio;

        const headAudio = new HeadAudio(audioCtx);
        await headAudio.loadModel("/modules/model-en-mixed.bin");

        // Route TalkingHead's stream audio into HeadAudio for viseme analysis
        if (head.audioStreamGainNode) {
          head.audioStreamGainNode.connect(headAudio);
        } else if (head.audioAnalyzerNode) {
          head.audioAnalyzerNode.connect(headAudio);
        }

        // When HeadAudio detects a viseme, update TalkingHead's morph targets
        headAudio.onvalue = (visemeName: string, value: number) => {
          const h = headRef.current;
          if (!h || !h.mtAvatar) return;
          const mt = h.mtAvatar[visemeName];
          if (mt) {
            mt.realtime = value > 0.01 ? value : null;
            mt.needsUpdate = true;
          }
        };

        headAudio.start();
        headAudioRef.current = headAudio;
      } catch (err) {
        // HeadAudio is optional — avatar still works, just without audio-driven visemes
        console.warn("HeadAudio setup failed, lip sync will be limited:", err);
      }
    }, []);

    const feedAudio = useCallback((base64: string) => {
      const head = headRef.current;
      if (!head) return;
      const int16 = base64ToInt16(base64);
      head.streamAudio({ audio: int16 });
    }, []);

    const stopAudio = useCallback(() => {
      const head = headRef.current;
      if (!head) return;
      try {
        head.streamInterrupt();
      } catch {
        /* ignore if not streaming */
      }
    }, []);

    const isReady = useCallback(() => headRef.current !== null || failedRef.current, []);

    useImperativeHandle(ref, () => ({ feedAudio, stopAudio, startStream, isReady }), [
      feedAudio,
      stopAudio,
      startStream,
      isReady,
    ]);

    /* ---- Render ---- */
    return (
      <div className={`relative ${className ?? ""}`}>
        <div
          ref={containerRef}
          className="w-full h-full"
          style={{ minHeight: 300 }}
        />
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-100/80">
            <div className="text-slate-500 text-sm flex items-center gap-2">
              <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Loading avatar...
            </div>
          </div>
        )}
        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-red-50/80">
            <p className="text-red-600 text-sm">Avatar error: {error}</p>
          </div>
        )}
      </div>
    );
  }
);

export default Avatar;
