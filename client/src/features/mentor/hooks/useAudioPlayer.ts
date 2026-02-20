/**
 * Audio Player Hook
 *
 * Manages audio playback for mentor voice messages.
 * Includes a client-side cache to avoid duplicate TTS API calls
 * and a prefetch method for look-ahead audio generation.
 */

import { useRef, useCallback } from "react";
import { useMentorStore } from "../store/mentorStore";
import { mentorApi } from "../api/mentorApi";
import { toast } from "react-hot-toast";

// ── Simple string hash for cache keys ───────────────────────────

function simpleHash(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const ch = str.charCodeAt(i);
    hash = ((hash << 5) - hash + ch) | 0;
  }
  return hash.toString(36);
}

// ── Module-level audio cache (shared across hook instances) ─────

const audioCache = new Map<string, string>();

export const useAudioPlayer = () => {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const requestIdRef = useRef(0);
  const { isSpeaking, preferences, setIsSpeaking } = useMentorStore();

  // ── Base64 → Audio playback ───────────────────────────────────

  const playAudioFromBase64 = useCallback(
    (base64Audio: string) => {
      try {
        const binaryString = atob(base64Audio);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }

        const blob = new Blob([bytes], { type: "audio/ogg; codecs=opus" });
        const audioUrl = URL.createObjectURL(blob);

        // Stop any currently playing audio
        if (audioRef.current) {
          audioRef.current.pause();
          URL.revokeObjectURL(audioRef.current.src);
        }

        audioRef.current = new Audio(audioUrl);

        audioRef.current.onplay = () => setIsSpeaking(true);

        audioRef.current.onended = () => {
          setIsSpeaking(false);
          URL.revokeObjectURL(audioUrl);
        };

        audioRef.current.onerror = () => {
          toast.error("Audio playback failed");
          setIsSpeaking(false);
          URL.revokeObjectURL(audioUrl);
        };

        audioRef.current.play().catch((err) => {
          if (err.name !== "NotAllowedError") {
            toast.error("Could not play audio");
          }
          setIsSpeaking(false);
        });
      } catch {
        toast.error("Failed to play audio");
        setIsSpeaking(false);
      }
    },
    [setIsSpeaking],
  );

  // ── Play text (with cache lookup) ─────────────────────────────

  const playText = useCallback(
    async (text: string, force: boolean = false) => {
      // Check voice mode preference (skip check if forced)
      if (!force && preferences.voice_mode === "text_first") {
        return;
      }

      // Stop any currently playing audio before starting new
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.currentTime = 0;
        setIsSpeaking(false);
      }

      // Track this request to discard stale responses
      const currentRequestId = ++requestIdRef.current;

      // Check client-side cache first (keyed by text + personality + voice)
      const cacheKey = simpleHash(text + preferences.personality + preferences.voice_id);
      const cached = audioCache.get(cacheKey);
      if (cached) {
        if (currentRequestId !== requestIdRef.current) return;
        playAudioFromBase64(cached);
        return;
      }

      try {
        const response = await mentorApi.generateSpeech(
          text,
          preferences.personality,
          preferences.voice_id,
        );

        // Discard if a newer request was made while waiting
        if (currentRequestId !== requestIdRef.current) return;

        if (response.success && response.audio_base64) {
          // Store in cache
          audioCache.set(cacheKey, response.audio_base64);
          playAudioFromBase64(response.audio_base64);
        } else if (response.error) {
          toast.error(`TTS Error: ${response.error}`);
        }
      } catch {
        if (currentRequestId !== requestIdRef.current) return;
        toast.error("Failed to generate speech");
      }
    },
    [
      preferences.personality,
      preferences.voice_id,
      preferences.voice_mode,
      playAudioFromBase64,
      setIsSpeaking,
    ],
  );

  // ── Prefetch audio (fire-and-forget, populates cache) ─────────

  const prefetchAudio = useCallback(
    async (text: string) => {
      const cacheKey = simpleHash(text + preferences.personality + preferences.voice_id);
      if (audioCache.has(cacheKey)) return;

      try {
        const response = await mentorApi.generateSpeech(
          text,
          preferences.personality,
          preferences.voice_id,
        );
        if (response.success && response.audio_base64) {
          audioCache.set(cacheKey, response.audio_base64);
        }
      } catch {
        // Prefetch failures are silent
      }
    },
    [preferences.personality, preferences.voice_id],
  );

  // ── Playback controls ─────────────────────────────────────────

  const pause = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      setIsSpeaking(false);
    }
  }, [setIsSpeaking]);

  const resume = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.play();
      setIsSpeaking(true);
    }
  }, [setIsSpeaking]);

  const stop = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      setIsSpeaking(false);
    }
    // Invalidate any pending TTS requests so stale responses are discarded
    requestIdRef.current++;
  }, [setIsSpeaking]);

  return {
    playText,
    playAudioFromBase64,
    prefetchAudio,
    pause,
    resume,
    stop,
    isSpeaking,
  };
};
