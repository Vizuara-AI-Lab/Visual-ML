/**
 * Audio Player Hook - Simplified
 *
 * Manages audio playback for mentor voice messages
 */

import { useRef, useCallback } from "react";
import { useMentorStore } from "../store/mentorStore";
import { mentorApi } from "../api/mentorApi";
import { toast } from "react-hot-toast";

export const useAudioPlayer = () => {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const { isSpeaking, preferences, setIsSpeaking } = useMentorStore();

  // Simple base64 to audio blob
  const playAudioFromBase64 = useCallback(
    (base64Audio: string) => {
      try {
        console.log("[Audio] Decoding base64 audio...");

        // Decode base64 to binary
        const binaryString = atob(base64Audio);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }

        // Create blob with OGG format
        const blob = new Blob([bytes], { type: "audio/ogg; codecs=opus" });
        const audioUrl = URL.createObjectURL(blob);

        console.log("[Audio] Created blob URL:", audioUrl);

        // Create new audio element for each playback
        if (audioRef.current) {
          audioRef.current.pause();
          URL.revokeObjectURL(audioRef.current.src);
        }

        audioRef.current = new Audio(audioUrl);

        audioRef.current.onloadeddata = () => {
          console.log("[Audio] Audio loaded, playing...");
        };

        audioRef.current.onplay = () => {
          console.log("[Audio] Playback started");
          setIsSpeaking(true);
        };

        audioRef.current.onended = () => {
          console.log("[Audio] Playback ended");
          setIsSpeaking(false);
          URL.revokeObjectURL(audioUrl);
        };

        audioRef.current.onerror = (e) => {
          console.error("[Audio] Playback error:", e);
          toast.error("Audio playback failed");
          setIsSpeaking(false);
          URL.revokeObjectURL(audioUrl);
        };

        // Start playback
        audioRef.current.play().catch((err) => {
          console.error("[Audio] Play failed:", err);

          // Check if it's a browser autoplay restriction
          if (err.name === "NotAllowedError") {
            console.warn(
              "[Audio] Autoplay blocked by browser - user interaction required first",
            );
            // Don't show toast for autoplay blocking since it's expected behavior
            // User can click the play button to manually play audio
          } else {
            toast.error("Could not play audio");
          }
          setIsSpeaking(false);
        });
      } catch (error) {
        console.error("[Audio] Error playing audio:", error);
        toast.error("Failed to play audio");
        setIsSpeaking(false);
      }
    },
    [setIsSpeaking],
  );

  const playText = useCallback(
    async (text: string, force: boolean = false) => {
      console.log(
        "[TTS] Generating speech for:",
        text.substring(0, 50) + "...",
      );
      console.log("[TTS] Voice mode:", preferences.voice_mode, "force:", force);

      // Check voice mode preference (skip check if forced, e.g. for greetings)
      if (!force && preferences.voice_mode === "text_first") {
        console.log("[TTS] Text-first mode - skipping auto-play");
        return;
      }

      try {
        const response = await mentorApi.generateSpeech(
          text,
          preferences.personality,
        );

        console.log("[TTS] Response received:", {
          success: response.success,
          hasAudio: !!response.audio_base64,
          audioLength: response.audio_base64?.length,
          cached: response.cached,
          error: response.error,
        });

        if (response.success && response.audio_base64) {
          playAudioFromBase64(response.audio_base64);
        } else {
          console.warn("[TTS] No audio received:", response.error);
          if (response.error) {
            toast.error(`TTS Error: ${response.error}`);
          }
        }
      } catch (error) {
        console.error("[TTS] API error:", error);
        toast.error("Failed to generate speech");
      }
    },
    [preferences.personality, preferences.voice_mode, playAudioFromBase64],
  );

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
  }, [setIsSpeaking]);

  return {
    playText,
    playAudioFromBase64,
    pause,
    resume,
    stop,
    isSpeaking,
  };
};
