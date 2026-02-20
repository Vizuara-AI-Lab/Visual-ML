"""
Text-to-Speech Service (Inworld Integration)

Converts mentor text messages to natural voice audio using Inworld TTS API.
Uses cloned Rajat Sir voice for all mentor interactions with audio caching.
"""

from typing import Optional
import httpx
import base64
import hashlib
from pathlib import Path
from app.mentor.schemas import PersonalityStyle, TTSResponse
from app.core.config import settings
from app.core.logging import logger


class TTSService:
    """Inworld Text-to-Speech integration for mentor voice."""

    def __init__(self):
        self.api_key = getattr(settings, "INWORLD_API_KEY", None)
        self.workspace_id = getattr(settings, "INWORLD_WORKSPACE_ID", None)
        self.character_id = getattr(settings, "INWORLD_CHARACTER_ID", None)
        self.cache_dir = Path(settings.UPLOAD_DIR) / "tts_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cloned voice ID for all mentor interactions
        self.voice_id = "default-tdxiowf-g_jzcmgci-i_iw__rajat_sir_voice_clone"

    async def generate_speech(
        self,
        text: str,
        personality: PersonalityStyle = PersonalityStyle.ENCOURAGING,
        voice_id: Optional[str] = None,
        cache_key: Optional[str] = None,
        return_base64: bool = True,
    ) -> TTSResponse:
        """
        Generate speech audio from text using Inworld TTS.

        Args:
            text: Text to convert to speech
            personality: Used for cache key differentiation
            voice_id: Inworld voice ID (preset name or cloned ID). Falls back to default.
            cache_key: Optional key for caching
            return_base64: Return audio as base64 (True) or save file (False)

        Returns:
            TTSResponse with audio data or URL
        """
        try:
            # Check if API credentials are configured
            if not self.api_key:
                logger.warning("Inworld API key not configured - TTS disabled")
                return TTSResponse(
                    success=False,
                    audio_url=None,
                    audio_base64=None,
                    duration_seconds=None,
                    cached=False,
                    error="TTS service not configured. Please add INWORLD_API_KEY to environment.",
                )

            # Resolve voice ID
            effective_voice_id = voice_id or self.voice_id

            # Generate cache key from text + personality + voice
            if not cache_key:
                cache_key = self._generate_cache_key(text, personality, effective_voice_id)

            # Check cache first
            cached_audio = self._get_cached_audio(cache_key)
            if cached_audio:
                logger.info(f"Serving TTS from cache: {cache_key}")
                return TTSResponse(
                    success=True,
                    audio_url=None,
                    audio_base64=cached_audio if return_base64 else None,
                    duration_seconds=None,
                    cached=True,
                    error=None,
                )

            # Call Inworld API
            logger.info(f"Generating TTS with Inworld for text: {text[:50]}...")

            audio_data = await self._call_inworld_api(text, effective_voice_id)

            if not audio_data:
                raise Exception("Failed to generate audio from Inworld")

            # Cache the audio
            self._cache_audio(cache_key, audio_data)

            # Return response
            if return_base64:
                audio_base64 = base64.b64encode(audio_data).decode("utf-8")
                return TTSResponse(
                    success=True,
                    audio_url=None,
                    audio_base64=audio_base64,
                    duration_seconds=None,
                    cached=False,
                    error=None,
                )
            else:
                # Save to file and return URL (using .ogg for OGG_OPUS format)
                audio_path = self.cache_dir / f"{cache_key}.ogg"
                audio_path.write_bytes(audio_data)

                return TTSResponse(
                    success=True,
                    audio_url=f"/api/v1/mentor/audio/{cache_key}.ogg",
                    audio_base64=None,
                    duration_seconds=None,
                    cached=False,
                    error=None,
                )

        except Exception as e:
            logger.error(f"TTS generation failed: {str(e)}", exc_info=True)
            return TTSResponse(
                success=False,
                audio_url=None,
                audio_base64=None,
                duration_seconds=None,
                cached=False,
                error=str(e),
            )

    async def _call_inworld_api(self, text: str, voice_id: str) -> Optional[bytes]:
        """
        Call Inworld TTS API to generate audio using streaming endpoint.
        """
        try:
            model_id = "inworld-tts-1.5-mini"

            url = "https://api.inworld.ai/tts/v1/voice:stream"

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Basic {self.api_key}",
            }

            payload = {
                "text": text,
                "voice_id": voice_id,
                "model_id": model_id,
                "audio_config": {
                    "audio_encoding": "OGG_OPUS",
                    "sample_rate_hertz": 24000,
                    "bit_rate": 32000,
                },
            }

            logger.info(f"🎤 TTS Request: voice={voice_id}, text_len={len(text)}")

            async with httpx.AsyncClient(timeout=30.0) as client:
                # Make streaming request
                response = await client.post(url, headers=headers, json=payload)

                if response.status_code != 200:
                    error_text = response.text
                    logger.error(f"❌ Inworld API error {response.status_code}: {error_text}")
                    return None

                # Collect audio chunks
                audio_chunks = []
                chunk_count = 0

                for line in response.text.split("\n"):
                    if line.strip():
                        try:
                            import json

                            chunk_data = json.loads(line)
                            result = chunk_data.get("result", {})

                            if "audioContent" in result:
                                audio_chunk = base64.b64decode(result["audioContent"])
                                audio_chunks.append(audio_chunk)
                                chunk_count += 1

                        except Exception as e:
                            logger.debug(f"Skipping chunk: {e}")
                            continue

                if chunk_count > 0:
                    audio_data = b"".join(audio_chunks)
                    logger.info(f"✅ TTS Success: {chunk_count} chunks, {len(audio_data)} bytes")
                    return audio_data
                else:
                    logger.error("❌ No audio chunks received")
                    return None

        except Exception as e:
            logger.error(f"❌ TTS API error: {str(e)}", exc_info=True)
            return None

    def _generate_cache_key(self, text: str, personality: PersonalityStyle, voice_id: str = "") -> str:
        """Generate cache key from text, personality, and voice."""
        combined = f"{text}_{personality.value}_{voice_id}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _get_cached_audio(self, cache_key: str) -> Optional[str]:
        """Retrieve cached audio as base64."""
        # Try OGG format first (current), then fallback to MP3 (legacy)
        cache_file = self.cache_dir / f"{cache_key}.ogg"
        if not cache_file.exists():
            cache_file = self.cache_dir / f"{cache_key}.mp3"

        if cache_file.exists():
            try:
                audio_bytes = cache_file.read_bytes()
                return base64.b64encode(audio_bytes).decode("utf-8")
            except Exception as e:
                logger.error(f"Error reading cached audio: {str(e)}")
                return None
        return None

    def _cache_audio(self, cache_key: str, audio_data: bytes) -> None:
        """Cache audio data to file (OGG format)."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.ogg"
            cache_file.write_bytes(audio_data)
            logger.info(f"Cached TTS audio: {cache_key}")
        except Exception as e:
            logger.error(f"Error caching audio: {str(e)}")

    def clear_cache(self, max_age_days: int = 7) -> int:
        """Clear old cached audio files."""
        import time

        cleared = 0

        try:
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 60 * 60

            # Clear both OGG and MP3 files
            for cache_file in self.cache_dir.glob("*.ogg"):
                file_age = current_time - cache_file.stat().st_mtime
                if file_age > max_age_seconds:
                    cache_file.unlink()
                    cleared += 1

            for cache_file in self.cache_dir.glob("*.mp3"):
                file_age = current_time - cache_file.stat().st_mtime
                if file_age > max_age_seconds:
                    cache_file.unlink()
                    cleared += 1

            logger.info(f"Cleared {cleared} old TTS cache files")
            return cleared

        except Exception as e:
            logger.error(f"Error clearing TTS cache: {str(e)}")
            return 0

    async def clone_voice(
        self,
        display_name: str,
        audio_data: bytes,
        lang_code: str = "EN_US",
    ) -> dict:
        """
        Clone a voice using the Inworld Voice API.

        Args:
            display_name: Human-readable name for the cloned voice
            audio_data: Raw audio bytes (WAV or MP3)
            lang_code: Language code (default EN_US)

        Returns:
            dict with voiceId and displayName
        """
        if not self.api_key:
            raise Exception("Inworld API key not configured")

        url = "https://api.inworld.ai/voices/v1/voices:clone"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Basic {self.api_key}",
        }

        audio_b64 = base64.b64encode(audio_data).decode("utf-8")

        request_data = {
            "displayName": display_name,
            "langCode": lang_code,
            "voiceSamples": [{"audioData": audio_b64}],
        }

        logger.info(f"Cloning voice: {display_name}, audio_size={len(audio_data)} bytes")

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, headers=headers, json=request_data)

            if response.status_code != 200:
                error_text = response.text
                logger.error(f"Voice clone API error {response.status_code}: {error_text}")
                raise Exception(f"Voice cloning failed: {error_text}")

            result = response.json()
            voice = result.get("voice", {})

            logger.info(f"Voice cloned: {voice.get('voiceId', 'unknown')}")

            return {
                "voiceId": voice.get("voiceId", ""),
                "displayName": voice.get("displayName", display_name),
            }


# Singleton instance
tts_service = TTSService()
