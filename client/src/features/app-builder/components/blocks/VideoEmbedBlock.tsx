/**
 * Video Embed Block — YouTube or Vimeo embed via iframe.
 */

import { Video } from "lucide-react";
import type { BlockRenderProps } from "../BlockRenderer";
import type { VideoEmbedConfig } from "../../types/appBuilder";

function getEmbedUrl(url: string): string | null {
  const ytMatch = url.match(
    /(?:youtube\.com\/(?:watch\?v=|embed\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})/,
  );
  if (ytMatch) return `https://www.youtube.com/embed/${ytMatch[1]}`;

  const vimeoMatch = url.match(/vimeo\.com\/(\d+)/);
  if (vimeoMatch) return `https://player.vimeo.com/video/${vimeoMatch[1]}`;

  return null;
}

export default function VideoEmbedBlock({ block }: BlockRenderProps) {
  const config = block.config as VideoEmbedConfig;
  const embedUrl = config.url ? getEmbedUrl(config.url) : null;
  const paddingTop = config.aspectRatio === "4:3" ? "75%" : "56.25%";

  if (!embedUrl) {
    return (
      <div className="bg-gray-100 rounded-lg flex items-center justify-center py-16">
        <div className="text-center text-gray-400">
          <Video className="h-8 w-8 mx-auto mb-2" />
          <p className="text-sm">
            {config.url ? "Invalid video URL" : "No video URL set"}
          </p>
          <p className="text-xs mt-1">Paste a YouTube or Vimeo link</p>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div
        className="relative rounded-lg overflow-hidden"
        style={{ paddingTop }}
      >
        <iframe
          src={embedUrl}
          className="absolute inset-0 w-full h-full"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
          title="Video embed"
        />
      </div>
      {config.caption && (
        <p className="text-xs text-gray-500 text-center mt-2">
          {config.caption}
        </p>
      )}
    </div>
  );
}
