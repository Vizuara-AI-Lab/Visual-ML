/**
 * PublishPanel — Right panel for slug editing, publish toggle, and public URL.
 */

import { useState, useEffect } from "react";
import { Check, Copy, ExternalLink, Globe, Loader2 } from "lucide-react";
import { toast } from "react-hot-toast";
import { usePublishApp, useCheckSlug, useUpdateApp } from "../hooks/useAppBuilder";
import { useAppBuilderStore } from "../store/appBuilderStore";
import type { CustomApp } from "../types/appBuilder";

interface PublishPanelProps {
  app: CustomApp;
}

export default function PublishPanel({ app }: PublishPanelProps) {
  const [slug, setSlug] = useState(app.slug);
  const [copied, setCopied] = useState(false);
  const publishApp = usePublishApp(app.id);
  const updateApp = useUpdateApp(app.id);
  const markClean = useAppBuilderStore((s) => s.markClean);
  const { data: slugCheck, isFetching: checkingSlug } = useCheckSlug(
    slug !== app.slug ? slug : "",
  );

  useEffect(() => {
    setSlug(app.slug);
  }, [app.slug]);

  const publicUrl = `${window.location.origin}/app/${slug}`;
  const slugAvailable = slug === app.slug || slugCheck?.available;

  const handlePublish = async (publish: boolean) => {
    try {
      // Always save blocks & theme before publishing so the DB has the latest state.
      // Read from store directly via getState() to guarantee fresh values.
      if (publish) {
        const { blocks, theme } = useAppBuilderStore.getState();
        await updateApp.mutateAsync({ blocks, theme });
        markClean();
      }
      await publishApp.mutateAsync({
        is_published: publish,
        slug: slug !== app.slug ? slug : undefined,
      });
      toast.success(publish ? "App published!" : "App unpublished");
    } catch {
      toast.error("Failed to update publish status");
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(publicUrl);
    setCopied(true);
    toast.success("URL copied!");
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="p-4 space-y-6">
      <div className="pb-3 border-b">
        <h2 className="text-sm font-semibold text-gray-700">Publish</h2>
        <p className="text-xs text-gray-400">Share your app with the world</p>
      </div>

      {/* Slug editor */}
      <div>
        <label className="block text-xs font-medium text-gray-500 mb-1.5">
          URL Slug
        </label>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400">/app/</span>
          <input
            type="text"
            value={slug}
            onChange={(e) =>
              setSlug(
                e.target.value
                  .toLowerCase()
                  .replace(/[^a-z0-9-]/g, "-")
                  .replace(/-+/g, "-"),
              )
            }
            className="flex-1 px-2.5 py-1.5 border rounded-md text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
          />
        </div>
        {slug !== app.slug && (
          <p className="text-xs mt-1">
            {checkingSlug ? (
              <span className="text-gray-400">Checking...</span>
            ) : slugAvailable ? (
              <span className="text-green-600">Available</span>
            ) : (
              <span className="text-red-500">Slug already taken</span>
            )}
          </p>
        )}
      </div>

      {/* Public URL */}
      {app.is_published && (
        <div>
          <label className="block text-xs font-medium text-gray-500 mb-1.5">
            Public URL
          </label>
          <div className="flex items-center gap-1.5 bg-gray-50 rounded-lg px-3 py-2">
            <Globe className="h-3.5 w-3.5 text-green-500 shrink-0" />
            <span className="text-xs text-gray-600 truncate flex-1">{publicUrl}</span>
            <button onClick={handleCopy} className="p-1 hover:bg-gray-200 rounded">
              {copied ? (
                <Check className="h-3.5 w-3.5 text-green-500" />
              ) : (
                <Copy className="h-3.5 w-3.5 text-gray-400" />
              )}
            </button>
            <a
              href={publicUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="p-1 hover:bg-gray-200 rounded"
            >
              <ExternalLink className="h-3.5 w-3.5 text-gray-400" />
            </a>
          </div>
        </div>
      )}

      {/* Publish toggle */}
      <button
        onClick={() => handlePublish(!app.is_published)}
        disabled={publishApp.isPending || updateApp.isPending || (!slugAvailable && slug !== app.slug)}
        className={`w-full py-2.5 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed ${
          app.is_published
            ? "bg-red-50 text-red-600 hover:bg-red-100 border border-red-200"
            : "bg-indigo-600 text-white hover:bg-indigo-700"
        }`}
      >
        {publishApp.isPending || updateApp.isPending ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : app.is_published ? (
          "Unpublish"
        ) : (
          "Publish App"
        )}
      </button>

      {/* Stats */}
      {app.is_published && (
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-gray-50 rounded-lg p-3 text-center">
            <p className="text-lg font-bold text-gray-900">{app.view_count}</p>
            <p className="text-xs text-gray-500">Views</p>
          </div>
          <div className="bg-gray-50 rounded-lg p-3 text-center">
            <p className="text-lg font-bold text-gray-900">{app.execution_count}</p>
            <p className="text-xs text-gray-500">Executions</p>
          </div>
        </div>
      )}
    </div>
  );
}
