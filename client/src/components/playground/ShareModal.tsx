import { useState, useEffect } from "react";
import { X, Copy, Check, Link2, Eye, Users, Settings } from "lucide-react";

interface ShareModalProps {
  isOpen: boolean;
  onClose: () => void;
  projectId: number;
  projectName: string;
}

interface ShareSettings {
  isPublic: boolean;
  allowCloning: boolean;
  shareToken?: string;
  viewCount?: number;
  cloneCount?: number;
}

export const ShareModal = ({
  isOpen,
  onClose,
  projectId,
  projectName,
}: ShareModalProps) => {
  const [shareSettings, setShareSettings] = useState<ShareSettings>({
    isPublic: false,
    allowCloning: true,
  });
  const [isLoading, setIsLoading] = useState(false);
  const [isCopied, setIsCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch current share status when modal opens
  useEffect(() => {
    if (isOpen && projectId) {
      fetchShareStatus();
    }
  }, [isOpen, projectId]);

  const fetchShareStatus = async () => {
    try {
      const response = await fetch(
        `${import.meta.env.VITE_API_URL}/api/v1/projects/${projectId}/share-stats`,
        {
          credentials: "include",
        },
      );
      if (response.ok) {
        const data = await response.json();
        setShareSettings({
          isPublic: data.is_public || false,
          allowCloning:
            data.allow_cloning !== undefined ? data.allow_cloning : true,
          shareToken: data.share_token,
          viewCount: data.view_count || 0,
          cloneCount: data.clone_count || 0,
        });
      }
    } catch (err) {
      console.error("Failed to fetch share status:", err);
    }
  };

  const handleGenerateLink = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(
        `${import.meta.env.VITE_API_URL}/api/v1/projects/${projectId}/share`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          credentials: "include",
          body: JSON.stringify({
            is_public: shareSettings.isPublic,
            allow_cloning: shareSettings.allowCloning,
          }),
        },
      );

      if (!response.ok) {
        throw new Error("Failed to generate share link");
      }

      const data = await response.json();
      setShareSettings({
        ...shareSettings,
        shareToken: data.share_token,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to share project");
    } finally {
      setIsLoading(false);
    }
  };

  const handleUnshare = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(
        `${import.meta.env.VITE_API_URL}/api/v1/projects/${projectId}/share`,
        {
          method: "DELETE",
          credentials: "include",
        },
      );

      if (!response.ok) {
        throw new Error("Failed to unshare project");
      }

      setShareSettings({
        isPublic: false,
        allowCloning: true,
        shareToken: undefined,
        viewCount: 0,
        cloneCount: 0,
      });
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to unshare project",
      );
    } finally {
      setIsLoading(false);
    }
  };

  const copyToClipboard = async () => {
    if (shareSettings.shareToken) {
      const shareUrl = `${window.location.origin}/shared/${shareSettings.shareToken}`;
      try {
        await navigator.clipboard.writeText(shareUrl);
        setIsCopied(true);
        setTimeout(() => setIsCopied(false), 2000);
      } catch (err) {
        console.error("Failed to copy:", err);
      }
    }
  };

  if (!isOpen) return null;

  const shareUrl = shareSettings.shareToken
    ? `${window.location.origin}/shared/${shareSettings.shareToken}`
    : "";

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl mx-4 overflow-hidden">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-200 flex items-center justify-between bg-gradient-to-r from-blue-50 to-white">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-600 rounded-lg">
              <Link2 className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-slate-800">
                Share Project
              </h2>
              <p className="text-sm text-slate-500">{projectName}</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-slate-500" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {error && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
              {error}
            </div>
          )}

          {/* Share Settings */}
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-slate-50 rounded-lg border border-slate-200">
              <div className="flex items-center gap-3">
                <Settings className="w-5 h-5 text-slate-600" />
                <div>
                  <p className="font-semibold text-slate-800">Make Public</p>
                  <p className="text-sm text-slate-500">
                    Anyone with the link can view this project
                  </p>
                </div>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={shareSettings.isPublic}
                  onChange={(e) =>
                    setShareSettings({
                      ...shareSettings,
                      isPublic: e.target.checked,
                    })
                  }
                  className="sr-only peer"
                  disabled={isLoading}
                />
                <div className="w-11 h-6 bg-slate-300 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
              </label>
            </div>

            <div className="flex items-center justify-between p-4 bg-slate-50 rounded-lg border border-slate-200">
              <div className="flex items-center gap-3">
                <Users className="w-5 h-5 text-slate-600" />
                <div>
                  <p className="font-semibold text-slate-800">Allow Cloning</p>
                  <p className="text-sm text-slate-500">
                    Others can clone this project to their account
                  </p>
                </div>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={shareSettings.allowCloning}
                  onChange={(e) =>
                    setShareSettings({
                      ...shareSettings,
                      allowCloning: e.target.checked,
                    })
                  }
                  className="sr-only peer"
                  disabled={isLoading}
                />
                <div className="w-11 h-6 bg-slate-300 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
              </label>
            </div>
          </div>

          {/* Share Link */}
          {shareSettings.shareToken ? (
            <div className="space-y-4">
              <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                <div className="flex items-center justify-between mb-3">
                  <p className="font-semibold text-green-800">
                    Your project is shared!
                  </p>
                  <button
                    onClick={handleUnshare}
                    disabled={isLoading}
                    className="text-sm text-red-600 hover:text-red-700 font-medium disabled:opacity-50"
                  >
                    Unshare
                  </button>
                </div>
                <div className="flex items-center gap-2 p-3 bg-white rounded-lg border border-green-300">
                  <input
                    type="text"
                    value={shareUrl}
                    readOnly
                    className="flex-1 bg-transparent border-none outline-none text-sm text-slate-700 font-mono"
                  />
                  <button
                    onClick={copyToClipboard}
                    className="px-3 py-1.5 bg-green-600 hover:bg-green-700 text-white rounded-lg flex items-center gap-2 transition-colors text-sm font-medium"
                  >
                    {isCopied ? (
                      <>
                        <Check className="w-4 h-4" />
                        Copied!
                      </>
                    ) : (
                      <>
                        <Copy className="w-4 h-4" />
                        Copy
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Statistics */}
              {(shareSettings.viewCount! > 0 ||
                shareSettings.cloneCount! > 0) && (
                <div className="flex gap-4">
                  <div className="flex-1 p-4 bg-blue-50 rounded-lg border border-blue-200">
                    <div className="flex items-center gap-2 mb-1">
                      <Eye className="w-4 h-4 text-blue-600" />
                      <p className="text-sm font-medium text-blue-800">Views</p>
                    </div>
                    <p className="text-2xl font-bold text-blue-900">
                      {shareSettings.viewCount}
                    </p>
                  </div>
                  <div className="flex-1 p-4 bg-purple-50 rounded-lg border border-purple-200">
                    <div className="flex items-center gap-2 mb-1">
                      <Users className="w-4 h-4 text-purple-600" />
                      <p className="text-sm font-medium text-purple-800">
                        Clones
                      </p>
                    </div>
                    <p className="text-2xl font-bold text-purple-900">
                      {shareSettings.cloneCount}
                    </p>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-8">
              <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Link2 className="w-8 h-8 text-slate-400" />
              </div>
              <h3 className="text-lg font-semibold text-slate-800 mb-2">
                Project Not Shared
              </h3>
              <p className="text-sm text-slate-500 mb-6">
                Generate a shareable link to let others view{" "}
                {shareSettings.allowCloning ? "and clone " : ""}
                your project
              </p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 bg-slate-50 border-t border-slate-200 flex items-center justify-between">
          <button
            onClick={onClose}
            className="px-4 py-2 text-slate-600 hover:text-slate-800 font-medium transition-colors"
          >
            Close
          </button>
          {!shareSettings.shareToken && (
            <button
              onClick={handleGenerateLink}
              disabled={isLoading || !shareSettings.isPublic}
              className="px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 disabled:cursor-not-allowed text-white rounded-lg font-semibold transition-all shadow-lg hover:shadow-xl disabled:shadow-none"
            >
              {isLoading ? "Generating..." : "Generate Share Link"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
