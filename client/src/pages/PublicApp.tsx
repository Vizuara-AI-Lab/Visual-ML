/**
 * Public App Page — Renders a published custom app at /app/:slug
 * No authentication required.
 */

import { useParams } from "react-router";
import { usePublicApp } from "../features/app-builder";
import PublicAppView from "../features/app-builder/components/PublicAppView";

export default function PublicApp() {
  const { slug } = useParams<{ slug: string }>();

  const { data: app, isLoading, error } = usePublicApp(slug);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="flex flex-col items-center gap-3">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-indigo-600 border-t-transparent" />
          <p className="text-gray-500">Loading app...</p>
        </div>
      </div>
    );
  }

  if (error || !app) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-900 mb-2">App Not Found</h1>
          <p className="text-gray-500">
            This app doesn't exist or hasn't been published yet.
          </p>
        </div>
      </div>
    );
  }

  return <PublicAppView app={app} slug={slug!} />;
}
