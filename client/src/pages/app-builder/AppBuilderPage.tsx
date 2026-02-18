/**
 * App Builder Page — Protected route wrapper.
 * Loads the AppBuilder component for a given pipeline/app.
 */

import { useParams, useNavigate } from "react-router";
import { useEffect, useState } from "react";
import { useCustomApps, useCreateApp } from "../../features/app-builder";
import AppBuilder from "../../features/app-builder/components/AppBuilder";

export default function AppBuilderPage() {
  const { appId } = useParams<{ appId: string }>();
  const navigate = useNavigate();
  const parsedId = appId ? parseInt(appId, 10) : undefined;

  return (
    <div className="h-screen w-screen overflow-hidden bg-gray-50">
      <AppBuilder appId={parsedId} />
    </div>
  );
}
