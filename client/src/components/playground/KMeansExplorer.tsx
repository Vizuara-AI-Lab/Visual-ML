/**
 * KMeansExplorer — Interactive learning activities for K-Means Clustering
 * Tabbed component: Results | Cluster Map | Elbow Method | Cluster Profiles | Convergence | Quiz | How It Works
 * All chart visualizations use inline SVG (no chart libraries).
 */

import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import {
  ClipboardList,
  ScatterChart,
  LineChart,
  BarChart3,
  Play,
  Pause,
  SkipForward,
  SkipBack,
  RotateCcw,
  HelpCircle,
  Cog,
  CheckCircle,
  XCircle,
  Trophy,
  Target,
  Timer,
  Hash,
  Database,
  Layers,
  ChevronLeft,
  ChevronRight,
  Info,
} from "lucide-react";

// ═══════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════

const CLUSTER_COLORS = [
  "#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6",
  "#ec4899", "#14b8a6", "#f97316", "#06b6d4", "#84cc16",
  "#6366f1", "#e11d48", "#0ea5e9", "#a855f7", "#10b981",
  "#d946ef", "#eab308", "#64748b", "#dc2626", "#2563eb",
];

const CLUSTER_COLORS_LIGHT = [
  "#fecaca", "#bfdbfe", "#bbf7d0", "#fde68a", "#ddd6fe",
  "#fbcfe8", "#99f6e4", "#fed7aa", "#a5f3fc", "#d9f99d",
  "#c7d2fe", "#fda4af", "#bae6fd", "#e9d5ff", "#a7f3d0",
  "#f5d0fe", "#fef08a", "#cbd5e1", "#fca5a5", "#93c5fd",
];

interface KMeansExplorerProps {
  result: any;
}

type ExplorerTab =
  | "results"
  | "cluster_map"
  | "elbow"
  | "profiles"
  | "convergence"
  | "quiz"
  | "how_it_works";

export const KMeansExplorer = ({ result }: KMeansExplorerProps) => {
  const [activeTab, setActiveTab] = useState<ExplorerTab>("results");

  const tabs: { id: ExplorerTab; label: string; icon: any; available: boolean }[] = [
    { id: "results", label: "Results", icon: ClipboardList, available: true },
    { id: "cluster_map", label: "Cluster Map", icon: ScatterChart, available: !!result.pca_projection?.points?.length },
    { id: "elbow", label: "Elbow Method", icon: LineChart, available: !!result.elbow_data?.length },
    { id: "profiles", label: "Cluster Profiles", icon: BarChart3, available: !!result.cluster_profiles?.clusters?.length },
    { id: "convergence", label: "Convergence", icon: Play, available: !!result.convergence_history?.pca_iterations?.length },
    { id: "quiz", label: "Quiz", icon: HelpCircle, available: !!result.quiz_questions?.length },
    { id: "how_it_works", label: "How It Works", icon: Cog, available: true },
  ];

  return (
    <div className="space-y-4">
      {/* Tab navigation */}
      <div className="flex border-b border-gray-200 overflow-x-auto">
        {tabs
          .filter((t) => t.available)
          .map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
                  activeTab === tab.id
                    ? "border-cyan-500 text-cyan-700"
                    : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
      </div>

      {/* Tab content */}
      <div className="min-h-[400px]">
        {activeTab === "results" && <ResultsTab result={result} />}
        {activeTab === "cluster_map" && <ClusterMapTab result={result} />}
        {activeTab === "elbow" && <ElbowTab result={result} />}
        {activeTab === "profiles" && <ProfilesTab result={result} />}
        {activeTab === "convergence" && <ConvergenceTab result={result} />}
        {activeTab === "quiz" && <QuizTab questions={result.quiz_questions || []} />}
        {activeTab === "how_it_works" && <HowItWorksTab result={result} />}
      </div>
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════
// TAB 1: RESULTS
// ═══════════════════════════════════════════════════════════════════

function ResultsTab({ result }: { result: any }) {
  const silScore = result.silhouette_score ?? 0;
  const silPct = Math.max(0, Math.min(100, ((silScore + 1) / 2) * 100)); // map -1..1 → 0..100
  const silColor =
    silScore >= 0.7 ? "#14b8a6" : silScore >= 0.5 ? "#06b6d4" : silScore >= 0.25 ? "#f59e0b" : "#ef4444";
  const silLabel =
    silScore >= 0.7 ? "Strong" : silScore >= 0.5 ? "Reasonable" : silScore >= 0.25 ? "Weak" : "Poor";

  return (
    <div className="space-y-6">
      {/* Status Banner */}
      <div className="bg-gradient-to-r from-cyan-50 to-teal-50 border border-cyan-200 rounded-lg p-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-cyan-500 rounded-full flex items-center justify-center">
            <CheckCircle className="text-white w-5 h-5" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-cyan-900">K-Means Clustering Complete</h3>
            <p className="text-sm text-cyan-700">
              Found {result.n_clusters} clusters in {result.n_samples?.toLocaleString()} samples
              with {result.n_features} features
            </p>
          </div>
        </div>
      </div>

      {/* Hero Metric — Silhouette Score */}
      <div className="flex justify-center">
        <div className="relative w-36 h-36">
          <svg viewBox="0 0 120 120" className="w-full h-full -rotate-90">
            <circle cx="60" cy="60" r="52" fill="none" stroke="#e5e7eb" strokeWidth="8" />
            <circle
              cx="60" cy="60" r="52" fill="none"
              stroke={silColor} strokeWidth="8" strokeLinecap="round"
              strokeDasharray={`${(silPct / 100) * 327} 327`}
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-2xl font-bold text-gray-900">{silScore.toFixed(3)}</span>
            <span className="text-xs text-gray-500">Silhouette</span>
            <span className="text-[10px] font-medium mt-0.5 px-2 py-0.5 rounded-full" style={{ backgroundColor: silColor + "20", color: silColor }}>
              {silLabel}
            </span>
          </div>
        </div>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard icon={<Target className="w-3.5 h-3.5" />} label="Clusters (K)" value={result.n_clusters} />
        <MetricCard icon={<Database className="w-3.5 h-3.5" />} label="Samples" value={result.n_samples?.toLocaleString()} />
        <MetricCard icon={<Hash className="w-3.5 h-3.5" />} label="Features" value={result.n_features} />
        <MetricCard icon={<Timer className="w-3.5 h-3.5" />} label="Time" value={`${result.training_time_seconds?.toFixed(2)}s`} />
      </div>

      {/* Inertia */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-gray-900 mb-2">Inertia (WCSS)</h4>
        <p className="text-2xl font-bold text-gray-900">{result.inertia?.toFixed(1)}</p>
        <p className="text-xs text-gray-500 mt-1">
          Sum of squared distances from each point to its cluster center. Lower = tighter clusters.
        </p>
      </div>

      {/* Cluster Size Distribution */}
      {result.cluster_sizes && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-gray-900 mb-3">Cluster Size Distribution</h4>
          <svg viewBox="0 0 400 30" className="w-full">
            {result.cluster_sizes.map((size: number, i: number) => {
              const total = result.cluster_sizes.reduce((a: number, b: number) => a + b, 0);
              const pct = (size / total) * 100;
              const offset = result.cluster_sizes.slice(0, i).reduce((a: number, b: number) => a + b, 0) / total * 400;
              return (
                <g key={i}>
                  <rect
                    x={offset} y={2} width={Math.max(2, (pct / 100) * 400)} height={26}
                    fill={CLUSTER_COLORS[i % CLUSTER_COLORS.length]}
                    rx={i === 0 ? 4 : 0}
                    ry={i === 0 ? 4 : 0}
                  />
                  {pct > 8 && (
                    <text
                      x={offset + (pct / 100) * 400 / 2} y={19}
                      textAnchor="middle" fill="white" fontSize="10" fontWeight="600"
                    >
                      {size}
                    </text>
                  )}
                </g>
              );
            })}
          </svg>
          <div className="flex flex-wrap gap-3 mt-2">
            {result.cluster_sizes.map((size: number, i: number) => (
              <div key={i} className="flex items-center gap-1.5 text-xs text-gray-600">
                <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: CLUSTER_COLORS[i % CLUSTER_COLORS.length] }} />
                Cluster {i}: {size} ({((size / result.n_samples) * 100).toFixed(1)}%)
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Feature Importance */}
      {result.feature_importance?.features?.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-gray-900 mb-3">Feature Importance for Clustering</h4>
          <div className="space-y-2">
            {result.feature_importance.features.slice(0, 8).map((f: any) => (
              <div key={f.feature} className="flex items-center gap-3">
                <span className="text-xs text-gray-600 w-28 truncate" title={f.feature}>{f.feature}</span>
                <div className="flex-1 h-3 rounded-full bg-gray-100 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-cyan-500 transition-all"
                    style={{ width: `${f.bar_width_pct}%` }}
                  />
                </div>
                <span className="text-xs text-gray-500 w-12 text-right">{(f.importance * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function MetricCard({ icon, label, value }: { icon: React.ReactNode; label: string; value: any }) {
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <div className="flex items-center gap-2 text-xs text-gray-500 uppercase tracking-wide mb-1">
        {icon} {label}
      </div>
      <div className="text-lg font-semibold text-gray-900">{value}</div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════
// TAB 2: CLUSTER MAP (PCA 2D Scatter)
// ═══════════════════════════════════════════════════════════════════

function ClusterMapTab({ result }: { result: any }) {
  const [hoveredCluster, setHoveredCluster] = useState<number | null>(null);
  const pca = result.pca_projection;
  if (!pca?.points?.length) return null;

  const W = 600, H = 420, PAD = 50;

  const { scaleX, scaleY, minX, maxX, minY, maxY } = useMemo(() => {
    let mnX = Infinity, mxX = -Infinity, mnY = Infinity, mxY = -Infinity;
    for (const p of pca.points) {
      if (p.x < mnX) mnX = p.x;
      if (p.x > mxX) mxX = p.x;
      if (p.y < mnY) mnY = p.y;
      if (p.y > mxY) mxY = p.y;
    }
    const padX = (mxX - mnX) * 0.05 || 1;
    const padY = (mxY - mnY) * 0.05 || 1;
    mnX -= padX; mxX += padX; mnY -= padY; mxY += padY;
    return {
      scaleX: (v: number) => PAD + ((v - mnX) / (mxX - mnX)) * (W - 2 * PAD),
      scaleY: (v: number) => (H - PAD) - ((v - mnY) / (mxY - mnY)) * (H - 2 * PAD),
      minX: mnX, maxX: mxX, minY: mnY, maxY: mxY,
    };
  }, [pca.points]);

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-cyan-200 bg-cyan-50 px-4 py-3">
        <Info className="mt-0.5 h-5 w-5 shrink-0 text-cyan-600" />
        <p className="text-sm text-cyan-800">
          <span className="font-semibold">PCA 2D Projection</span> — your high-dimensional data projected onto 2 principal components.
          Each dot is a data point, colored by cluster. Stars mark cluster centroids.
        </p>
      </div>

      <div className="bg-white border border-gray-200 rounded-lg p-4 overflow-x-auto">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 500 }}>
          {/* Grid lines */}
          {[0.25, 0.5, 0.75].map((frac) => (
            <g key={frac}>
              <line x1={PAD} y1={PAD + frac * (H - 2 * PAD)} x2={W - PAD} y2={PAD + frac * (H - 2 * PAD)} stroke="#f3f4f6" strokeWidth="1" />
              <line x1={PAD + frac * (W - 2 * PAD)} y1={PAD} x2={PAD + frac * (W - 2 * PAD)} y2={H - PAD} stroke="#f3f4f6" strokeWidth="1" />
            </g>
          ))}

          {/* Axes */}
          <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="#d1d5db" strokeWidth="1" />
          <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="#d1d5db" strokeWidth="1" />
          <text x={W / 2} y={H - 8} textAnchor="middle" fontSize="11" fill="#6b7280">{pca.pc1_label || "PC1"}</text>
          <text x={12} y={H / 2} textAnchor="middle" fontSize="11" fill="#6b7280" transform={`rotate(-90,12,${H / 2})`}>{pca.pc2_label || "PC2"}</text>

          {/* Data points */}
          {pca.points.map((p: any, i: number) => (
            <circle
              key={i}
              cx={scaleX(p.x)} cy={scaleY(p.y)} r={3}
              fill={CLUSTER_COLORS[p.cluster % CLUSTER_COLORS.length]}
              opacity={hoveredCluster === null || hoveredCluster === p.cluster ? 0.7 : 0.1}
              onMouseEnter={() => setHoveredCluster(p.cluster)}
              onMouseLeave={() => setHoveredCluster(null)}
            />
          ))}

          {/* Centroids */}
          {pca.centroids_2d?.map((c: number[], i: number) => {
            const cx = scaleX(c[0]);
            const cy = scaleY(c[1] ?? 0);
            const size = 8;
            return (
              <g key={`c-${i}`}>
                {/* Diamond shape */}
                <polygon
                  points={`${cx},${cy - size} ${cx + size},${cy} ${cx},${cy + size} ${cx - size},${cy}`}
                  fill={CLUSTER_COLORS[i % CLUSTER_COLORS.length]}
                  stroke="white" strokeWidth="2"
                  opacity={hoveredCluster === null || hoveredCluster === i ? 1 : 0.3}
                />
                <text
                  x={cx} y={cy - size - 4}
                  textAnchor="middle" fontSize="10" fontWeight="600"
                  fill={CLUSTER_COLORS[i % CLUSTER_COLORS.length]}
                >
                  C{i}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-4">
        {result.cluster_sizes?.map((size: number, i: number) => (
          <button
            key={i}
            className={`flex items-center gap-2 text-xs px-3 py-1.5 rounded-full border transition-all ${
              hoveredCluster === i ? "border-gray-400 bg-gray-50" : "border-gray-200"
            }`}
            onMouseEnter={() => setHoveredCluster(i)}
            onMouseLeave={() => setHoveredCluster(null)}
          >
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: CLUSTER_COLORS[i % CLUSTER_COLORS.length] }} />
            <span className="font-medium">Cluster {i}</span>
            <span className="text-gray-400">({size} pts)</span>
          </button>
        ))}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════
// TAB 3: ELBOW METHOD
// ═══════════════════════════════════════════════════════════════════

function ElbowTab({ result }: { result: any }) {
  const [selectedK, setSelectedK] = useState<number | null>(null);
  const [showSilhouette, setShowSilhouette] = useState(true);
  const elbow = result.elbow_data as { k: number; inertia: number; silhouette: number }[];
  if (!elbow?.length) return null;

  const W = 600, H = 350, PAD = 60, RPAD = 60;

  const { elbowK, scales } = useMemo(() => {
    const ks = elbow.map((d) => d.k);
    const inertias = elbow.map((d) => d.inertia);
    const sils = elbow.map((d) => d.silhouette);
    const minK = Math.min(...ks), maxK = Math.max(...ks);
    const maxIn = Math.max(...inertias);
    const minSil = Math.min(...sils), maxSil = Math.max(...sils);

    // Detect elbow: biggest drop in inertia change rate
    let bestDrop = 0, bestK = ks[0];
    for (let i = 1; i < inertias.length - 1; i++) {
      const drop = (inertias[i - 1] - inertias[i]) - (inertias[i] - inertias[i + 1]);
      if (drop > bestDrop) { bestDrop = drop; bestK = ks[i]; }
    }

    return {
      elbowK: bestK,
      scales: {
        x: (k: number) => PAD + ((k - minK) / (maxK - minK || 1)) * (W - PAD - RPAD),
        yIn: (v: number) => (H - PAD) - (v / (maxIn || 1)) * (H - 2 * PAD),
        ySil: (v: number) => (H - PAD) - ((v - minSil) / ((maxSil - minSil) || 1)) * (H - 2 * PAD),
        minK, maxK, maxIn, minSil, maxSil,
      },
    };
  }, [elbow]);

  const hovered = elbow.find((d) => d.k === selectedK);

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-cyan-200 bg-cyan-50 px-4 py-3">
        <Info className="mt-0.5 h-5 w-5 shrink-0 text-cyan-600" />
        <p className="text-sm text-cyan-800">
          <span className="font-semibold">Elbow Method</span> — find the optimal K by looking for the "bend" where
          adding more clusters stops giving significant improvement.
          {elbowK && <> Suggested K = <strong>{elbowK}</strong>.</>}
        </p>
      </div>

      <div className="flex gap-2 mb-2">
        <button
          onClick={() => setShowSilhouette(false)}
          className={`px-3 py-1 text-xs rounded-full border transition-colors ${
            !showSilhouette ? "bg-blue-100 border-blue-300 text-blue-700" : "border-gray-200 text-gray-500"
          }`}
        >
          Inertia Only
        </button>
        <button
          onClick={() => setShowSilhouette(true)}
          className={`px-3 py-1 text-xs rounded-full border transition-colors ${
            showSilhouette ? "bg-cyan-100 border-cyan-300 text-cyan-700" : "border-gray-200 text-gray-500"
          }`}
        >
          + Silhouette Score
        </button>
      </div>

      <div className="bg-white border border-gray-200 rounded-lg p-4 overflow-x-auto">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 420 }}>
          {/* Axes */}
          <line x1={PAD} y1={H - PAD} x2={W - RPAD} y2={H - PAD} stroke="#d1d5db" strokeWidth="1" />
          <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="#3b82f6" strokeWidth="1" />
          {showSilhouette && <line x1={W - RPAD} y1={PAD} x2={W - RPAD} y2={H - PAD} stroke="#14b8a6" strokeWidth="1" />}

          {/* Labels */}
          <text x={W / 2} y={H - 8} textAnchor="middle" fontSize="11" fill="#6b7280">Number of Clusters (K)</text>
          <text x={14} y={H / 2} textAnchor="middle" fontSize="10" fill="#3b82f6" transform={`rotate(-90,14,${H / 2})`}>Inertia (WCSS)</text>
          {showSilhouette && (
            <text x={W - 14} y={H / 2} textAnchor="middle" fontSize="10" fill="#14b8a6" transform={`rotate(90,${W - 14},${H / 2})`}>Silhouette Score</text>
          )}

          {/* X tick marks */}
          {elbow.map((d) => (
            <g key={`tick-${d.k}`}>
              <line x1={scales.x(d.k)} y1={H - PAD} x2={scales.x(d.k)} y2={H - PAD + 4} stroke="#9ca3af" />
              <text x={scales.x(d.k)} y={H - PAD + 16} textAnchor="middle" fontSize="10" fill="#6b7280">{d.k}</text>
            </g>
          ))}

          {/* Elbow point vertical line */}
          {elbowK && (
            <line
              x1={scales.x(elbowK)} y1={PAD} x2={scales.x(elbowK)} y2={H - PAD}
              stroke="#f59e0b" strokeWidth="1.5" strokeDasharray="6 3" opacity="0.6"
            />
          )}

          {/* Inertia line */}
          <polyline
            points={elbow.map((d) => `${scales.x(d.k)},${scales.yIn(d.inertia)}`).join(" ")}
            fill="none" stroke="#3b82f6" strokeWidth="2.5" strokeLinejoin="round"
          />

          {/* Silhouette line */}
          {showSilhouette && (
            <polyline
              points={elbow.map((d) => `${scales.x(d.k)},${scales.ySil(d.silhouette)}`).join(" ")}
              fill="none" stroke="#14b8a6" strokeWidth="2.5" strokeLinejoin="round"
            />
          )}

          {/* Dots (interactive) */}
          {elbow.map((d) => (
            <g key={`dot-${d.k}`}>
              <circle
                cx={scales.x(d.k)} cy={scales.yIn(d.inertia)} r={d.k === result.n_clusters ? 6 : selectedK === d.k ? 5 : 4}
                fill={d.k === result.n_clusters ? "#f59e0b" : "#3b82f6"}
                stroke="white" strokeWidth="2"
                className="cursor-pointer"
                onMouseEnter={() => setSelectedK(d.k)}
                onMouseLeave={() => setSelectedK(null)}
              />
              {showSilhouette && (
                <circle
                  cx={scales.x(d.k)} cy={scales.ySil(d.silhouette)} r={d.k === result.n_clusters ? 6 : selectedK === d.k ? 5 : 4}
                  fill={d.k === result.n_clusters ? "#f59e0b" : "#14b8a6"}
                  stroke="white" strokeWidth="2"
                  className="cursor-pointer"
                  onMouseEnter={() => setSelectedK(d.k)}
                  onMouseLeave={() => setSelectedK(null)}
                />
              )}
            </g>
          ))}

          {/* Current K label */}
          {result.n_clusters && (
            <text
              x={scales.x(result.n_clusters)} y={PAD - 6}
              textAnchor="middle" fontSize="10" fontWeight="700" fill="#f59e0b"
            >
              Current K={result.n_clusters}
            </text>
          )}

          {/* Elbow label */}
          {elbowK && elbowK !== result.n_clusters && (
            <text
              x={scales.x(elbowK)} y={PAD - 6}
              textAnchor="middle" fontSize="10" fontWeight="600" fill="#f59e0b"
            >
              Elbow K={elbowK}
            </text>
          )}

          {/* Hover tooltip */}
          {hovered && (
            <g>
              <rect
                x={scales.x(hovered.k) - 55} y={scales.yIn(hovered.inertia) - 50}
                width={110} height={showSilhouette ? 44 : 30} rx="4" fill="white" stroke="#e5e7eb" strokeWidth="1"
              />
              <text x={scales.x(hovered.k)} y={scales.yIn(hovered.inertia) - 35} textAnchor="middle" fontSize="10" fontWeight="600" fill="#111827">
                K={hovered.k} | Inertia={hovered.inertia.toFixed(0)}
              </text>
              {showSilhouette && (
                <text x={scales.x(hovered.k)} y={scales.yIn(hovered.inertia) - 20} textAnchor="middle" fontSize="10" fill="#14b8a6">
                  Silhouette={hovered.silhouette.toFixed(3)}
                </text>
              )}
            </g>
          )}
        </svg>
      </div>

      {/* Legend */}
      <div className="flex gap-4 text-xs text-gray-600">
        <div className="flex items-center gap-1.5">
          <div className="w-4 h-0.5 bg-blue-500" /> Inertia (lower = tighter)
        </div>
        {showSilhouette && (
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-0.5 bg-teal-500" /> Silhouette (higher = better separated)
          </div>
        )}
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full bg-amber-500" /> Current K / Elbow
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════
// TAB 4: CLUSTER PROFILES
// ═══════════════════════════════════════════════════════════════════

function ProfilesTab({ result }: { result: any }) {
  const profiles = result.cluster_profiles;
  const [selectedCluster, setSelectedCluster] = useState<number | null>(null);
  const [viewMode, setViewMode] = useState<"bar" | "radar">("bar");

  if (!profiles?.clusters?.length) return null;

  const features = result.feature_names as string[];
  const nClusters = profiles.clusters.length;

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-cyan-200 bg-cyan-50 px-4 py-3">
        <Info className="mt-0.5 h-5 w-5 shrink-0 text-cyan-600" />
        <p className="text-sm text-cyan-800">
          <span className="font-semibold">Cluster Profiles</span> — see what makes each cluster unique by comparing
          feature averages across clusters.
        </p>
      </div>

      {/* Controls */}
      <div className="flex gap-2 flex-wrap">
        <button
          onClick={() => setSelectedCluster(null)}
          className={`px-3 py-1 text-xs rounded-full border transition-colors ${
            selectedCluster === null ? "bg-gray-800 text-white border-gray-800" : "border-gray-200 text-gray-600"
          }`}
        >
          All Clusters
        </button>
        {profiles.clusters.map((c: any) => (
          <button
            key={c.cluster_id}
            onClick={() => setSelectedCluster(selectedCluster === c.cluster_id ? null : c.cluster_id)}
            className={`px-3 py-1 text-xs rounded-full border transition-colors ${
              selectedCluster === c.cluster_id ? "text-white border-transparent" : "border-gray-200 text-gray-600"
            }`}
            style={selectedCluster === c.cluster_id ? { backgroundColor: CLUSTER_COLORS[c.cluster_id % CLUSTER_COLORS.length] } : {}}
          >
            Cluster {c.cluster_id} ({c.size})
          </button>
        ))}
        <div className="ml-auto flex gap-1">
          <button
            onClick={() => setViewMode("bar")}
            className={`px-2 py-1 text-xs rounded border ${viewMode === "bar" ? "bg-cyan-100 border-cyan-300" : "border-gray-200"}`}
          >
            Bars
          </button>
          <button
            onClick={() => setViewMode("radar")}
            className={`px-2 py-1 text-xs rounded border ${viewMode === "radar" ? "bg-cyan-100 border-cyan-300" : "border-gray-200"}`}
          >
            Radar
          </button>
        </div>
      </div>

      {/* Bar Chart View */}
      {viewMode === "bar" && (
        <ProfileBarChart
          profiles={profiles} features={features}
          selectedCluster={selectedCluster} nClusters={nClusters}
        />
      )}

      {/* Radar View */}
      {viewMode === "radar" && (
        <ProfileRadarChart
          profiles={profiles} features={features}
          selectedCluster={selectedCluster} nClusters={nClusters}
        />
      )}

      {/* Distinguishing Features */}
      {profiles.clusters.map((c: any) => {
        if (selectedCluster !== null && selectedCluster !== c.cluster_id) return null;
        return (
          <div key={c.cluster_id} className="bg-white border border-gray-200 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-3">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: CLUSTER_COLORS[c.cluster_id % CLUSTER_COLORS.length] }} />
              <h4 className="text-sm font-semibold text-gray-900">Cluster {c.cluster_id} — What makes it unique</h4>
              <span className="text-xs text-gray-400">({c.size} points)</span>
            </div>
            <div className="space-y-2">
              {c.distinguishing_features?.map((df: any) => (
                <div key={df.feature} className="flex items-center gap-3 text-sm">
                  <span className={`text-xs font-medium px-2 py-0.5 rounded ${
                    df.direction === "higher" ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"
                  }`}>
                    {df.direction === "higher" ? "+" : "-"}{Math.abs(df.deviation).toFixed(1)}σ
                  </span>
                  <span className="text-gray-700 font-medium">{df.feature}</span>
                  <span className="text-gray-400 text-xs">
                    cluster avg: {df.cluster_mean} vs global: {df.global_mean}
                  </span>
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function ProfileBarChart({ profiles, features, selectedCluster, nClusters }: {
  profiles: any; features: string[]; selectedCluster: number | null; nClusters: number;
}) {
  const displayFeatures = features.slice(0, 10); // Cap at 10 features for readability
  const W = 600, H = 300, PAD = 60, BPAD = 80;
  const chartW = W - PAD - 20;
  const chartH = H - BPAD - 20;
  const groupW = chartW / displayFeatures.length;
  const barW = Math.min(groupW / (nClusters + 1), 20);

  // Find min/max normalized means
  const allVals = profiles.clusters.flatMap((c: any) =>
    displayFeatures.map((f: string) => c.normalized_means?.[f] ?? 0)
  );
  const minV = Math.min(0, ...allVals);
  const maxV = Math.max(0, ...allVals);
  const range = maxV - minV || 1;
  const yScale = (v: number) => 20 + chartH - ((v - minV) / range) * chartH;
  const zeroY = yScale(0);

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 overflow-x-auto">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 380 }}>
        {/* Zero line */}
        <line x1={PAD} y1={zeroY} x2={W - 20} y2={zeroY} stroke="#d1d5db" strokeWidth="1" strokeDasharray="4 2" />

        {/* Bars */}
        {displayFeatures.map((feat, fi) => {
          const gx = PAD + fi * groupW;
          return (
            <g key={feat}>
              {profiles.clusters.map((c: any) => {
                if (selectedCluster !== null && selectedCluster !== c.cluster_id) return null;
                const val = c.normalized_means?.[feat] ?? 0;
                const y = yScale(val);
                const barHeight = Math.abs(y - zeroY);
                const color = CLUSTER_COLORS[c.cluster_id % CLUSTER_COLORS.length];
                const bx = gx + (c.cluster_id + 0.5) * (groupW / nClusters) - barW / 2;
                return (
                  <rect
                    key={c.cluster_id}
                    x={bx}
                    y={val >= 0 ? y : zeroY}
                    width={barW}
                    height={Math.max(1, barHeight)}
                    fill={color}
                    opacity={0.8}
                    rx={2}
                  />
                );
              })}
              {/* Feature label */}
              <text
                x={gx + groupW / 2} y={H - 10}
                textAnchor="middle" fontSize="9" fill="#6b7280"
                transform={`rotate(-30, ${gx + groupW / 2}, ${H - 10})`}
              >
                {feat.length > 12 ? feat.slice(0, 12) + "…" : feat}
              </text>
            </g>
          );
        })}

        {/* Y axis label */}
        <text x={10} y={H / 2} textAnchor="middle" fontSize="10" fill="#6b7280" transform={`rotate(-90,10,${H / 2})`}>Normalized Mean</text>
      </svg>
    </div>
  );
}

function ProfileRadarChart({ profiles, features, selectedCluster, nClusters }: {
  profiles: any; features: string[]; selectedCluster: number | null; nClusters: number;
}) {
  const displayFeatures = features.slice(0, 8); // Cap at 8 for radar readability
  const n = displayFeatures.length;
  if (n < 3) return <p className="text-sm text-gray-500">Need at least 3 features for radar chart.</p>;

  const CX = 200, CY = 200, R = 140;

  // Normalize values to 0-1 range across all clusters
  const allVals = profiles.clusters.flatMap((c: any) =>
    displayFeatures.map((f: string) => c.normalized_means?.[f] ?? 0)
  );
  const minV = Math.min(...allVals);
  const maxV = Math.max(...allVals);
  const range = maxV - minV || 1;
  const norm = (v: number) => (v - minV) / range;

  const angleStep = (2 * Math.PI) / n;
  const getPoint = (i: number, r: number) => ({
    x: CX + r * Math.cos(i * angleStep - Math.PI / 2),
    y: CY + r * Math.sin(i * angleStep - Math.PI / 2),
  });

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 overflow-x-auto">
      <svg viewBox="0 0 400 400" className="w-full" style={{ maxHeight: 400 }}>
        {/* Grid rings */}
        {[0.25, 0.5, 0.75, 1].map((frac) => (
          <polygon
            key={frac}
            points={displayFeatures.map((_, i) => {
              const p = getPoint(i, R * frac);
              return `${p.x},${p.y}`;
            }).join(" ")}
            fill="none" stroke="#e5e7eb" strokeWidth="1"
          />
        ))}

        {/* Axis lines + labels */}
        {displayFeatures.map((feat, i) => {
          const p = getPoint(i, R);
          const lp = getPoint(i, R + 18);
          return (
            <g key={feat}>
              <line x1={CX} y1={CY} x2={p.x} y2={p.y} stroke="#e5e7eb" strokeWidth="1" />
              <text x={lp.x} y={lp.y} textAnchor="middle" fontSize="9" fill="#6b7280" dominantBaseline="middle">
                {feat.length > 10 ? feat.slice(0, 10) + "…" : feat}
              </text>
            </g>
          );
        })}

        {/* Cluster polygons */}
        {profiles.clusters.map((c: any) => {
          if (selectedCluster !== null && selectedCluster !== c.cluster_id) return null;
          const color = CLUSTER_COLORS[c.cluster_id % CLUSTER_COLORS.length];
          const pts = displayFeatures.map((feat, i) => {
            const val = norm(c.normalized_means?.[feat] ?? 0);
            const p = getPoint(i, R * val);
            return `${p.x},${p.y}`;
          }).join(" ");
          return (
            <g key={c.cluster_id}>
              <polygon points={pts} fill={color} fillOpacity="0.15" stroke={color} strokeWidth="2" />
              {displayFeatures.map((feat, i) => {
                const val = norm(c.normalized_means?.[feat] ?? 0);
                const p = getPoint(i, R * val);
                return <circle key={i} cx={p.x} cy={p.y} r={3} fill={color} />;
              })}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════
// TAB 5: CONVERGENCE ANIMATION
// ═══════════════════════════════════════════════════════════════════

function ConvergenceTab({ result }: { result: any }) {
  const conv = result.convergence_history;
  const pca = result.pca_projection;
  const hasData = !!(conv?.pca_iterations?.length && pca?.points?.length);

  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(600);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const maxFrame = hasData ? conv.pca_iterations.length - 1 : 0;

  useEffect(() => {
    if (playing && hasData) {
      intervalRef.current = setInterval(() => {
        setFrame((f) => {
          if (f >= maxFrame) { setPlaying(false); return maxFrame; }
          return f + 1;
        });
      }, speed);
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [playing, speed, maxFrame, hasData]);

  const W = 420, H = 320, PAD = 30;

  const scales = useMemo(() => {
    if (!hasData) return { x: (v: number) => v, y: (v: number) => v };
    let mnX = Infinity, mxX = -Infinity, mnY = Infinity, mxY = -Infinity;
    for (const p of pca.points) {
      if (p.x < mnX) mnX = p.x; if (p.x > mxX) mxX = p.x;
      if (p.y < mnY) mnY = p.y; if (p.y > mxY) mxY = p.y;
    }
    const padX = (mxX - mnX) * 0.05 || 1;
    const padY = (mxY - mnY) * 0.05 || 1;
    mnX -= padX; mxX += padX; mnY -= padY; mxY += padY;
    return {
      x: (v: number) => PAD + ((v - mnX) / (mxX - mnX)) * (W - 2 * PAD),
      y: (v: number) => (H - PAD) - ((v - mnY) / (mxY - mnY)) * (H - 2 * PAD),
    };
  }, [pca?.points, hasData]);

  // Compute point assignments at current frame (nearest centroid)
  const frameAssignments = useMemo(() => {
    if (!hasData) return [];
    const centroids = conv.pca_iterations[frame] || [];
    if (!centroids.length) return [];
    return pca.points.map((p: any) => {
      let best = 0, bestD = Infinity;
      centroids.forEach((c: number[], ci: number) => {
        const d = (p.x - c[0]) ** 2 + (p.y - (c[1] ?? 0)) ** 2;
        if (d < bestD) { bestD = d; best = ci; }
      });
      return best;
    });
  }, [hasData, conv?.pca_iterations, frame, pca?.points]);

  // WCSS chart scales
  const wcssValues = conv?.iterations?.map((it: any) => it.wcss) || [];
  const maxWcss = Math.max(...(wcssValues.length ? wcssValues : [1])) || 1;
  const CW = 200, CH = 120;

  const currentCentroids = hasData ? (conv.pca_iterations[frame] || []) : [];

  if (!hasData) {
    return <p className="text-sm text-gray-500 py-8 text-center">No convergence data available.</p>;
  }

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-cyan-200 bg-cyan-50 px-4 py-3">
        <Info className="mt-0.5 h-5 w-5 shrink-0 text-cyan-600" />
        <p className="text-sm text-cyan-800">
          <span className="font-semibold">Convergence Visualization</span> — watch how centroids move and points get reassigned at each iteration.
          Converged in <strong>{conv.converged_at}</strong> iterations.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Left: Scatter with centroid trails */}
        <div className="lg:col-span-2 bg-white border border-gray-200 rounded-lg p-3">
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 400 }}>
            {/* Grid lines */}
            {[0.25, 0.5, 0.75].map((frac) => (
              <g key={frac}>
                <line x1={PAD} y1={PAD + frac * (H - 2 * PAD)} x2={W - PAD} y2={PAD + frac * (H - 2 * PAD)} stroke="#f3f4f6" strokeWidth="0.5" />
                <line x1={PAD + frac * (W - 2 * PAD)} y1={PAD} x2={PAD + frac * (W - 2 * PAD)} y2={H - PAD} stroke="#f3f4f6" strokeWidth="0.5" />
              </g>
            ))}
            <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="#d1d5db" strokeWidth="1" />
            <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="#d1d5db" strokeWidth="1" />

            {/* Data points — colored by current frame's nearest centroid */}
            {pca.points.map((p: any, i: number) => {
              const clusterIdx = frameAssignments[i] ?? p.cluster;
              return (
                <circle
                  key={i} cx={scales.x(p.x)} cy={scales.y(p.y)} r={2.5}
                  fill={CLUSTER_COLORS[clusterIdx % CLUSTER_COLORS.length]}
                  opacity={0.55}
                  style={{ transition: "fill 0.3s ease" }}
                />
              );
            })}

            {/* Centroid trails (path from iteration 0 to current frame) */}
            {currentCentroids.map((_: any, ci: number) => {
              const color = CLUSTER_COLORS[ci % CLUSTER_COLORS.length];
              const trail = conv.pca_iterations.slice(0, frame + 1).map((iter: number[][]) => iter[ci]);
              if (trail.length < 2) return null;
              const pathD = trail.map((pt: number[], i: number) =>
                `${i === 0 ? "M" : "L"}${scales.x(pt[0])},${scales.y(pt[1] ?? 0)}`
              ).join(" ");
              return (
                <g key={`trail-${ci}`}>
                  <path d={pathD} fill="none" stroke={color} strokeWidth="2" strokeDasharray="5 3" opacity="0.5" />
                  {/* Ghost centroids along trail */}
                  {trail.slice(0, -1).map((pt: number[], ti: number) => (
                    <circle
                      key={ti} cx={scales.x(pt[0])} cy={scales.y(pt[1] ?? 0)} r={4}
                      fill={color} opacity={0.15 + (ti / trail.length) * 0.2}
                      stroke={color} strokeWidth="0.5" strokeOpacity={0.3}
                    />
                  ))}
                  {/* Arrow from initial position */}
                  {trail.length >= 2 && (
                    <circle
                      cx={scales.x(trail[0][0])} cy={scales.y(trail[0][1] ?? 0)}
                      r={5} fill="none" stroke={color} strokeWidth="1.5" strokeDasharray="2 2" opacity={0.4}
                    />
                  )}
                </g>
              );
            })}

            {/* Current centroids — large, prominent */}
            {currentCentroids.map((c: number[], ci: number) => {
              const color = CLUSTER_COLORS[ci % CLUSTER_COLORS.length];
              const cx = scales.x(c[0]);
              const cy = scales.y(c[1] ?? 0);
              return (
                <g key={`cur-${ci}`}>
                  {/* Glow ring */}
                  <circle cx={cx} cy={cy} r={13} fill={color} opacity={0.12} />
                  {/* Main centroid */}
                  <circle cx={cx} cy={cy} r={8} fill={color} stroke="white" strokeWidth="2.5" />
                  <text x={cx} y={cy + 3.5} textAnchor="middle" fontSize="8" fontWeight="700" fill="white">{ci}</text>
                </g>
              );
            })}

            {/* Iteration label */}
            <text x={W - PAD} y={16} textAnchor="end" fontSize="11" fill="#6b7280" fontWeight="600">
              Iteration {frame + 1} / {maxFrame + 1}
            </text>
          </svg>
        </div>

        {/* Right panel: WCSS curve + cluster sizes */}
        <div className="space-y-3">
          <div className="bg-white border border-gray-200 rounded-lg p-3">
            <h4 className="text-xs font-semibold text-gray-700 mb-2">WCSS over Iterations</h4>
            <svg viewBox={`0 0 ${CW} ${CH}`} className="w-full">
              <line x1={20} y1={CH - 20} x2={CW - 10} y2={CH - 20} stroke="#d1d5db" strokeWidth="0.5" />
              {wcssValues.length > 1 && (
                <>
                  {/* Full curve (dimmed) */}
                  <polyline
                    points={wcssValues.map((v: number, i: number) =>
                      `${20 + (i / (wcssValues.length - 1)) * (CW - 30)},${(CH - 25) - (v / maxWcss) * (CH - 35)}`
                    ).join(" ")}
                    fill="none" stroke="#e5e7eb" strokeWidth="1.5"
                  />
                  {/* Progress curve up to current frame */}
                  <polyline
                    points={wcssValues.slice(0, frame + 1).map((v: number, i: number) =>
                      `${20 + (i / (wcssValues.length - 1)) * (CW - 30)},${(CH - 25) - (v / maxWcss) * (CH - 35)}`
                    ).join(" ")}
                    fill="none" stroke="#06b6d4" strokeWidth="2"
                  />
                </>
              )}
              {/* Current position marker */}
              {frame < wcssValues.length && (
                <circle
                  cx={20 + (frame / Math.max(wcssValues.length - 1, 1)) * (CW - 30)}
                  cy={(CH - 25) - (wcssValues[frame] / maxWcss) * (CH - 35)}
                  r={4} fill="#f59e0b" stroke="white" strokeWidth="1.5"
                />
              )}
              <text x={CW / 2} y={CH - 4} textAnchor="middle" fontSize="8" fill="#9ca3af">Iteration</text>
            </svg>
            <div className="text-center mt-1">
              <span className="text-xs text-gray-500">
                WCSS: <strong className="text-gray-900">{wcssValues[frame]?.toFixed(1) ?? "—"}</strong>
              </span>
            </div>
          </div>

          {/* Cluster sizes at current frame */}
          {frameAssignments.length > 0 && (
            <div className="bg-white border border-gray-200 rounded-lg p-3">
              <h4 className="text-[10px] font-semibold text-gray-400 uppercase mb-2">Cluster Sizes (iter {frame + 1})</h4>
              {Array.from({ length: result.n_clusters }, (_, ci) => {
                const count = frameAssignments.filter((a: number) => a === ci).length;
                const pct = (count / pca.points.length) * 100;
                return (
                  <div key={ci} className="flex items-center gap-2 text-xs mb-1">
                    <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: CLUSTER_COLORS[ci % CLUSTER_COLORS.length] }} />
                    <span className="text-gray-600 w-6">C{ci}</span>
                    <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                      <div className="h-full rounded-full transition-all duration-300" style={{ width: `${pct}%`, backgroundColor: CLUSTER_COLORS[ci % CLUSTER_COLORS.length] }} />
                    </div>
                    <span className="text-gray-500 w-6 text-right">{count}</span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Playback Controls */}
      <div className="flex items-center justify-center gap-3 bg-gray-50 rounded-lg p-3">
        <button
          onClick={() => setFrame((f) => Math.max(0, f - 1))}
          className="p-1.5 rounded-lg hover:bg-gray-200 transition-colors"
          disabled={frame === 0}
        >
          <SkipBack className="w-4 h-4 text-gray-600" />
        </button>
        <button
          onClick={() => {
            if (frame >= maxFrame) { setFrame(0); setPlaying(true); }
            else setPlaying(!playing);
          }}
          className="p-2 rounded-full bg-cyan-500 text-white hover:bg-cyan-600 transition-colors"
        >
          {playing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
        </button>
        <button
          onClick={() => setFrame((f) => Math.min(maxFrame, f + 1))}
          className="p-1.5 rounded-lg hover:bg-gray-200 transition-colors"
          disabled={frame >= maxFrame}
        >
          <SkipForward className="w-4 h-4 text-gray-600" />
        </button>
        <button
          onClick={() => { setFrame(0); setPlaying(false); }}
          className="p-1.5 rounded-lg hover:bg-gray-200 transition-colors"
        >
          <RotateCcw className="w-4 h-4 text-gray-600" />
        </button>

        {/* Frame slider */}
        <input
          type="range" min={0} max={maxFrame} value={frame}
          onChange={(e) => { setFrame(Number(e.target.value)); setPlaying(false); }}
          className="flex-1 max-w-48 accent-cyan-500"
        />
        <span className="text-xs text-gray-500 w-20 text-center">
          {frame + 1} / {maxFrame + 1}
        </span>

        {/* Speed */}
        <select
          value={speed}
          onChange={(e) => setSpeed(Number(e.target.value))}
          className="text-xs border border-gray-200 rounded px-2 py-1"
        >
          <option value={1000}>0.5x</option>
          <option value={600}>1x</option>
          <option value={300}>2x</option>
          <option value={150}>4x</option>
        </select>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════
// TAB 6: QUIZ
// ═══════════════════════════════════════════════════════════════════

function QuizTab({ questions }: { questions: any[] }) {
  const [currentQ, setCurrentQ] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [answered, setAnswered] = useState(false);
  const [score, setScore] = useState(0);
  const [answers, setAnswers] = useState<(number | null)[]>([]);
  const [showResults, setShowResults] = useState(false);

  if (!questions.length) return null;

  const question = questions[currentQ];
  const isCorrect = selectedAnswer === question.correct_answer;

  const handleSelect = (idx: number) => {
    if (answered) return;
    setSelectedAnswer(idx);
    setAnswered(true);
    if (idx === question.correct_answer) setScore((s) => s + 1);
    setAnswers((a) => [...a, idx]);
  };

  const handleNext = () => {
    if (currentQ + 1 >= questions.length) {
      setShowResults(true);
    } else {
      setCurrentQ((q) => q + 1);
      setSelectedAnswer(null);
      setAnswered(false);
    }
  };

  const handleRetry = () => {
    setCurrentQ(0);
    setSelectedAnswer(null);
    setAnswered(false);
    setScore(0);
    setAnswers([]);
    setShowResults(false);
  };

  if (showResults) {
    const pct = Math.round((score / questions.length) * 100);
    return (
      <div className="space-y-6">
        <div className="text-center py-8">
          <Trophy
            className={`w-16 h-16 mx-auto mb-4 ${
              pct >= 80 ? "text-cyan-500" : pct >= 50 ? "text-yellow-500" : "text-red-500"
            }`}
          />
          <h3 className="text-2xl font-bold text-gray-900">
            {score} / {questions.length}
          </h3>
          <p className="text-gray-600 mt-1">
            {pct >= 80
              ? "Excellent! You understand K-Means clustering well!"
              : pct >= 50
              ? "Good job! Review the topics you missed."
              : "Keep learning! Clustering concepts take practice to master."}
          </p>
          <button
            onClick={handleRetry}
            className="mt-4 px-6 py-2 bg-cyan-600 text-white rounded-lg hover:bg-cyan-700 transition-colors text-sm font-medium"
          >
            Try Again
          </button>
        </div>
        <div className="space-y-3">
          <h4 className="text-sm font-semibold text-gray-700">Review</h4>
          {questions.map((q: any, i: number) => {
            const userAns = answers[i];
            const correct = userAns === q.correct_answer;
            return (
              <div key={i} className={`rounded-lg border p-3 ${correct ? "border-green-200 bg-green-50" : "border-red-200 bg-red-50"}`}>
                <div className="flex items-start gap-2">
                  {correct ? (
                    <CheckCircle className="w-4 h-4 text-green-600 mt-0.5 shrink-0" />
                  ) : (
                    <XCircle className="w-4 h-4 text-red-600 mt-0.5 shrink-0" />
                  )}
                  <div>
                    <p className="text-sm font-medium text-gray-900">{q.question}</p>
                    <p className="text-xs text-gray-600 mt-1">
                      Your answer: {q.options[userAns ?? 0]} {!correct && `| Correct: ${q.options[q.correct_answer]}`}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">{q.explanation}</p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 rounded-lg border border-cyan-200 bg-cyan-50 px-4 py-3">
        <HelpCircle className="mt-0.5 h-5 w-5 shrink-0 text-cyan-600" />
        <p className="text-sm text-cyan-800">
          <span className="font-semibold">Test your knowledge</span> about K-Means Clustering!
          Answer {questions.length} questions to check your understanding.
        </p>
      </div>

      {/* Progress dots */}
      <div className="flex items-center justify-center gap-2">
        {questions.map((_: any, i: number) => (
          <div
            key={i}
            className={`w-2.5 h-2.5 rounded-full transition-all ${
              i === currentQ
                ? "bg-cyan-500 scale-125"
                : i < answers.length
                ? answers[i] === questions[i].correct_answer
                  ? "bg-green-400"
                  : "bg-red-400"
                : "bg-gray-300"
            }`}
          />
        ))}
      </div>

      {/* Question */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <p className="text-xs text-gray-400 mb-2">
          Question {currentQ + 1} of {questions.length}
        </p>
        <p className="text-base font-medium text-gray-900 mb-4">{question.question}</p>

        <div className="space-y-2">
          {question.options.map((opt: string, idx: number) => {
            let style = "border-gray-200 bg-white hover:bg-gray-50 text-gray-700";
            if (answered) {
              if (idx === question.correct_answer) {
                style = "border-green-300 bg-green-50 text-green-800";
              } else if (idx === selectedAnswer && !isCorrect) {
                style = "border-red-300 bg-red-50 text-red-800";
              } else {
                style = "border-gray-200 bg-gray-50 text-gray-400";
              }
            } else if (idx === selectedAnswer) {
              style = "border-cyan-400 bg-cyan-50 text-cyan-700";
            }
            return (
              <button
                key={idx}
                onClick={() => handleSelect(idx)}
                disabled={answered}
                className={`w-full text-left px-4 py-3 rounded-lg border text-sm transition-colors ${style} disabled:cursor-default`}
              >
                <span className="font-medium mr-2">{String.fromCharCode(65 + idx)}.</span>
                {opt}
              </button>
            );
          })}
        </div>

        {answered && (
          <div className={`mt-4 p-3 rounded-lg text-sm ${isCorrect ? "bg-green-50 text-green-800" : "bg-red-50 text-red-800"}`}>
            <p className="font-semibold mb-1">{isCorrect ? "Correct!" : "Not quite."}</p>
            <p>{question.explanation}</p>
          </div>
        )}

        {answered && (
          <button
            onClick={handleNext}
            className="mt-4 px-4 py-2 bg-cyan-600 text-white rounded-lg hover:bg-cyan-700 transition-colors text-sm font-medium"
          >
            {currentQ + 1 >= questions.length ? "See Results" : "Next Question"}
          </button>
        )}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════
// TAB 7: HOW IT WORKS
// ═══════════════════════════════════════════════════════════════════

function HowItWorksTab({ result }: { result: any }) {
  const pca = result.pca_projection;
  const nK = result.n_clusters || 3;

  // Interactive state
  const [userCentroids, setUserCentroids] = useState<{ x: number; y: number }[]>([]);
  const [assignments, setAssignments] = useState<number[]>([]);
  const [phase, setPhase] = useState<"place" | "assigned" | "updated" | "converged">("place");
  const [iteration, setIteration] = useState(0);
  const [wcssHistory, setWcssHistory] = useState<number[]>([]);
  const [showAlgoResult, setShowAlgoResult] = useState(false);
  const [autoRunning, setAutoRunning] = useState(false);
  const autoRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [ripple, setRipple] = useState<{ x: number; y: number; id: number } | null>(null);
  // Trail history: array of centroid positions at each iteration
  const [centroidTrail, setCentroidTrail] = useState<{ x: number; y: number }[][]>([]);
  // Animation flash: "assigning" shows lines from points→centroids, "updating" shows centroid move arrows
  const [animFlash, setAnimFlash] = useState<"idle" | "assigning" | "updating">("idle");
  // Previous centroid positions (for showing movement arrows during update)
  const [prevCentroids, setPrevCentroids] = useState<{ x: number; y: number }[]>([]);

  const W = 560, H = 400, PAD = 30;

  // Scale PCA points to SVG coordinates
  const { points, scaleX, scaleY } = useMemo(() => {
    if (!pca?.points?.length) return { points: [], scaleX: (v: number) => v, scaleY: (v: number) => v };
    const pts = pca.points as { x: number; y: number; cluster: number }[];
    let mnX = Infinity, mxX = -Infinity, mnY = Infinity, mxY = -Infinity;
    for (const p of pts) {
      if (p.x < mnX) mnX = p.x; if (p.x > mxX) mxX = p.x;
      if (p.y < mnY) mnY = p.y; if (p.y > mxY) mxY = p.y;
    }
    const padX = (mxX - mnX) * 0.05 || 1;
    const padY = (mxY - mnY) * 0.05 || 1;
    mnX -= padX; mxX += padX; mnY -= padY; mxY += padY;
    return {
      points: pts,
      scaleX: (v: number) => PAD + ((v - mnX) / (mxX - mnX)) * (W - 2 * PAD),
      scaleY: (v: number) => (H - PAD) - ((v - mnY) / (mxY - mnY)) * (H - 2 * PAD),
    };
  }, [pca?.points]);

  // Inverse scale: SVG pixel → data coords
  const invScaleX = useCallback((px: number) => {
    if (!pca?.points?.length) return 0;
    const pts = pca.points as { x: number; y: number }[];
    let mnX = Infinity, mxX = -Infinity;
    for (const p of pts) { if (p.x < mnX) mnX = p.x; if (p.x > mxX) mxX = p.x; }
    const padX = (mxX - mnX) * 0.05 || 1;
    mnX -= padX; mxX += padX;
    return mnX + ((px - PAD) / (W - 2 * PAD)) * (mxX - mnX);
  }, [pca?.points]);

  const invScaleY = useCallback((py: number) => {
    if (!pca?.points?.length) return 0;
    const pts = pca.points as { x: number; y: number }[];
    let mnY = Infinity, mxY = -Infinity;
    for (const p of pts) { if (p.y < mnY) mnY = p.y; if (p.y > mxY) mxY = p.y; }
    const padY = (mxY - mnY) * 0.05 || 1;
    mnY -= padY; mxY += padY;
    return mnY + ((H - PAD - py) / (H - 2 * PAD)) * (mxY - mnY);
  }, [pca?.points]);

  // Euclidean distance
  const dist = (a: { x: number; y: number }, b: { x: number; y: number }) =>
    Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);

  // Assign each point to nearest centroid
  const doAssign = useCallback((centroids: { x: number; y: number }[]) => {
    const asgn = points.map((p) => {
      let best = 0, bestD = Infinity;
      centroids.forEach((c, ci) => {
        const d = dist(p, c);
        if (d < bestD) { bestD = d; best = ci; }
      });
      return best;
    });
    return asgn;
  }, [points]);

  // Compute WCSS
  const computeWcss = useCallback((centroids: { x: number; y: number }[], asgn: number[]) => {
    let wcss = 0;
    points.forEach((p, i) => {
      const c = centroids[asgn[i]];
      if (c) wcss += (p.x - c.x) ** 2 + (p.y - c.y) ** 2;
    });
    return wcss;
  }, [points]);

  // Update centroids to cluster means
  const doUpdate = useCallback((centroids: { x: number; y: number }[], asgn: number[]) => {
    const newC = centroids.map((c, ci) => {
      const clusterPts = points.filter((_, i) => asgn[i] === ci);
      if (clusterPts.length === 0) return c;
      return {
        x: clusterPts.reduce((s, p) => s + p.x, 0) / clusterPts.length,
        y: clusterPts.reduce((s, p) => s + p.y, 0) / clusterPts.length,
      };
    });
    return newC;
  }, [points]);

  // Handle SVG click to place centroid — use getScreenCTM for accurate viewBox mapping
  const handleSvgClick = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (phase !== "place" || userCentroids.length >= nK) return;
    const svg = e.currentTarget;
    const ctm = svg.getScreenCTM();
    if (!ctm) return;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const svgPt = pt.matrixTransform(ctm.inverse());
    const dataX = invScaleX(svgPt.x);
    const dataY = invScaleY(svgPt.y);
    setUserCentroids((prev) => [...prev, { x: dataX, y: dataY }]);
    setRipple({ x: svgPt.x, y: svgPt.y, id: Date.now() });
  }, [phase, userCentroids.length, nK, invScaleX, invScaleY]);

  // Step: assign — flash assignment lines briefly
  const handleAssign = useCallback(() => {
    if (userCentroids.length < nK) return;
    const asgn = doAssign(userCentroids);
    setAssignments(asgn);
    const wcss = computeWcss(userCentroids, asgn);
    setWcssHistory((h) => [...h, wcss]);
    // Record initial placement in trail if first assign
    if (centroidTrail.length === 0) {
      setCentroidTrail([[...userCentroids.map((c) => ({ ...c }))]]);
    }
    setPhase("assigned");
    setAnimFlash("assigning");
    setTimeout(() => setAnimFlash("idle"), 800);
  }, [userCentroids, nK, doAssign, computeWcss, centroidTrail.length]);

  // Step: update — flash centroid movement arrows
  const handleUpdate = useCallback(() => {
    setPrevCentroids([...userCentroids]);
    const newC = doUpdate(userCentroids, assignments);
    const moved = newC.some((c, i) => dist(c, userCentroids[i]) > 0.001);
    setUserCentroids(newC);
    setCentroidTrail((t) => [...t, [...newC]]);
    setAnimFlash("updating");
    setTimeout(() => setAnimFlash("idle"), 600);
    if (!moved) {
      setPhase("converged");
      setAutoRunning(false);
    } else {
      setPhase("updated");
    }
    setIteration((i) => i + 1);
  }, [userCentroids, assignments, doUpdate]);

  // Full step: assign then update
  const handleFullStep = useCallback(() => {
    if (userCentroids.length < nK) return;
    const asgn = doAssign(userCentroids);
    setAssignments(asgn);
    const wcss = computeWcss(userCentroids, asgn);
    setWcssHistory((h) => [...h, wcss]);
    setPrevCentroids([...userCentroids]);
    const newC = doUpdate(userCentroids, asgn);
    const moved = newC.some((c, i) => dist(c, userCentroids[i]) > 0.001);
    setUserCentroids(newC);
    setCentroidTrail((t) => {
      // If first step, also record initial position
      if (t.length === 0) return [[...userCentroids.map((c) => ({ ...c }))], [...newC]];
      return [...t, [...newC]];
    });
    setIteration((i) => i + 1);
    setAnimFlash("updating");
    setTimeout(() => setAnimFlash("idle"), 600);
    if (!moved) {
      setPhase("converged");
      setAutoRunning(false);
    } else {
      setPhase("updated");
    }
  }, [userCentroids, nK, doAssign, doUpdate, computeWcss]);

  // Auto-run
  useEffect(() => {
    if (autoRunning && phase !== "converged" && userCentroids.length >= nK) {
      autoRef.current = setInterval(() => {
        handleFullStep();
      }, 500);
    }
    return () => { if (autoRef.current) clearInterval(autoRef.current); };
  }, [autoRunning, phase, handleFullStep, userCentroids.length, nK]);

  // Random placement
  const handleRandomPlace = useCallback(() => {
    if (!points.length) return;
    const rng = [...points].sort(() => Math.random() - 0.5);
    setUserCentroids(rng.slice(0, nK).map((p) => ({ x: p.x, y: p.y })));
    setAssignments([]);
    setPhase("place");
    setIteration(0);
    setWcssHistory([]);
    setShowAlgoResult(false);
    setAutoRunning(false);
    setCentroidTrail([]);
    setPrevCentroids([]);
    setAnimFlash("idle");
  }, [points, nK]);

  // Reset
  const handleReset = useCallback(() => {
    setUserCentroids([]);
    setAssignments([]);
    setPhase("place");
    setIteration(0);
    setWcssHistory([]);
    setShowAlgoResult(false);
    setAutoRunning(false);
    setCentroidTrail([]);
    setPrevCentroids([]);
    setAnimFlash("idle");
  }, []);

  // Current WCSS
  const currentWcss = wcssHistory.length > 0 ? wcssHistory[wcssHistory.length - 1] : null;
  // Algorithm's WCSS from actual result
  const algoInertia = result.inertia;

  if (!points.length) {
    return <p className="text-sm text-gray-500 py-8 text-center">No PCA data available for interactive exploration.</p>;
  }

  return (
    <div className="space-y-4">
      {/* Instructions */}
      <div className="flex items-start gap-3 rounded-lg border border-cyan-200 bg-cyan-50 px-4 py-3">
        <Target className="mt-0.5 h-5 w-5 shrink-0 text-cyan-600" />
        <div className="text-sm text-cyan-800">
          <span className="font-semibold">Try it yourself on your real data!</span>
          {phase === "place" && userCentroids.length < nK && (
            <> Click on the chart to place {nK - userCentroids.length} more centroid{nK - userCentroids.length > 1 ? "s" : ""}. Try to guess where the cluster centers are!</>
          )}
          {phase === "place" && userCentroids.length >= nK && (
            <> All {nK} centroids placed. Click <strong>Assign</strong> to color points by nearest centroid.</>
          )}
          {phase === "assigned" && (
            <> Points assigned! Click <strong>Update</strong> to move centroids to cluster means.</>
          )}
          {phase === "updated" && (
            <> Centroids moved. Click <strong>Assign</strong> again or <strong>Step</strong> to do both at once.</>
          )}
          {phase === "converged" && (
            <> Converged in {iteration} iterations! Toggle "Show Algorithm's Result" to compare.</>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Main SVG */}
        <div className="lg:col-span-3 bg-white border border-gray-200 rounded-lg p-3">
          <svg
            viewBox={`0 0 ${W} ${H}`}
            className={`w-full ${phase === "place" && userCentroids.length < nK ? "cursor-crosshair" : ""}`}
            style={{ maxHeight: 480 }}
            onClick={handleSvgClick}
          >
            {/* Grid */}
            {[0.25, 0.5, 0.75].map((frac) => (
              <g key={frac}>
                <line x1={PAD} y1={PAD + frac * (H - 2 * PAD)} x2={W - PAD} y2={PAD + frac * (H - 2 * PAD)} stroke="#f3f4f6" strokeWidth="1" />
                <line x1={PAD + frac * (W - 2 * PAD)} y1={PAD} x2={PAD + frac * (W - 2 * PAD)} y2={H - PAD} stroke="#f3f4f6" strokeWidth="1" />
              </g>
            ))}
            <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="#d1d5db" strokeWidth="1" />
            <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="#d1d5db" strokeWidth="1" />
            <text x={W / 2} y={H - 6} textAnchor="middle" fontSize="10" fill="#9ca3af">{pca?.pc1_label || "PC1"}</text>
            <text x={10} y={H / 2} textAnchor="middle" fontSize="10" fill="#9ca3af" transform={`rotate(-90,10,${H / 2})`}>{pca?.pc2_label || "PC2"}</text>

            {/* Centroid trails — dashed paths showing movement history */}
            {centroidTrail.length >= 2 && userCentroids.map((_, ci) => {
              const color = CLUSTER_COLORS[ci % CLUSTER_COLORS.length];
              const trail = centroidTrail.map((step) => step[ci]).filter(Boolean);
              if (trail.length < 2) return null;
              const pathD = trail.map((pt, i) =>
                `${i === 0 ? "M" : "L"}${scaleX(pt.x)},${scaleY(pt.y)}`
              ).join(" ");
              return (
                <g key={`trail-${ci}`}>
                  <path d={pathD} fill="none" stroke={color} strokeWidth="1.5" strokeDasharray="5 3" opacity="0.4" />
                  {/* Ghost positions along trail */}
                  {trail.slice(0, -1).map((pt, ti) => (
                    <circle
                      key={ti} cx={scaleX(pt.x)} cy={scaleY(pt.y)} r={4}
                      fill={color} opacity={0.1 + (ti / trail.length) * 0.15}
                      stroke={color} strokeWidth="0.5" strokeOpacity="0.3"
                    />
                  ))}
                </g>
              );
            })}

            {/* Data points */}
            {points.map((p, i) => {
              const hasAssignment = assignments.length > 0;
              const c = hasAssignment ? assignments[i] : -1;
              const fill = hasAssignment ? CLUSTER_COLORS[c % CLUSTER_COLORS.length] : "#94a3b8";
              const opacity = hasAssignment ? 0.7 : 0.5;
              const algoC = showAlgoResult ? p.cluster : -1;
              const algoFill = showAlgoResult ? CLUSTER_COLORS[algoC % CLUSTER_COLORS.length] : fill;
              return (
                <circle
                  key={i}
                  cx={scaleX(p.x)} cy={scaleY(p.y)} r={3.5}
                  fill={showAlgoResult ? algoFill : fill}
                  opacity={opacity}
                  style={{ transition: "fill 0.4s ease, opacity 0.4s ease" }}
                />
              );
            })}

            {/* Assignment lines flash — thin lines from sampled points to their centroid */}
            {animFlash === "assigning" && assignments.length > 0 && userCentroids.length > 0 && (() => {
              // Sample ~60 evenly spaced points to avoid clutter
              const step = Math.max(1, Math.floor(points.length / 60));
              return points.filter((_, i) => i % step === 0).map((p, si) => {
                const idx = si * step;
                const ci = assignments[idx];
                const centroid = userCentroids[ci];
                if (!centroid) return null;
                const color = CLUSTER_COLORS[ci % CLUSTER_COLORS.length];
                return (
                  <line
                    key={`aline-${si}`}
                    x1={scaleX(p.x)} y1={scaleY(p.y)}
                    x2={scaleX(centroid.x)} y2={scaleY(centroid.y)}
                    stroke={color} strokeWidth="0.7" opacity="0"
                  >
                    <animate attributeName="opacity" from="0.5" to="0" dur="0.8s" fill="freeze" />
                  </line>
                );
              });
            })()}

            {/* Centroid "radar pulse" on assignment — expanding circles from each centroid */}
            {animFlash === "assigning" && userCentroids.map((c, ci) => {
              const color = CLUSTER_COLORS[ci % CLUSTER_COLORS.length];
              return (
                <circle key={`radar-${ci}`} cx={scaleX(c.x)} cy={scaleY(c.y)} r="5" fill="none" stroke={color} strokeWidth="1.5" opacity="0">
                  <animate attributeName="r" from="5" to="50" dur="0.7s" fill="freeze" />
                  <animate attributeName="opacity" from="0.5" to="0" dur="0.7s" fill="freeze" />
                </circle>
              );
            })}

            {/* Movement arrows — show previous→current centroid move during update */}
            {animFlash === "updating" && prevCentroids.length > 0 && userCentroids.map((c, ci) => {
              const prev = prevCentroids[ci];
              if (!prev) return null;
              const color = CLUSTER_COLORS[ci % CLUSTER_COLORS.length];
              const x1 = scaleX(prev.x), y1 = scaleY(prev.y);
              const x2 = scaleX(c.x), y2 = scaleY(c.y);
              const d = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
              if (d < 1) return null;
              return (
                <g key={`move-${ci}`}>
                  {/* Ghost of old position */}
                  <circle cx={x1} cy={y1} r={7} fill="none" stroke={color} strokeWidth="1.5" strokeDasharray="3 2" opacity="0">
                    <animate attributeName="opacity" from="0.6" to="0" dur="0.6s" fill="freeze" />
                  </circle>
                  {/* Arrow line from old to new */}
                  <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth="2" opacity="0" markerEnd={`url(#arrowhead-${ci})`}>
                    <animate attributeName="opacity" from="0.7" to="0" dur="0.6s" fill="freeze" />
                  </line>
                </g>
              );
            })}

            {/* Algorithm centroids (when showing comparison) */}
            {showAlgoResult && pca?.centroids_2d?.map((c: number[], i: number) => {
              const cx = scaleX(c[0]);
              const cy = scaleY(c[1] ?? 0);
              return (
                <g key={`algo-${i}`}>
                  <circle cx={cx} cy={cy} r={10} fill="none" stroke={CLUSTER_COLORS[i % CLUSTER_COLORS.length]} strokeWidth="2" strokeDasharray="4 2" />
                  <text x={cx} y={cy - 13} textAnchor="middle" fontSize="9" fill={CLUSTER_COLORS[i % CLUSTER_COLORS.length]} fontWeight="600">algo</text>
                </g>
              );
            })}

            {/* User centroids with animations */}
            {userCentroids.map((c, ci) => {
              const cx = scaleX(c.x);
              const cy = scaleY(c.y);
              const color = CLUSTER_COLORS[ci % CLUSTER_COLORS.length];
              const size = 9;
              return (
                <g key={`uc-${ci}`} style={{ transition: "transform 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)" }} transform={`translate(${cx}, ${cy})`}>
                  {/* Glow ring */}
                  <circle r={14} fill={color} opacity={0.08} />
                  {/* Pulse ring — plays once on mount */}
                  <circle r={size} fill="none" stroke={color} strokeWidth="2" opacity="0">
                    <animate attributeName="r" from={String(size)} to="28" dur="0.6s" fill="freeze" />
                    <animate attributeName="opacity" from="0.6" to="0" dur="0.6s" fill="freeze" />
                  </circle>
                  {/* Scale-in bounce for diamond */}
                  <g>
                    <animateTransform attributeName="transform" type="scale" from="0" to="1" dur="0.35s" fill="freeze" calcMode="spline" keySplines="0.34 1.56 0.64 1" />
                    <polygon
                      points={`0,${-size} ${size},0 0,${size} ${-size},0`}
                      fill={color} stroke="white" strokeWidth="2.5"
                    />
                    <text x={0} y={3} textAnchor="middle" fontSize="8" fontWeight="700" fill="white">
                      {ci}
                    </text>
                  </g>
                </g>
              );
            })}

            {/* Click ripple */}
            {ripple && (
              <circle key={ripple.id} cx={ripple.x} cy={ripple.y} r="5" fill="none" stroke="#06b6d4" strokeWidth="1.5">
                <animate attributeName="r" from="5" to="35" dur="0.5s" fill="freeze" />
                <animate attributeName="opacity" from="0.7" to="0" dur="0.5s" fill="freeze" />
              </circle>
            )}

            {/* Iteration counter overlay */}
            {iteration > 0 && (
              <text x={W - PAD} y={16} textAnchor="end" fontSize="11" fill="#6b7280" fontWeight="600">
                Iteration {iteration}
              </text>
            )}

            {/* "Click to place" hint */}
            {phase === "place" && userCentroids.length < nK && (
              <text x={W / 2} y={20} textAnchor="middle" fontSize="12" fill="#06b6d4" fontWeight="600">
                Click to place centroid {userCentroids.length + 1} of {nK}
              </text>
            )}
          </svg>
        </div>

        {/* Side panel */}
        <div className="space-y-3">
          {/* Controls */}
          <div className="bg-white border border-gray-200 rounded-lg p-3 space-y-2">
            <h4 className="text-xs font-semibold text-gray-700 uppercase tracking-wider">Controls</h4>

            <button
              onClick={handleAssign}
              disabled={userCentroids.length < nK || phase === "assigned" || phase === "converged"}
              className="w-full px-3 py-2 text-xs font-medium rounded-lg border transition-colors bg-blue-50 border-blue-200 text-blue-700 hover:bg-blue-100 disabled:opacity-30 disabled:cursor-not-allowed"
            >
              Assign Points
            </button>
            <button
              onClick={handleUpdate}
              disabled={phase !== "assigned"}
              className="w-full px-3 py-2 text-xs font-medium rounded-lg border transition-colors bg-green-50 border-green-200 text-green-700 hover:bg-green-100 disabled:opacity-30 disabled:cursor-not-allowed"
            >
              Update Centroids
            </button>
            <button
              onClick={handleFullStep}
              disabled={userCentroids.length < nK || phase === "converged"}
              className="w-full px-3 py-2 text-xs font-medium rounded-lg border transition-colors bg-cyan-50 border-cyan-200 text-cyan-700 hover:bg-cyan-100 disabled:opacity-30 disabled:cursor-not-allowed"
            >
              Step (Assign + Update)
            </button>
            <button
              onClick={() => {
                if (autoRunning) { setAutoRunning(false); }
                else if (phase !== "converged" && userCentroids.length >= nK) { setAutoRunning(true); }
              }}
              disabled={userCentroids.length < nK || phase === "converged"}
              className={`w-full px-3 py-2 text-xs font-medium rounded-lg border transition-colors ${
                autoRunning ? "bg-amber-50 border-amber-300 text-amber-700" : "bg-purple-50 border-purple-200 text-purple-700 hover:bg-purple-100"
              } disabled:opacity-30 disabled:cursor-not-allowed`}
            >
              {autoRunning ? "Pause Auto-Run" : "Auto-Run"}
            </button>

            <div className="border-t border-gray-100 pt-2 flex gap-2">
              <button
                onClick={handleRandomPlace}
                className="flex-1 px-2 py-1.5 text-xs rounded-lg border border-gray-200 text-gray-600 hover:bg-gray-50"
              >
                Random
              </button>
              <button
                onClick={handleReset}
                className="flex-1 px-2 py-1.5 text-xs rounded-lg border border-gray-200 text-gray-600 hover:bg-gray-50"
              >
                <RotateCcw className="w-3 h-3 inline mr-1" />Reset
              </button>
            </div>
          </div>

          {/* Stats */}
          <div className="bg-white border border-gray-200 rounded-lg p-3 space-y-2">
            <h4 className="text-xs font-semibold text-gray-700 uppercase tracking-wider">Stats</h4>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="bg-gray-50 rounded p-2">
                <span className="text-gray-400">Iteration</span>
                <div className="font-bold text-gray-900">{iteration}</div>
              </div>
              <div className="bg-gray-50 rounded p-2">
                <span className="text-gray-400">Phase</span>
                <div className="font-bold text-gray-900 capitalize">{phase}</div>
              </div>
              <div className="col-span-2 bg-gray-50 rounded p-2">
                <span className="text-gray-400">Your WCSS</span>
                <div className="font-bold text-gray-900">{currentWcss != null ? currentWcss.toFixed(1) : "—"}</div>
              </div>
              <div className="col-span-2 bg-cyan-50 rounded p-2">
                <span className="text-cyan-600 text-[10px]">Algorithm's WCSS</span>
                <div className="font-bold text-cyan-800">{algoInertia?.toFixed(1) ?? "—"}</div>
              </div>
            </div>

            {/* WCSS mini chart */}
            {wcssHistory.length > 1 && (
              <div>
                <span className="text-[10px] text-gray-400">WCSS over your iterations</span>
                <svg viewBox="0 0 160 50" className="w-full">
                  {(() => {
                    const maxW = Math.max(...wcssHistory);
                    return (
                      <polyline
                        points={wcssHistory.map((v, i) =>
                          `${10 + (i / (wcssHistory.length - 1)) * 140},${45 - (v / maxW) * 40}`
                        ).join(" ")}
                        fill="none" stroke="#06b6d4" strokeWidth="2"
                      />
                    );
                  })()}
                </svg>
              </div>
            )}
          </div>

          {/* Compare toggle */}
          {phase === "converged" && (
            <button
              onClick={() => setShowAlgoResult(!showAlgoResult)}
              className={`w-full px-3 py-2 text-xs font-medium rounded-lg border transition-colors ${
                showAlgoResult ? "bg-cyan-100 border-cyan-300 text-cyan-800" : "bg-white border-gray-200 text-gray-700 hover:bg-gray-50"
              }`}
            >
              {showAlgoResult ? "Hide" : "Show"} Algorithm's Result
            </button>
          )}

          {/* Cluster sizes */}
          {assignments.length > 0 && (
            <div className="bg-white border border-gray-200 rounded-lg p-3">
              <h4 className="text-[10px] font-semibold text-gray-400 uppercase mb-1">Your Cluster Sizes</h4>
              {Array.from({ length: nK }, (_, ci) => {
                const count = assignments.filter((a) => a === ci).length;
                const pct = (count / points.length) * 100;
                return (
                  <div key={ci} className="flex items-center gap-2 text-xs mb-1">
                    <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: CLUSTER_COLORS[ci % CLUSTER_COLORS.length] }} />
                    <span className="text-gray-600 w-8">C{ci}</span>
                    <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                      <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: CLUSTER_COLORS[ci % CLUSTER_COLORS.length] }} />
                    </div>
                    <span className="text-gray-500 w-6 text-right">{count}</span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Info cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-cyan-50 border border-cyan-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-cyan-900 mb-1">When to Use</h4>
          <p className="text-xs text-cyan-700">
            Customer segmentation, image compression, feature engineering, anomaly detection,
            and any task where you need to find natural groups in unlabeled data.
          </p>
        </div>
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-amber-900 mb-1">Limitations</h4>
          <p className="text-xs text-amber-700">
            Assumes spherical clusters of similar size. Sensitive to feature scale (always standardize).
            You must choose K in advance. Can get stuck in local optima (use K-Means++).
          </p>
        </div>
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-green-900 mb-1">K-Means++ Advantage</h4>
          <p className="text-xs text-green-700">
            Standard K-Means picks random initial centroids. K-Means++ spreads them apart
            (each new centroid placed far from existing ones), giving better and more consistent results.
          </p>
        </div>
      </div>
    </div>
  );
}
