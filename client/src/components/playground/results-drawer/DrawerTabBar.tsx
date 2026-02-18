import {
  LayoutDashboard,
  Boxes,
  TrendingUp,
  ScrollText,
  Minus,
  X,
} from "lucide-react";

export type DrawerTab = "summary" | "nodeResults" | "metrics" | "logs";

const tabs: { id: DrawerTab; label: string; icon: typeof LayoutDashboard }[] = [
  { id: "summary", label: "Summary", icon: LayoutDashboard },
  { id: "nodeResults", label: "Node Results", icon: Boxes },
  { id: "metrics", label: "Metrics", icon: TrendingUp },
  { id: "logs", label: "Logs", icon: ScrollText },
];

interface DrawerTabBarProps {
  activeTab: DrawerTab;
  onTabChange: (tab: DrawerTab) => void;
  onMinimize: () => void;
  onClose: () => void;
}

export function DrawerTabBar({
  activeTab,
  onTabChange,
  onMinimize,
  onClose,
}: DrawerTabBarProps) {
  return (
    <div className="flex items-center justify-between px-3 py-1.5 border-b border-slate-200/60 bg-slate-50/80">
      <div className="flex items-center gap-1">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
                isActive
                  ? "bg-white text-slate-800 shadow-sm"
                  : "text-slate-500 hover:text-slate-700 hover:bg-slate-100/80"
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              {tab.label}
            </button>
          );
        })}
      </div>
      <div className="flex items-center gap-0.5">
        <button
          onClick={onMinimize}
          className="p-1.5 hover:bg-slate-200/80 rounded-md transition-colors"
          title="Minimize"
        >
          <Minus className="w-4 h-4 text-slate-500" />
        </button>
        <button
          onClick={onClose}
          className="p-1.5 hover:bg-slate-200/80 rounded-md transition-colors"
          title="Close"
        >
          <X className="w-4 h-4 text-slate-500" />
        </button>
      </div>
    </div>
  );
}
