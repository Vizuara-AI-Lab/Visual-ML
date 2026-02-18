import { useEffect, useRef } from "react";
import {
  Play,
  CheckCircle,
  XCircle,
  CheckCheck,
  AlertOctagon,
  ScrollText,
} from "lucide-react";
import type { ExecutionLogEntry } from "../../../../types/pipeline";

interface LogsTabProps {
  logs: ExecutionLogEntry[];
  isExecuting: boolean;
}

const eventConfig: Record<
  ExecutionLogEntry["event"],
  { icon: typeof Play; color: string; bgColor: string }
> = {
  node_started: { icon: Play, color: "text-blue-600", bgColor: "bg-blue-100" },
  node_completed: {
    icon: CheckCircle,
    color: "text-green-600",
    bgColor: "bg-green-100",
  },
  node_failed: { icon: XCircle, color: "text-red-600", bgColor: "bg-red-100" },
  pipeline_completed: {
    icon: CheckCheck,
    color: "text-green-700",
    bgColor: "bg-green-100",
  },
  pipeline_failed: {
    icon: AlertOctagon,
    color: "text-red-700",
    bgColor: "bg-red-100",
  },
};

function formatTime(isoString: string): string {
  try {
    const date = new Date(isoString);
    return date.toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return "--:--:--";
  }
}

export function LogsTab({ logs, isExecuting }: LogsTabProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom during execution
  useEffect(() => {
    if (isExecuting && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs.length, isExecuting]);

  if (logs.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-slate-400">
        <ScrollText className="w-8 h-8 mb-2" />
        <p className="text-sm">No execution logs yet</p>
        <p className="text-xs mt-1">Logs will appear here when you run the pipeline</p>
      </div>
    );
  }

  return (
    <div ref={scrollRef} className="p-4 overflow-y-auto h-full">
      <div className="relative">
        {/* Timeline line */}
        <div className="absolute left-[15px] top-2 bottom-2 w-px bg-slate-200" />

        <div className="space-y-1">
          {logs.map((log, idx) => {
            const config = eventConfig[log.event];
            const Icon = config.icon;
            const isLast = idx === logs.length - 1;

            return (
              <div
                key={idx}
                className={`flex items-start gap-3 py-1.5 pl-0 pr-2 rounded-lg transition-colors ${
                  isLast && isExecuting ? "bg-slate-50" : ""
                }`}
              >
                {/* Icon */}
                <div
                  className={`w-[30px] h-[30px] rounded-full ${config.bgColor} flex items-center justify-center shrink-0 z-10`}
                >
                  <Icon className={`w-3.5 h-3.5 ${config.color}`} />
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0 pt-1">
                  <span className="text-sm text-slate-700">{log.message}</span>
                </div>

                {/* Timestamp */}
                <span className="text-xs text-slate-400 font-mono shrink-0 pt-1.5">
                  {formatTime(log.timestamp)}
                </span>
              </div>
            );
          })}

          {/* Executing indicator */}
          {isExecuting && (
            <div className="flex items-center gap-3 py-1.5">
              <div className="w-[30px] h-[30px] rounded-full bg-blue-50 flex items-center justify-center shrink-0 z-10">
                <div className="w-3 h-3 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
              </div>
              <span className="text-sm text-slate-500">Processing...</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
