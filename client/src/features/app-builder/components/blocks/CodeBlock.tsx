/**
 * Code Block — Displays code with optional line numbers.
 */

import { Code as CodeIcon } from "lucide-react";
import type { BlockRenderProps } from "../BlockRenderer";
import type { CodeConfig } from "../../types/appBuilder";

export default function CodeBlock({ block }: BlockRenderProps) {
  const config = block.config as CodeConfig;
  const lines = config.code.split("\n");

  return (
    <div className="rounded-lg overflow-hidden border border-gray-200">
      {config.title && (
        <div className="flex items-center gap-2 px-4 py-2 bg-gray-100 border-b border-gray-200">
          <CodeIcon className="h-3.5 w-3.5 text-gray-400" />
          <span className="text-xs font-medium text-gray-600">
            {config.title}
          </span>
          <span className="text-xs text-gray-400 ml-auto">
            {config.language}
          </span>
        </div>
      )}
      <div className="bg-gray-900 text-gray-100 p-4 overflow-x-auto">
        <pre className="text-sm font-mono leading-relaxed">
          {lines.map((line, i) => (
            <div key={i} className="flex">
              {config.showLineNumbers && (
                <span className="select-none text-gray-500 w-8 shrink-0 text-right mr-4">
                  {i + 1}
                </span>
              )}
              <span>{line || "\u00A0"}</span>
            </div>
          ))}
        </pre>
      </div>
    </div>
  );
}
