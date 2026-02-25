/**
 * ThemePanel — Right panel for theme settings (color, font, dark mode).
 */

import { useAppBuilderStore } from "../store/appBuilderStore";

const PRESET_COLORS = [
  "#6366f1", // indigo
  "#8b5cf6", // violet
  "#ec4899", // pink
  "#ef4444", // red
  "#f97316", // orange
  "#eab308", // yellow
  "#22c55e", // green
  "#06b6d4", // cyan
  "#3b82f6", // blue
  "#1e293b", // slate
];

const FONT_OPTIONS = [
  { value: "Inter", label: "Inter" },
  { value: "system-ui", label: "System Default" },
  { value: "Georgia", label: "Georgia (Serif)" },
  { value: "'Fira Code'", label: "Fira Code (Mono)" },
];

export default function ThemePanel() {
  const { theme, setTheme } = useAppBuilderStore();

  return (
    <div className="p-4 space-y-5">
      <div className="pb-2">
        <h2 className="text-sm font-semibold text-gray-700">Theme</h2>
        <p className="text-[11px] text-gray-400 mt-0.5">Customize the look and feel</p>
      </div>

      {/* Primary Color */}
      <div className="space-y-2.5">
        <label className="block text-xs font-medium text-gray-500">
          Primary Color
        </label>
        <div className="flex flex-wrap gap-2">
          {PRESET_COLORS.map((color) => (
            <button
              key={color}
              onClick={() => setTheme({ primaryColor: color })}
              className={`w-7 h-7 rounded-lg transition-all duration-200 ${
                theme.primaryColor === color
                  ? "ring-2 ring-offset-2 ring-gray-400 scale-110"
                  : "hover:scale-110"
              }`}
              style={{ backgroundColor: color }}
            />
          ))}
        </div>
        <div className="flex items-center gap-2">
          <input
            type="color"
            value={theme.primaryColor}
            onChange={(e) => setTheme({ primaryColor: e.target.value })}
            className="w-8 h-8 rounded-lg cursor-pointer border-0"
          />
          <input
            type="text"
            value={theme.primaryColor}
            onChange={(e) => setTheme({ primaryColor: e.target.value })}
            className="flex-1 px-2.5 py-1.5 border border-gray-200 rounded-lg text-sm font-mono focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-shadow"
          />
        </div>
      </div>

      {/* Font Family */}
      <div className="space-y-2">
        <label className="block text-xs font-medium text-gray-500">
          Font Family
        </label>
        <select
          value={theme.fontFamily}
          onChange={(e) => setTheme({ fontFamily: e.target.value })}
          className="w-full px-2.5 py-1.5 border border-gray-200 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-shadow"
        >
          {FONT_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>

      {/* Dark Mode */}
      <div>
        <label className="flex items-center gap-2.5 cursor-pointer">
          <div
            onClick={() => setTheme({ darkMode: !theme.darkMode })}
            className={`w-9 h-5 rounded-full transition-colors relative ${
              theme.darkMode ? "bg-indigo-600" : "bg-gray-300"
            }`}
          >
            <div
              className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow-sm transition-transform ${
                theme.darkMode ? "translate-x-4" : ""
              }`}
            />
          </div>
          <span className="text-xs font-medium text-gray-600">Dark Mode</span>
        </label>
      </div>

      {/* Preview swatch */}
      <div className="rounded-xl p-4 bg-gray-50/80 border border-gray-100" style={{ fontFamily: theme.fontFamily }}>
        <p className="text-[10px] text-gray-400 font-medium mb-2.5 uppercase tracking-wider">Preview</p>
        <div
          className="rounded-xl px-4 py-3 text-white text-sm font-medium text-center shadow-sm transition-colors"
          style={{ backgroundColor: theme.primaryColor }}
        >
          Sample Button
        </div>
      </div>
    </div>
  );
}
