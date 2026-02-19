/**
 * BlockEditor — Right panel config form for the selected block.
 * Renders different form fields based on block type.
 */

import { useState } from "react";
import { useAppBuilderStore } from "../store/appBuilderStore";
import { getBlockDefinition } from "../config/blockDefinitions";
import type {
  AppBlock,
  HeroConfig,
  TextConfig,
  FileUploadConfig,
  InputFieldsConfig,
  SubmitButtonConfig,
  ResultsDisplayConfig,
  MetricsCardConfig,
  DividerConfig,
  ImageConfig,
  SpacerConfig,
  AlertConfig,
  CodeConfig,
  VideoEmbedConfig,
  BlockStyleConfig,
  InputField,
  MetricItem,
} from "../types/appBuilder";
import {
  Plus,
  Trash2,
  Link,
  MousePointerClick,
  Settings2,
  ChevronDown,
  Upload,
  X,
} from "lucide-react";

interface BlockEditorProps {
  block: AppBlock | null;
}

export default function BlockEditor({ block }: BlockEditorProps) {
  const updateBlock = useAppBuilderStore((s) => s.updateBlock);

  if (!block) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-6">
        <div className="w-16 h-16 rounded-2xl bg-linear-to-br from-indigo-50 to-violet-50 border border-indigo-100 flex items-center justify-center mb-4">
          <MousePointerClick className="h-7 w-7 text-indigo-300" />
        </div>
        <p className="text-sm font-semibold text-gray-600 mb-1">
          No block selected
        </p>
        <p className="text-xs text-gray-400 text-center leading-relaxed max-w-[180px]">
          Click a block on the canvas to edit its properties here.
        </p>
      </div>
    );
  }

  const def = getBlockDefinition(block.type);

  const update = (partial: Record<string, unknown>) => {
    updateBlock(block.id, partial as Partial<AppBlock["config"]>);
  };

  return (
    <div className="p-4 space-y-4">
      <div className="bg-gradient-to-r from-indigo-50/80 to-violet-50/50 rounded-lg px-3 py-3 border border-indigo-100/60">
        <div className="flex items-center gap-2 mb-0.5">
          <div className="w-6 h-6 rounded-md bg-white/80 flex items-center justify-center shadow-sm">
            <Settings2 className="h-3.5 w-3.5 text-indigo-500" />
          </div>
          <h2 className="text-sm font-semibold text-gray-800">
            {def?.label ?? block.type}
          </h2>
        </div>
        <p className="text-[11px] text-gray-500 ml-8">{def?.description}</p>
      </div>

      {block.nodeLabel && (
        <div className="flex items-center gap-2 bg-indigo-50 border border-indigo-200 rounded-lg px-3 py-2">
          <Link className="h-3.5 w-3.5 text-indigo-500 shrink-0" />
          <div>
            <p className="text-xs font-medium text-indigo-700">
              Mapped to: {block.nodeLabel}
            </p>
            <p className="text-[10px] text-indigo-400">{block.nodeId}</p>
          </div>
        </div>
      )}

      {block.type === "hero" && (
        <HeroEditor config={block.config as HeroConfig} update={update} />
      )}
      {block.type === "text" && (
        <TextEditor config={block.config as TextConfig} update={update} />
      )}
      {block.type === "file_upload" && (
        <FileUploadEditor
          config={block.config as FileUploadConfig}
          update={update}
        />
      )}
      {block.type === "input_fields" && (
        <InputFieldsEditor
          config={block.config as InputFieldsConfig}
          update={update}
        />
      )}
      {block.type === "submit_button" && (
        <SubmitButtonEditor
          config={block.config as SubmitButtonConfig}
          update={update}
        />
      )}
      {block.type === "results_display" && (
        <ResultsDisplayEditor
          config={block.config as ResultsDisplayConfig}
          update={update}
        />
      )}
      {block.type === "metrics_card" && (
        <MetricsCardEditor
          config={block.config as MetricsCardConfig}
          update={update}
        />
      )}
      {block.type === "divider" && (
        <DividerEditor config={block.config as DividerConfig} update={update} />
      )}
      {block.type === "image" && (
        <ImageEditor config={block.config as ImageConfig} update={update} />
      )}
      {block.type === "spacer" && (
        <SpacerEditor config={block.config as SpacerConfig} update={update} />
      )}
      {block.type === "alert" && (
        <AlertEditor config={block.config as AlertConfig} update={update} />
      )}
      {block.type === "code" && (
        <CodeEditor config={block.config as CodeConfig} update={update} />
      )}
      {block.type === "video_embed" && (
        <VideoEmbedEditor
          config={block.config as VideoEmbedConfig}
          update={update}
        />
      )}

      <BlockStyleEditor block={block} />
    </div>
  );
}

// ─── Field Helpers ────────────────────────────────────────────────

function Label({ children }: { children: React.ReactNode }) {
  return (
    <label className="block text-xs font-medium text-gray-500 mb-1">
      {children}
    </label>
  );
}

function TextInput({
  value,
  onChange,
  placeholder,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}) {
  return (
    <input
      type="text"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      className="w-full px-2.5 py-1.5 border rounded-md text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
    />
  );
}

function SelectInput({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full px-2.5 py-1.5 border rounded-md text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
    >
      {options.map((opt) => (
        <option key={opt.value} value={opt.value}>
          {opt.label}
        </option>
      ))}
    </select>
  );
}

function Toggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
}) {
  return (
    <label className="flex items-center gap-2 cursor-pointer">
      <div
        onClick={() => onChange(!checked)}
        className={`w-8 h-5 rounded-full transition-colors relative ${
          checked ? "bg-indigo-600" : "bg-gray-300"
        }`}
      >
        <div
          className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${
            checked ? "translate-x-3" : ""
          }`}
        />
      </div>
      <span className="text-xs text-gray-600">{label}</span>
    </label>
  );
}

// ─── Per-block Editors ────────────────────────────────────────────

function HeroEditor({
  config,
  update,
}: {
  config: HeroConfig;
  update: (p: Record<string, unknown>) => void;
}) {
  return (
    <div className="space-y-3">
      <div>
        <Label>Title</Label>
        <TextInput
          value={config.title}
          onChange={(v) => update({ title: v })}
        />
      </div>
      <div>
        <Label>Subtitle</Label>
        <TextInput
          value={config.subtitle}
          onChange={(v) => update({ subtitle: v })}
        />
      </div>
      <div>
        <Label>Alignment</Label>
        <SelectInput
          value={config.alignment}
          onChange={(v) => update({ alignment: v })}
          options={[
            { value: "left", label: "Left" },
            { value: "center", label: "Center" },
            { value: "right", label: "Right" },
          ]}
        />
      </div>
      <Toggle
        checked={config.showGradient}
        onChange={(v) => update({ showGradient: v })}
        label="Show gradient"
      />
    </div>
  );
}

function TextEditor({
  config,
  update,
}: {
  config: TextConfig;
  update: (p: Record<string, unknown>) => void;
}) {
  return (
    <div className="space-y-3">
      <div>
        <Label>Content</Label>
        <textarea
          value={config.content}
          onChange={(e) => update({ content: e.target.value })}
          rows={4}
          className="w-full px-2.5 py-1.5 border rounded-md text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 resize-none"
        />
      </div>
      <div>
        <Label>Size</Label>
        <SelectInput
          value={config.size}
          onChange={(v) => update({ size: v })}
          options={[
            { value: "sm", label: "Small" },
            { value: "md", label: "Medium" },
            { value: "lg", label: "Large" },
          ]}
        />
      </div>
      <div>
        <Label>Alignment</Label>
        <SelectInput
          value={config.alignment}
          onChange={(v) => update({ alignment: v })}
          options={[
            { value: "left", label: "Left" },
            { value: "center", label: "Center" },
            { value: "right", label: "Right" },
          ]}
        />
      </div>
    </div>
  );
}

function FileUploadEditor({
  config,
  update,
}: {
  config: FileUploadConfig;
  update: (p: Record<string, unknown>) => void;
}) {
  return (
    <div className="space-y-3">
      <div>
        <Label>Label</Label>
        <TextInput
          value={config.label}
          onChange={(v) => update({ label: v })}
        />
      </div>
      <div>
        <Label>Accept Types</Label>
        <TextInput
          value={config.acceptTypes}
          onChange={(v) => update({ acceptTypes: v })}
          placeholder=".csv"
        />
      </div>
      <div>
        <Label>Help Text</Label>
        <TextInput
          value={config.helpText}
          onChange={(v) => update({ helpText: v })}
        />
      </div>
    </div>
  );
}

function InputFieldsEditor({
  config,
  update,
}: {
  config: InputFieldsConfig;
  update: (p: Record<string, unknown>) => void;
}) {
  const addField = () => {
    const fields = [
      ...config.fields,
      {
        name: `field_${config.fields.length + 1}`,
        label: `Field ${config.fields.length + 1}`,
        type: "text" as const,
        placeholder: "",
        required: false,
      },
    ];
    update({ fields });
  };

  const updateField = (index: number, partial: Partial<InputField>) => {
    const fields = config.fields.map((f, i) =>
      i === index ? { ...f, ...partial } : f,
    );
    update({ fields });
  };

  const removeField = (index: number) => {
    update({ fields: config.fields.filter((_, i) => i !== index) });
  };

  return (
    <div className="space-y-3">
      {config.fields.map((field, i) => (
        <div key={i} className="bg-gray-50 rounded-lg p-3 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-gray-500">
              Field {i + 1}
            </span>
            <button
              onClick={() => removeField(i)}
              className="p-1 hover:bg-gray-200 rounded"
            >
              <Trash2 className="h-3 w-3 text-red-500" />
            </button>
          </div>
          <TextInput
            value={field.name}
            onChange={(v) => updateField(i, { name: v })}
            placeholder="Field name"
          />
          <TextInput
            value={field.label}
            onChange={(v) => updateField(i, { label: v })}
            placeholder="Label"
          />
          <SelectInput
            value={field.type}
            onChange={(v) => updateField(i, { type: v as InputField["type"] })}
            options={[
              { value: "text", label: "Text" },
              { value: "number", label: "Number" },
              { value: "select", label: "Select" },
              { value: "textarea", label: "Textarea" },
            ]}
          />
          <TextInput
            value={field.placeholder}
            onChange={(v) => updateField(i, { placeholder: v })}
            placeholder="Placeholder"
          />
        </div>
      ))}
      <button
        onClick={addField}
        className="w-full flex items-center justify-center gap-1.5 py-2 border-2 border-dashed rounded-lg text-sm text-gray-500 hover:text-indigo-600 hover:border-indigo-300 transition-colors"
      >
        <Plus className="h-4 w-4" />
        Add Field
      </button>
    </div>
  );
}

function SubmitButtonEditor({
  config,
  update,
}: {
  config: SubmitButtonConfig;
  update: (p: Record<string, unknown>) => void;
}) {
  return (
    <div className="space-y-3">
      <div>
        <Label>Button Label</Label>
        <TextInput
          value={config.label}
          onChange={(v) => update({ label: v })}
        />
      </div>
      <div>
        <Label>Loading Text</Label>
        <TextInput
          value={config.loadingText}
          onChange={(v) => update({ loadingText: v })}
        />
      </div>
      <div>
        <Label>Variant</Label>
        <SelectInput
          value={config.variant}
          onChange={(v) => update({ variant: v })}
          options={[
            { value: "gradient", label: "Gradient" },
            { value: "primary", label: "Primary Color" },
            { value: "secondary", label: "Secondary" },
          ]}
        />
      </div>
    </div>
  );
}

function ResultsDisplayEditor({
  config,
  update,
}: {
  config: ResultsDisplayConfig;
  update: (p: Record<string, unknown>) => void;
}) {
  return (
    <div className="space-y-3">
      <div>
        <Label>Title</Label>
        <TextInput
          value={config.title}
          onChange={(v) => update({ title: v })}
        />
      </div>
      <div>
        <Label>Display Mode</Label>
        <SelectInput
          value={config.displayMode}
          onChange={(v) => update({ displayMode: v })}
          options={[
            { value: "table", label: "Table" },
            { value: "card", label: "Cards" },
            { value: "json", label: "Raw JSON" },
          ]}
        />
      </div>
    </div>
  );
}

function MetricsCardEditor({
  config,
  update,
}: {
  config: MetricsCardConfig;
  update: (p: Record<string, unknown>) => void;
}) {
  const addMetric = () => {
    const metrics = [
      ...config.metrics,
      {
        key: `metric_${config.metrics.length + 1}`,
        label: `Metric ${config.metrics.length + 1}`,
        format: "number" as const,
      },
    ];
    update({ metrics });
  };

  const updateMetric = (index: number, partial: Partial<MetricItem>) => {
    const metrics = config.metrics.map((m, i) =>
      i === index ? { ...m, ...partial } : m,
    );
    update({ metrics });
  };

  const removeMetric = (index: number) => {
    update({ metrics: config.metrics.filter((_, i) => i !== index) });
  };

  return (
    <div className="space-y-3">
      <div>
        <Label>Title</Label>
        <TextInput
          value={config.title}
          onChange={(v) => update({ title: v })}
        />
      </div>
      {config.metrics.map((metric, i) => (
        <div key={i} className="bg-gray-50 rounded-lg p-3 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-gray-500">
              Metric {i + 1}
            </span>
            <button
              onClick={() => removeMetric(i)}
              className="p-1 hover:bg-gray-200 rounded"
            >
              <Trash2 className="h-3 w-3 text-red-500" />
            </button>
          </div>
          <TextInput
            value={metric.key}
            onChange={(v) => updateMetric(i, { key: v })}
            placeholder="Key (from results)"
          />
          <TextInput
            value={metric.label}
            onChange={(v) => updateMetric(i, { label: v })}
            placeholder="Display label"
          />
          <SelectInput
            value={metric.format}
            onChange={(v) =>
              updateMetric(i, { format: v as MetricItem["format"] })
            }
            options={[
              { value: "number", label: "Number" },
              { value: "percentage", label: "Percentage" },
              { value: "text", label: "Text" },
            ]}
          />
        </div>
      ))}
      <button
        onClick={addMetric}
        className="w-full flex items-center justify-center gap-1.5 py-2 border-2 border-dashed rounded-lg text-sm text-gray-500 hover:text-indigo-600 hover:border-indigo-300 transition-colors"
      >
        <Plus className="h-4 w-4" />
        Add Metric
      </button>
    </div>
  );
}

function DividerEditor({
  config,
  update,
}: {
  config: DividerConfig;
  update: (p: Record<string, unknown>) => void;
}) {
  return (
    <div>
      <Label>Style</Label>
      <SelectInput
        value={config.style}
        onChange={(v) => update({ style: v })}
        options={[
          { value: "line", label: "Line" },
          { value: "space", label: "Space" },
          { value: "dots", label: "Dots" },
        ]}
      />
    </div>
  );
}

function ImageEditor({
  config,
  update,
}: {
  config: ImageConfig;
  update: (p: Record<string, unknown>) => void;
}) {
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) return;
    if (file.size > 5 * 1024 * 1024) {
      alert("Image must be smaller than 5MB");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      update({ base64Data: reader.result as string, url: "" });
    };
    reader.readAsDataURL(file);
  };

  const hasImage = config.base64Data || config.url;

  return (
    <div className="space-y-3">
      <div>
        <Label>Upload Image</Label>
        <label className="flex items-center justify-center gap-2 w-full py-3 border-2 border-dashed rounded-lg cursor-pointer hover:border-indigo-400 hover:bg-indigo-50 transition-colors">
          <Upload className="h-4 w-4 text-gray-400" />
          <span className="text-sm text-gray-500">
            {config.base64Data ? "Replace image" : "Choose image file"}
          </span>
          <input
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            className="hidden"
          />
        </label>
        {config.base64Data && (
          <div className="flex items-center justify-between mt-2 bg-green-50 border border-green-200 rounded-md px-3 py-1.5">
            <span className="text-xs text-green-700">Image uploaded</span>
            <button
              onClick={() => update({ base64Data: undefined })}
              className="p-0.5 hover:bg-green-100 rounded"
            >
              <X className="h-3 w-3 text-green-600" />
            </button>
          </div>
        )}
      </div>

      {!config.base64Data && (
        <div>
          <Label>Or paste Image URL</Label>
          <TextInput
            value={config.url}
            onChange={(v) => update({ url: v })}
            placeholder="https://..."
          />
        </div>
      )}

      <div>
        <Label>Alt Text</Label>
        <TextInput value={config.alt} onChange={(v) => update({ alt: v })} />
      </div>
      <div>
        <Label>Width</Label>
        <SelectInput
          value={config.width}
          onChange={(v) => update({ width: v })}
          options={[
            { value: "sm", label: "Small" },
            { value: "md", label: "Medium" },
            { value: "lg", label: "Large" },
            { value: "full", label: "Full Width" },
          ]}
        />
      </div>

      {hasImage && (
        <div>
          <Label>Preview</Label>
          <img
            src={config.base64Data || config.url}
            alt={config.alt}
            className="w-full rounded-md border max-h-32 object-cover"
          />
        </div>
      )}
    </div>
  );
}

// ─── New Block Editors ───────────────────────────────────────────

function SpacerEditor({
  config,
  update,
}: {
  config: SpacerConfig;
  update: (p: Record<string, unknown>) => void;
}) {
  return (
    <div className="space-y-3">
      <div>
        <Label>Height (px)</Label>
        <input
          type="range"
          min={8}
          max={128}
          step={8}
          value={config.height}
          onChange={(e) => update({ height: Number(e.target.value) })}
          className="w-full accent-indigo-600"
        />
        <p className="text-xs text-gray-400 text-right">{config.height}px</p>
      </div>
    </div>
  );
}

function AlertEditor({
  config,
  update,
}: {
  config: AlertConfig;
  update: (p: Record<string, unknown>) => void;
}) {
  return (
    <div className="space-y-3">
      <div>
        <Label>Variant</Label>
        <SelectInput
          value={config.variant}
          onChange={(v) => update({ variant: v })}
          options={[
            { value: "info", label: "Info" },
            { value: "warning", label: "Warning" },
            { value: "success", label: "Success" },
            { value: "error", label: "Error" },
          ]}
        />
      </div>
      <div>
        <Label>Title</Label>
        <TextInput
          value={config.title}
          onChange={(v) => update({ title: v })}
        />
      </div>
      <div>
        <Label>Message</Label>
        <textarea
          value={config.message}
          onChange={(e) => update({ message: e.target.value })}
          rows={3}
          className="w-full px-2.5 py-1.5 border rounded-md text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 resize-none"
        />
      </div>
      <Toggle
        checked={config.showIcon}
        onChange={(v) => update({ showIcon: v })}
        label="Show icon"
      />
    </div>
  );
}

function CodeEditor({
  config,
  update,
}: {
  config: CodeConfig;
  update: (p: Record<string, unknown>) => void;
}) {
  return (
    <div className="space-y-3">
      <div>
        <Label>Title / Filename</Label>
        <TextInput
          value={config.title}
          onChange={(v) => update({ title: v })}
          placeholder="e.g. model.py"
        />
      </div>
      <div>
        <Label>Language</Label>
        <SelectInput
          value={config.language}
          onChange={(v) => update({ language: v })}
          options={[
            { value: "python", label: "Python" },
            { value: "javascript", label: "JavaScript" },
            { value: "typescript", label: "TypeScript" },
            { value: "json", label: "JSON" },
            { value: "html", label: "HTML" },
            { value: "css", label: "CSS" },
            { value: "bash", label: "Bash" },
            { value: "sql", label: "SQL" },
            { value: "text", label: "Plain Text" },
          ]}
        />
      </div>
      <div>
        <Label>Code</Label>
        <textarea
          value={config.code}
          onChange={(e) => update({ code: e.target.value })}
          rows={8}
          className="w-full px-2.5 py-1.5 border rounded-md text-sm font-mono focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 resize-y"
          spellCheck={false}
        />
      </div>
      <Toggle
        checked={config.showLineNumbers}
        onChange={(v) => update({ showLineNumbers: v })}
        label="Show line numbers"
      />
    </div>
  );
}

function VideoEmbedEditor({
  config,
  update,
}: {
  config: VideoEmbedConfig;
  update: (p: Record<string, unknown>) => void;
}) {
  return (
    <div className="space-y-3">
      <div>
        <Label>Video URL</Label>
        <TextInput
          value={config.url}
          onChange={(v) => update({ url: v })}
          placeholder="https://youtube.com/watch?v=..."
        />
      </div>
      <div>
        <Label>Aspect Ratio</Label>
        <SelectInput
          value={config.aspectRatio}
          onChange={(v) => update({ aspectRatio: v })}
          options={[
            { value: "16:9", label: "16:9 (Widescreen)" },
            { value: "4:3", label: "4:3 (Standard)" },
          ]}
        />
      </div>
      <div>
        <Label>Caption</Label>
        <TextInput
          value={config.caption}
          onChange={(v) => update({ caption: v })}
          placeholder="Optional caption"
        />
      </div>
    </div>
  );
}

// ─── Per-Block Style Editor ──────────────────────────────────────

function BlockStyleEditor({ block }: { block: AppBlock }) {
  const [isOpen, setIsOpen] = useState(false);

  const updateStyle = (partial: Partial<BlockStyleConfig>) => {
    const { blocks } = useAppBuilderStore.getState();
    useAppBuilderStore.setState({
      blocks: blocks.map((b) =>
        b.id === block.id
          ? { ...b, style: { ...(b.style || {}), ...partial } }
          : b,
      ),
      isDirty: true,
    });
  };

  const resetStyles = () => {
    const { blocks } = useAppBuilderStore.getState();
    useAppBuilderStore.setState({
      blocks: blocks.map((b) =>
        b.id === block.id ? { ...b, style: undefined } : b,
      ),
      isDirty: true,
    });
  };

  const s = block.style || {};

  return (
    <div className="border-t pt-3 mt-3">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center justify-between w-full px-2.5 py-1.5 rounded-md text-xs font-medium text-gray-500 hover:text-indigo-600 hover:bg-indigo-50/50 transition-colors"
      >
        <div className="flex items-center gap-1.5">
          <div
            className={`w-1.5 h-1.5 rounded-full ${Object.keys(s).length > 0 ? "bg-indigo-500" : "bg-gray-300"}`}
          />
          <span>Custom Style</span>
        </div>
        <ChevronDown
          className={`h-3.5 w-3.5 transition-transform duration-200 ${isOpen ? "rotate-180" : ""}`}
        />
      </button>
      {isOpen && (
        <div className="space-y-3 mt-3 bg-gray-50/50 rounded-lg p-3 border border-gray-100">
          <div>
            <Label>Background Color</Label>
            <div className="flex items-center gap-2">
              <input
                type="color"
                value={s.backgroundColor || "#ffffff"}
                onChange={(e) =>
                  updateStyle({ backgroundColor: e.target.value })
                }
                className="w-8 h-8 rounded cursor-pointer border-0"
              />
              <TextInput
                value={s.backgroundColor || ""}
                onChange={(v) =>
                  updateStyle({ backgroundColor: v || undefined })
                }
                placeholder="transparent"
              />
            </div>
          </div>
          <div>
            <Label>Text Color</Label>
            <div className="flex items-center gap-2">
              <input
                type="color"
                value={s.textColor || "#000000"}
                onChange={(e) => updateStyle({ textColor: e.target.value })}
                className="w-8 h-8 rounded cursor-pointer border-0"
              />
              <TextInput
                value={s.textColor || ""}
                onChange={(v) => updateStyle({ textColor: v || undefined })}
                placeholder="inherit"
              />
            </div>
          </div>
          <div>
            <Label>Border Radius (px)</Label>
            <input
              type="number"
              min={0}
              max={32}
              value={s.borderRadius ?? ""}
              onChange={(e) =>
                updateStyle({
                  borderRadius: e.target.value
                    ? Number(e.target.value)
                    : undefined,
                })
              }
              className="w-full px-2.5 py-1.5 border rounded-md text-sm"
              placeholder="0"
            />
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <Label>Padding X (px)</Label>
              <input
                type="number"
                min={0}
                max={64}
                value={s.paddingX ?? ""}
                onChange={(e) =>
                  updateStyle({
                    paddingX: e.target.value
                      ? Number(e.target.value)
                      : undefined,
                  })
                }
                className="w-full px-2.5 py-1.5 border rounded-md text-sm"
                placeholder="0"
              />
            </div>
            <div>
              <Label>Padding Y (px)</Label>
              <input
                type="number"
                min={0}
                max={64}
                value={s.paddingY ?? ""}
                onChange={(e) =>
                  updateStyle({
                    paddingY: e.target.value
                      ? Number(e.target.value)
                      : undefined,
                  })
                }
                className="w-full px-2.5 py-1.5 border rounded-md text-sm"
                placeholder="0"
              />
            </div>
          </div>
          <div>
            <Label>Border Color</Label>
            <div className="flex items-center gap-2">
              <input
                type="color"
                value={s.borderColor || "#e5e7eb"}
                onChange={(e) => updateStyle({ borderColor: e.target.value })}
                className="w-8 h-8 rounded cursor-pointer border-0"
              />
              <TextInput
                value={s.borderColor || ""}
                onChange={(v) => updateStyle({ borderColor: v || undefined })}
                placeholder="none"
              />
            </div>
          </div>
          <div>
            <Label>Border Width (px)</Label>
            <input
              type="number"
              min={0}
              max={8}
              value={s.borderWidth ?? ""}
              onChange={(e) =>
                updateStyle({
                  borderWidth: e.target.value
                    ? Number(e.target.value)
                    : undefined,
                })
              }
              className="w-full px-2.5 py-1.5 border rounded-md text-sm"
              placeholder="0"
            />
          </div>
          <button
            onClick={resetStyles}
            className="text-xs text-red-500 hover:text-red-700"
          >
            Reset styles
          </button>
        </div>
      )}
    </div>
  );
}
