/**
 * BlockEditor — Right panel config form for the selected block.
 * Renders different form fields based on block type.
 */

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
  InputField,
  MetricItem,
} from "../types/appBuilder";
import { Plus, Trash2, Link } from "lucide-react";

interface BlockEditorProps {
  block: AppBlock | null;
}

export default function BlockEditor({ block }: BlockEditorProps) {
  const updateBlock = useAppBuilderStore((s) => s.updateBlock);

  if (!block) {
    return (
      <div className="p-6 flex items-center justify-center h-full">
        <p className="text-sm text-gray-400 text-center">
          Select a block to edit its properties
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
      <div className="pb-3 border-b">
        <h2 className="text-sm font-semibold text-gray-700">
          {def?.label ?? block.type}
        </h2>
        <p className="text-xs text-gray-400">{def?.description}</p>
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

      {block.type === "hero" && <HeroEditor config={block.config as HeroConfig} update={update} />}
      {block.type === "text" && <TextEditor config={block.config as TextConfig} update={update} />}
      {block.type === "file_upload" && <FileUploadEditor config={block.config as FileUploadConfig} update={update} />}
      {block.type === "input_fields" && <InputFieldsEditor config={block.config as InputFieldsConfig} update={update} />}
      {block.type === "submit_button" && <SubmitButtonEditor config={block.config as SubmitButtonConfig} update={update} />}
      {block.type === "results_display" && <ResultsDisplayEditor config={block.config as ResultsDisplayConfig} update={update} />}
      {block.type === "metrics_card" && <MetricsCardEditor config={block.config as MetricsCardConfig} update={update} />}
      {block.type === "divider" && <DividerEditor config={block.config as DividerConfig} update={update} />}
      {block.type === "image" && <ImageEditor config={block.config as ImageConfig} update={update} />}
    </div>
  );
}

// ─── Field Helpers ────────────────────────────────────────────────

function Label({ children }: { children: React.ReactNode }) {
  return <label className="block text-xs font-medium text-gray-500 mb-1">{children}</label>;
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

function HeroEditor({ config, update }: { config: HeroConfig; update: (p: Record<string, unknown>) => void }) {
  return (
    <div className="space-y-3">
      <div>
        <Label>Title</Label>
        <TextInput value={config.title} onChange={(v) => update({ title: v })} />
      </div>
      <div>
        <Label>Subtitle</Label>
        <TextInput value={config.subtitle} onChange={(v) => update({ subtitle: v })} />
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
      <Toggle checked={config.showGradient} onChange={(v) => update({ showGradient: v })} label="Show gradient" />
    </div>
  );
}

function TextEditor({ config, update }: { config: TextConfig; update: (p: Record<string, unknown>) => void }) {
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

function FileUploadEditor({ config, update }: { config: FileUploadConfig; update: (p: Record<string, unknown>) => void }) {
  return (
    <div className="space-y-3">
      <div>
        <Label>Label</Label>
        <TextInput value={config.label} onChange={(v) => update({ label: v })} />
      </div>
      <div>
        <Label>Accept Types</Label>
        <TextInput value={config.acceptTypes} onChange={(v) => update({ acceptTypes: v })} placeholder=".csv" />
      </div>
      <div>
        <Label>Help Text</Label>
        <TextInput value={config.helpText} onChange={(v) => update({ helpText: v })} />
      </div>
    </div>
  );
}

function InputFieldsEditor({ config, update }: { config: InputFieldsConfig; update: (p: Record<string, unknown>) => void }) {
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
    const fields = config.fields.map((f, i) => (i === index ? { ...f, ...partial } : f));
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
            <span className="text-xs font-medium text-gray-500">Field {i + 1}</span>
            <button onClick={() => removeField(i)} className="p-1 hover:bg-gray-200 rounded">
              <Trash2 className="h-3 w-3 text-red-500" />
            </button>
          </div>
          <TextInput value={field.name} onChange={(v) => updateField(i, { name: v })} placeholder="Field name" />
          <TextInput value={field.label} onChange={(v) => updateField(i, { label: v })} placeholder="Label" />
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
          <TextInput value={field.placeholder} onChange={(v) => updateField(i, { placeholder: v })} placeholder="Placeholder" />
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

function SubmitButtonEditor({ config, update }: { config: SubmitButtonConfig; update: (p: Record<string, unknown>) => void }) {
  return (
    <div className="space-y-3">
      <div>
        <Label>Button Label</Label>
        <TextInput value={config.label} onChange={(v) => update({ label: v })} />
      </div>
      <div>
        <Label>Loading Text</Label>
        <TextInput value={config.loadingText} onChange={(v) => update({ loadingText: v })} />
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

function ResultsDisplayEditor({ config, update }: { config: ResultsDisplayConfig; update: (p: Record<string, unknown>) => void }) {
  return (
    <div className="space-y-3">
      <div>
        <Label>Title</Label>
        <TextInput value={config.title} onChange={(v) => update({ title: v })} />
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

function MetricsCardEditor({ config, update }: { config: MetricsCardConfig; update: (p: Record<string, unknown>) => void }) {
  const addMetric = () => {
    const metrics = [
      ...config.metrics,
      { key: `metric_${config.metrics.length + 1}`, label: `Metric ${config.metrics.length + 1}`, format: "number" as const },
    ];
    update({ metrics });
  };

  const updateMetric = (index: number, partial: Partial<MetricItem>) => {
    const metrics = config.metrics.map((m, i) => (i === index ? { ...m, ...partial } : m));
    update({ metrics });
  };

  const removeMetric = (index: number) => {
    update({ metrics: config.metrics.filter((_, i) => i !== index) });
  };

  return (
    <div className="space-y-3">
      <div>
        <Label>Title</Label>
        <TextInput value={config.title} onChange={(v) => update({ title: v })} />
      </div>
      {config.metrics.map((metric, i) => (
        <div key={i} className="bg-gray-50 rounded-lg p-3 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-gray-500">Metric {i + 1}</span>
            <button onClick={() => removeMetric(i)} className="p-1 hover:bg-gray-200 rounded">
              <Trash2 className="h-3 w-3 text-red-500" />
            </button>
          </div>
          <TextInput value={metric.key} onChange={(v) => updateMetric(i, { key: v })} placeholder="Key (from results)" />
          <TextInput value={metric.label} onChange={(v) => updateMetric(i, { label: v })} placeholder="Display label" />
          <SelectInput
            value={metric.format}
            onChange={(v) => updateMetric(i, { format: v as MetricItem["format"] })}
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

function DividerEditor({ config, update }: { config: DividerConfig; update: (p: Record<string, unknown>) => void }) {
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

function ImageEditor({ config, update }: { config: ImageConfig; update: (p: Record<string, unknown>) => void }) {
  return (
    <div className="space-y-3">
      <div>
        <Label>Image URL</Label>
        <TextInput value={config.url} onChange={(v) => update({ url: v })} placeholder="https://..." />
      </div>
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
    </div>
  );
}
