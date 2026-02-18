/**
 * Input Fields Block — Dynamic form inputs for feature values.
 */

import type { BlockRenderProps } from "../BlockRenderer";
import type { InputFieldsConfig } from "../../types/appBuilder";

export default function InputFieldsBlock({
  block,
  mode,
  formData,
  onFieldChange,
}: BlockRenderProps) {
  const config = block.config as InputFieldsConfig;
  const isInteractive = mode === "live";

  // Build a lookup from field name to its node mapping
  const mappingByField = new Map(
    (config.fieldMappings ?? []).map((m) => [m.fieldName, m]),
  );

  const handleChange = (fieldName: string, value: unknown) => {
    const mapping = mappingByField.get(fieldName);
    onFieldChange?.(fieldName, value, mapping?.nodeId, mapping?.nodeConfigKey);
  };

  return (
    <div className="bg-white rounded-xl border p-6 space-y-4">
      {config.fields.map((field) => (
        <div key={field.name}>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            {field.label}
            {field.required && <span className="text-red-500 ml-0.5">*</span>}
          </label>

          {field.type === "select" ? (
            <select
              disabled={!isInteractive}
              value={(formData?.[field.name] as string) ?? ""}
              onChange={(e) => handleChange(field.name, e.target.value)}
              className="w-full px-3 py-2 border rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 disabled:bg-gray-50"
            >
              <option value="">{field.placeholder || "Select..."}</option>
              {field.options?.map((opt) => (
                <option key={opt} value={opt}>
                  {opt}
                </option>
              ))}
            </select>
          ) : field.type === "textarea" ? (
            <textarea
              disabled={!isInteractive}
              placeholder={field.placeholder}
              value={(formData?.[field.name] as string) ?? ""}
              onChange={(e) => handleChange(field.name, e.target.value)}
              rows={3}
              className="w-full px-3 py-2 border rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 disabled:bg-gray-50 resize-none"
            />
          ) : (
            <input
              type={field.type}
              disabled={!isInteractive}
              placeholder={field.placeholder}
              value={(formData?.[field.name] as string) ?? ""}
              onChange={(e) =>
                handleChange(
                  field.name,
                  field.type === "number" ? parseFloat(e.target.value) || "" : e.target.value,
                )
              }
              className="w-full px-3 py-2 border rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 disabled:bg-gray-50"
            />
          )}
        </div>
      ))}

      {config.fields.length === 0 && (
        <p className="text-sm text-gray-400 text-center py-4">
          No input fields configured yet
        </p>
      )}
    </div>
  );
}
