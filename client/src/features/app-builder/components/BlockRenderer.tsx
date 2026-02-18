/**
 * BlockRenderer — Switch component that routes to the correct block component.
 * Supports "edit" (builder canvas), "preview", and "live" (public) modes.
 */

import type { AppBlock, AppTheme, ExecuteAppResponse } from "../types/appBuilder";
import HeroBlock from "./blocks/HeroBlock";
import TextBlock from "./blocks/TextBlock";
import FileUploadBlock from "./blocks/FileUploadBlock";
import InputFieldsBlock from "./blocks/InputFieldsBlock";
import SubmitButtonBlock from "./blocks/SubmitButtonBlock";
import ResultsDisplayBlock from "./blocks/ResultsDisplayBlock";
import MetricsCardBlock from "./blocks/MetricsCardBlock";
import DividerBlock from "./blocks/DividerBlock";
import ImageBlock from "./blocks/ImageBlock";

export type BlockMode = "edit" | "preview" | "live";

export interface BlockRenderProps {
  block: AppBlock;
  mode: BlockMode;
  theme: AppTheme;
  formData?: Record<string, unknown>;
  results?: ExecuteAppResponse | null;
  isExecuting?: boolean;
  onFieldChange?: (name: string, value: unknown, nodeId?: string, nodeConfigKey?: string) => void;
  onFileUpload?: (base64: string, filename: string, nodeId?: string) => void;
  onSubmit?: () => void;
}

export default function BlockRenderer(props: BlockRenderProps) {
  const { block } = props;

  switch (block.type) {
    case "hero":
      return <HeroBlock {...props} />;
    case "text":
      return <TextBlock {...props} />;
    case "file_upload":
      return <FileUploadBlock {...props} />;
    case "input_fields":
      return <InputFieldsBlock {...props} />;
    case "submit_button":
      return <SubmitButtonBlock {...props} />;
    case "results_display":
      return <ResultsDisplayBlock {...props} />;
    case "metrics_card":
      return <MetricsCardBlock {...props} />;
    case "divider":
      return <DividerBlock {...props} />;
    case "image":
      return <ImageBlock {...props} />;
    default:
      return (
        <div className="p-4 border border-dashed border-red-300 rounded-lg text-red-500 text-sm">
          Unknown block type: {block.type}
        </div>
      );
  }
}
