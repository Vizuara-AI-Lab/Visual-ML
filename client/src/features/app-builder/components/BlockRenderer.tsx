/**
 * BlockRenderer — Switch component that routes to the correct block component.
 * Supports "edit" (builder canvas), "preview", and "live" (public) modes.
 * Applies per-block custom styles when present.
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
import SpacerBlock from "./blocks/SpacerBlock";
import AlertBlock from "./blocks/AlertBlock";
import CodeBlock from "./blocks/CodeBlock";
import VideoEmbedBlock from "./blocks/VideoEmbedBlock";

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

  let content: React.ReactNode;

  switch (block.type) {
    case "hero":
      content = <HeroBlock {...props} />;
      break;
    case "text":
      content = <TextBlock {...props} />;
      break;
    case "file_upload":
      content = <FileUploadBlock {...props} />;
      break;
    case "input_fields":
      content = <InputFieldsBlock {...props} />;
      break;
    case "submit_button":
      content = <SubmitButtonBlock {...props} />;
      break;
    case "results_display":
      content = <ResultsDisplayBlock {...props} />;
      break;
    case "metrics_card":
      content = <MetricsCardBlock {...props} />;
      break;
    case "divider":
      content = <DividerBlock {...props} />;
      break;
    case "image":
      content = <ImageBlock {...props} />;
      break;
    case "spacer":
      content = <SpacerBlock {...props} />;
      break;
    case "alert":
      content = <AlertBlock {...props} />;
      break;
    case "code":
      content = <CodeBlock {...props} />;
      break;
    case "video_embed":
      content = <VideoEmbedBlock {...props} />;
      break;
    default:
      content = (
        <div className="p-4 border border-dashed border-red-300 rounded-lg text-red-500 text-sm">
          Unknown block type: {block.type}
        </div>
      );
  }

  if (block.style) {
    const s = block.style;
    return (
      <div
        style={{
          backgroundColor: s.backgroundColor || undefined,
          color: s.textColor || undefined,
          borderRadius: s.borderRadius != null ? `${s.borderRadius}px` : undefined,
          paddingLeft: s.paddingX != null ? `${s.paddingX}px` : undefined,
          paddingRight: s.paddingX != null ? `${s.paddingX}px` : undefined,
          paddingTop: s.paddingY != null ? `${s.paddingY}px` : undefined,
          paddingBottom: s.paddingY != null ? `${s.paddingY}px` : undefined,
          borderColor: s.borderColor || undefined,
          borderWidth: s.borderWidth != null ? `${s.borderWidth}px` : undefined,
          borderStyle: s.borderWidth ? "solid" : undefined,
        }}
      >
        {content}
      </div>
    );
  }

  return <>{content}</>;
}
