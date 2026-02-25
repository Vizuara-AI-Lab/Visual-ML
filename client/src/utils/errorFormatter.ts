/**
 * Frontend utilities for displaying structured error messages
 * from the ML pipeline backend.
 *
 * The backend returns errors as:
 * {
 *   friendly_message: string,
 *   suggestion: string,
 *   technical_detail: string,
 *   error_code: string,
 *   node_type: string | null,
 * }
 */

export interface StructuredError {
  friendly_message: string;
  suggestion: string;
  technical_detail: string;
  error_code: string;
  node_type: string | null;
}

/** Type guard: is this error a structured dict from the backend? */
export function isStructuredError(error: unknown): error is StructuredError {
  if (!error || typeof error !== "object") return false;
  const obj = error as Record<string, unknown>;
  return (
    typeof obj.friendly_message === "string" &&
    typeof obj.suggestion === "string" &&
    typeof obj.error_code === "string"
  );
}

/** Extract the student-friendly message, with fallback for legacy string errors. */
export function getFriendlyMessage(error: unknown): string {
  if (isStructuredError(error)) return error.friendly_message;
  if (typeof error === "string") return error;
  if (error && typeof error === "object") {
    const obj = error as Record<string, unknown>;
    // Legacy format: { details: { reason } }
    if (obj.details && typeof obj.details === "object") {
      const details = obj.details as Record<string, unknown>;
      if (typeof details.reason === "string") return details.reason;
    }
    // Legacy format: { message: "Node execution failed [type]: reason" }
    if (typeof obj.message === "string") {
      const match = obj.message.match(/\[.*?\]: (.+)/);
      return match ? match[1] : obj.message;
    }
    if (typeof obj.error === "string") return obj.error;
    return JSON.stringify(error);
  }
  return "Unknown error";
}

/** Extract the fix suggestion (empty string if none). */
export function getErrorSuggestion(error: unknown): string {
  if (isStructuredError(error)) return error.suggestion;
  return "";
}

/** Extract the raw technical detail for collapsible display. */
export function getTechnicalDetail(error: unknown): string {
  if (isStructuredError(error)) return error.technical_detail;
  if (typeof error === "string") return error;
  if (error && typeof error === "object") {
    return JSON.stringify(error, null, 2);
  }
  return String(error);
}
