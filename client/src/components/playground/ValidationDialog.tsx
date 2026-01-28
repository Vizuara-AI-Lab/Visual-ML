/**
 * Validation Error Dialog Component
 *
 * Displays validation errors and warnings in a user-friendly modal
 */

import { X, AlertCircle, AlertTriangle } from "lucide-react";
import type { ValidationError } from "../../utils/validation";

interface ValidationDialogProps {
  errors: ValidationError[];
  onClose: () => void;
  onProceed?: () => void;
}

export const ValidationDialog = ({
  errors,
  onClose,
  onProceed,
}: ValidationDialogProps) => {
  const errorMessages = errors.filter((e) => e.type === "error");
  const warningMessages = errors.filter((e) => e.type === "warning");

  const hasErrors = errorMessages.length > 0;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-11/12 max-w-2xl flex flex-col max-h-[80vh]">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
          <div className="flex items-center gap-3">
            {hasErrors ? (
              <AlertCircle className="w-6 h-6 text-red-500" />
            ) : (
              <AlertTriangle className="w-6 h-6 text-yellow-500" />
            )}
            <h2 className="text-xl font-bold text-gray-800">
              {hasErrors ? "Pipeline Validation Failed" : "Pipeline Warnings"}
            </h2>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 text-2xl leading-none"
          >
            ×
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-6">
          {/* Errors */}
          {errorMessages.length > 0 && (
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-red-700 mb-3 flex items-center gap-2">
                <AlertCircle className="w-5 h-5" />
                Errors ({errorMessages.length})
              </h3>
              <div className="space-y-3">
                {errorMessages.map((error, index) => (
                  <div
                    key={index}
                    className="bg-red-50 border-l-4 border-red-500 p-4 rounded"
                  >
                    <p className="text-red-800 font-medium">{error.message}</p>
                    {error.suggestion && (
                      <p className="text-red-600 text-sm mt-2">
                        💡 {error.suggestion}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Warnings */}
          {warningMessages.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-yellow-700 mb-3 flex items-center gap-2">
                <AlertTriangle className="w-5 h-5" />
                Warnings ({warningMessages.length})
              </h3>
              <div className="space-y-3">
                {warningMessages.map((warning, index) => (
                  <div
                    key={index}
                    className="bg-yellow-50 border-l-4 border-yellow-500 p-4 rounded"
                  >
                    <p className="text-yellow-800 font-medium">
                      {warning.message}
                    </p>
                    {warning.suggestion && (
                      <p className="text-yellow-600 text-sm mt-2">
                        💡 {warning.suggestion}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Help text */}
          <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
            <p className="text-sm text-blue-800">
              <strong>Tip:</strong> View nodes (Table View, Chart View, etc.)
              must be connected directly to data source nodes (Upload File,
              Select Dataset, Load from URL). They cannot be connected to other
              view nodes or processing nodes.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-gray-200 flex justify-end gap-3">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300"
          >
            Close
          </button>
          {!hasErrors && onProceed && (
            <button
              onClick={onProceed}
              className="px-4 py-2 bg-yellow-500 text-white rounded-md hover:bg-yellow-600"
            >
              Proceed Anyway
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
