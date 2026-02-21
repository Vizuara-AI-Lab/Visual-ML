/**
 * RMSE Score Result Visualization Component
 */

interface RMSEScoreProps {
  result: any;
}

export function RMSEScoreResult({ result }: RMSEScoreProps) {
  const rmseScore = result.rmse_score ?? 0;
  const displayValue = result.display_value ?? "N/A";
  const interpretation = result.interpretation ?? "No interpretation available";
  const modelInfo = result.model_info ?? {};

  return (
    <div className="space-y-6">
      {/* RMSE Score Display */}
      <div className="bg-linear-to-br from-purple-50 to-pink-50 border-2 border-purple-300 rounded-xl p-8 text-center">
        <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wide mb-2">
          RMSE (Root Mean Squared Error)
        </h3>
        <div className="text-6xl font-bold text-purple-600 mb-4">
          {displayValue}
        </div>
        <div className="inline-flex items-center gap-2 px-4 py-2 bg-white rounded-lg border border-purple-200">
          <span className="text-sm font-medium text-gray-700">
            {interpretation}
          </span>
        </div>
      </div>

      {/* Score Interpretation Guide */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <h4 className="text-md font-semibold text-gray-800 mb-4">
          Understanding RMSE Score
        </h4>
        <div className="space-y-3">
          <div className="flex items-center gap-3">
            <div className="w-24 h-8 bg-linear-to-r from-green-500 to-green-600 rounded flex items-center justify-center text-white text-xs font-semibold">
              &lt; 0.5
            </div>
            <span className="text-sm text-gray-700">
              Excellent - Very low error
            </span>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-24 h-8 bg-linear-to-r from-blue-500 to-blue-600 rounded flex items-center justify-center text-white text-xs font-semibold">
              0.5 - 2.0
            </div>
            <span className="text-sm text-gray-700">Good - Low error</span>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-24 h-8 bg-linear-to-r from-yellow-500 to-yellow-600 rounded flex items-center justify-center text-white text-xs font-semibold">
              2.0 - 5.0
            </div>
            <span className="text-sm text-gray-700">
              Moderate - Average error
            </span>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-24 h-8 bg-linear-to-r from-orange-500 to-orange-600 rounded flex items-center justify-center text-white text-xs font-semibold">
              5.0 - 10
            </div>
            <span className="text-sm text-gray-700">
              High - Significant error
            </span>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-24 h-8 bg-linear-to-r from-red-500 to-red-600 rounded flex items-center justify-center text-white text-xs font-semibold">
              &gt; 10
            </div>
            <span className="text-sm text-gray-700">
              Very High - Large error
            </span>
          </div>
        </div>
      </div>

      {/* Model Information */}
      {modelInfo && Object.keys(modelInfo).length > 0 && (
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-6">
          <h4 className="text-md font-semibold text-gray-800 mb-4">
            Model Information
          </h4>
          <div className="grid grid-cols-2 gap-4">
            {modelInfo.model_id && (
              <div className="col-span-2">
                <span className="text-sm font-medium text-gray-600">
                  Model ID:
                </span>
                <p className="text-sm text-gray-800 font-mono mt-1 break-all">
                  {modelInfo.model_id}
                </p>
              </div>
            )}
            {modelInfo.training_samples && (
              <div>
                <span className="text-sm font-medium text-gray-600">
                  Training Samples:
                </span>
                <p className="text-sm text-gray-800 mt-1">
                  {modelInfo.training_samples.toLocaleString()}
                </p>
              </div>
            )}
            {modelInfo.n_features && (
              <div>
                <span className="text-sm font-medium text-gray-600">
                  Features:
                </span>
                <p className="text-sm text-gray-800 mt-1">
                  {modelInfo.n_features}
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* What is RMSE? */}
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-6">
        <h4 className="text-md font-semibold text-purple-900 mb-3">
          📚 What is RMSE?
        </h4>
        <p className="text-sm text-purple-800 leading-relaxed">
          Root Mean Squared Error (RMSE) is the square root of MSE. It provides
          error measurements in the same units as the target variable, making it
          more interpretable than MSE. Lower values indicate better model
          performance.
        </p>
      </div>
    </div>
  );
}
