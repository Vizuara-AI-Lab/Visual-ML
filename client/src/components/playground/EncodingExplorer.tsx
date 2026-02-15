/**
 * EncodingExplorer - Interactive learning activities for Encoding node
 * Tabs: Results | Before & After | Encoding Map | Method Lab | Quiz
 */

import { useState, type ReactNode } from "react";
import {
  ArrowLeftRight,
  Map,
  FlaskConical,
  HelpCircle,
  CheckCircle,
  XCircle,
  Trophy,
  Info,
  ChevronRight,
  ClipboardList,
} from "lucide-react";

interface EncodingExplorerProps {
  result: any;
  renderResults: () => ReactNode;
}

type ExplorerTab = "results" | "before_after" | "encoding_map" | "method_lab" | "quiz";

export const EncodingExplorer = ({ result, renderResults }: EncodingExplorerProps) => {
  const [activeTab, setActiveTab] = useState<ExplorerTab>("results");

  const tabs: { id: ExplorerTab; label: string; icon: any; available: boolean }[] = [
    { id: "results", label: "Results", icon: ClipboardList, available: true },
    {
      id: "before_after",
      label: "Before & After",
      icon: ArrowLeftRight,
      available: !!result.encoding_before_after?.columns && Object.keys(result.encoding_before_after.columns).length > 0,
    },
    {
      id: "encoding_map",
      label: "Encoding Map",
      icon: Map,
      available: !!result.encoding_map && Object.keys(result.encoding_map).length > 0,
    },
    {
      id: "method_lab",
      label: "Method Lab",
      icon: FlaskConical,
      available: !!result.encoding_method_comparison && Object.keys(result.encoding_method_comparison).length > 0,
    },
    {
      id: "quiz",
      label: "Quiz",
      icon: HelpCircle,
      available: !!result.quiz_questions && result.quiz_questions.length > 0,
    },
  ];

  return (
    <div className="space-y-4">
      <div className="flex border-b border-gray-200 overflow-x-auto">
        {tabs
          .filter((t) => t.available)
          .map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
                  activeTab === tab.id
                    ? "border-amber-600 text-amber-600"
                    : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
      </div>

      {activeTab === "results" && renderResults()}
      {activeTab === "before_after" && result.encoding_before_after && (
        <BeforeAfterTab data={result.encoding_before_after} />
      )}
      {activeTab === "encoding_map" && result.encoding_map && (
        <EncodingMapTab data={result.encoding_map} />
      )}
      {activeTab === "method_lab" && result.encoding_method_comparison && (
        <MethodLabTab data={result.encoding_method_comparison} />
      )}
      {activeTab === "quiz" && result.quiz_questions && (
        <QuizTab questions={result.quiz_questions} />
      )}
    </div>
  );
};

// --- Before & After Tab ---
function BeforeAfterTab({ data }: { data: any }) {
  const columns = Object.keys(data.columns || {});
  const [selectedColumn, setSelectedColumn] = useState(columns[0] || "");

  const colData = data.columns?.[selectedColumn];

  return (
    <div className="space-y-4">
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-amber-900">Before & After Encoding</h3>
        <p className="text-xs text-amber-700 mt-1">
          See how each category was transformed into numbers the model can understand.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        {columns.map((col) => (
          <button
            key={col}
            onClick={() => setSelectedColumn(col)}
            className={`px-3 py-1.5 text-sm rounded-md border transition-colors ${
              selectedColumn === col
                ? "bg-amber-600 text-white border-amber-600"
                : "bg-white text-gray-700 border-gray-300 hover:border-amber-400"
            }`}
          >
            {col}
          </button>
        ))}
      </div>

      {colData && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <span className="px-2 py-0.5 text-xs font-semibold bg-amber-100 text-amber-800 rounded-full border border-amber-300">
              {colData.method === "onehot" ? "One-Hot" : "Label"}
            </span>
          </div>

          <div className="space-y-2">
            {colData.samples?.map((sample: any, i: number) => (
              <div key={i} className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
                <div className="flex-shrink-0 px-3 py-2 bg-orange-100 text-orange-800 rounded-md text-sm font-mono font-medium border border-orange-200">
                  {sample.original}
                </div>
                <ChevronRight className="w-4 h-4 text-gray-400 flex-shrink-0" />
                {colData.method === "onehot" && typeof sample.encoded === "object" ? (
                  <div className="flex flex-wrap gap-1.5">
                    {Object.entries(sample.encoded).map(([col, val]: [string, any]) => (
                      <span
                        key={col}
                        className={`px-2 py-1 text-xs font-mono rounded border ${
                          val === 1
                            ? "bg-green-100 text-green-800 border-green-300 font-bold"
                            : "bg-gray-100 text-gray-500 border-gray-200"
                        }`}
                      >
                        {col.split("_").pop()}={val}
                      </span>
                    ))}
                  </div>
                ) : (
                  <span className="px-3 py-2 bg-blue-100 text-blue-800 rounded-md text-sm font-mono font-bold border border-blue-200">
                    {sample.encoded}
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
        <div className="flex items-start gap-2">
          <Info className="w-4 h-4 text-blue-600 mt-0.5" />
          <p className="text-xs text-blue-700">
            <strong>One-Hot</strong> creates a new column for each category (1 = yes, 0 = no).{" "}
            <strong>Label</strong> assigns a single number to each category. The model sees numbers,
            not text — encoding bridges that gap!
          </p>
        </div>
      </div>
    </div>
  );
}

// --- Encoding Map Tab ---
function EncodingMapTab({ data }: { data: any }) {
  const columns = Object.keys(data);
  const [selectedColumn, setSelectedColumn] = useState(columns[0] || "");
  const [sortBy, setSortBy] = useState<"name" | "count">("count");

  const colData = data[selectedColumn];
  const mapping = colData?.mapping || [];
  const sorted = [...mapping].sort((a: any, b: any) =>
    sortBy === "count" ? b.count - a.count : a.category.localeCompare(b.category)
  );

  return (
    <div className="space-y-4">
      <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-indigo-900">Encoding Map</h3>
        <p className="text-xs text-indigo-700 mt-1">
          Complete lookup table showing how each category was encoded into numbers.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        {columns.map((col) => (
          <button
            key={col}
            onClick={() => setSelectedColumn(col)}
            className={`px-3 py-1.5 text-sm rounded-md border transition-colors ${
              selectedColumn === col
                ? "bg-indigo-600 text-white border-indigo-600"
                : "bg-white text-gray-700 border-gray-300 hover:border-indigo-400"
            }`}
          >
            {col}
          </button>
        ))}
      </div>

      {colData && (
        <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
          <div className="flex items-center justify-between px-4 py-2 bg-gray-50 border-b border-gray-200">
            <span className="text-xs font-medium text-gray-600">
              {colData.method === "onehot" ? "One-Hot" : "Label"} Encoding — {mapping.length} categories
            </span>
            <div className="flex gap-1">
              <button
                onClick={() => setSortBy("name")}
                className={`px-2 py-0.5 text-xs rounded ${sortBy === "name" ? "bg-indigo-600 text-white" : "bg-white text-gray-600 border border-gray-300"}`}
              >
                Name
              </button>
              <button
                onClick={() => setSortBy("count")}
                className={`px-2 py-0.5 text-xs rounded ${sortBy === "count" ? "bg-indigo-600 text-white" : "bg-white text-gray-600 border border-gray-300"}`}
              >
                Frequency
              </button>
            </div>
          </div>
          <table className="min-w-full text-sm">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="px-4 py-2 text-left text-xs font-semibold text-gray-700">Category</th>
                <th className="px-4 py-2 text-left text-xs font-semibold text-gray-700">
                  {colData.method === "onehot" ? "Encoded Column" : "Value"}
                </th>
                <th className="px-4 py-2 text-right text-xs font-semibold text-gray-700">Count</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {sorted.map((item: any, idx: number) => (
                <tr key={idx} className="hover:bg-gray-50">
                  <td className="px-4 py-2 font-medium text-gray-900">{item.category}</td>
                  <td className="px-4 py-2 font-mono text-gray-700">
                    {item.encoded_column || item.value}
                  </td>
                  <td className="px-4 py-2 text-right text-gray-600">{item.count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// --- Method Lab Tab ---
function MethodLabTab({ data }: { data: any }) {
  const columns = Object.keys(data);
  const [selectedColumn, setSelectedColumn] = useState(columns[0] || "");

  const colData = data[selectedColumn];

  return (
    <div className="space-y-4">
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-purple-900">Method Lab</h3>
        <p className="text-xs text-purple-700 mt-1">
          Compare what One-Hot and Label encoding would produce for each column.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        {columns.map((col) => (
          <button
            key={col}
            onClick={() => setSelectedColumn(col)}
            className={`px-3 py-1.5 text-sm rounded-md border transition-colors ${
              selectedColumn === col
                ? "bg-purple-600 text-white border-purple-600"
                : "bg-white text-gray-700 border-gray-300 hover:border-purple-400"
            }`}
          >
            {col} ({data[col].unique_count} categories)
          </button>
        ))}
      </div>

      {colData && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* One-Hot Card */}
            <div className={`border-2 rounded-lg p-4 ${colData.applied_method === "onehot" ? "border-green-500 bg-green-50/30" : "border-gray-200 bg-white"}`}>
              <div className="flex items-center justify-between mb-3">
                <h4 className="text-sm font-semibold text-gray-900">One-Hot Encoding</h4>
                {colData.applied_method === "onehot" && (
                  <span className="px-2 py-0.5 text-xs font-semibold bg-green-100 text-green-800 rounded-full border border-green-300">Applied</span>
                )}
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">New columns:</span>
                  <span className="font-semibold">{colData.onehot_result?.new_columns_count}</span>
                </div>
                {colData.onehot_result?.sample_row && (
                  <div className="bg-gray-50 rounded p-2">
                    <span className="text-xs text-gray-500 block mb-1">Sample row:</span>
                    <div className="flex flex-wrap gap-1">
                      {Object.entries(colData.onehot_result.sample_row).map(([k, v]: [string, any]) => (
                        <span key={k} className={`px-1.5 py-0.5 text-xs font-mono rounded ${v === 1 ? "bg-green-100 text-green-800" : "bg-gray-100 text-gray-500"}`}>
                          {k.split("_").pop()}={v}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                <div className="pt-2 border-t border-gray-200">
                  <p className="text-xs text-green-700">+ {colData.onehot_result?.pros}</p>
                  <p className="text-xs text-red-600 mt-1">- {colData.onehot_result?.cons}</p>
                </div>
              </div>
            </div>

            {/* Label Card */}
            <div className={`border-2 rounded-lg p-4 ${colData.applied_method === "label" ? "border-green-500 bg-green-50/30" : "border-gray-200 bg-white"}`}>
              <div className="flex items-center justify-between mb-3">
                <h4 className="text-sm font-semibold text-gray-900">Label Encoding</h4>
                {colData.applied_method === "label" && (
                  <span className="px-2 py-0.5 text-xs font-semibold bg-green-100 text-green-800 rounded-full border border-green-300">Applied</span>
                )}
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">New columns:</span>
                  <span className="font-semibold">0 (in-place)</span>
                </div>
                {colData.label_result?.sample_mapping && (
                  <div className="bg-gray-50 rounded p-2">
                    <span className="text-xs text-gray-500 block mb-1">Mapping:</span>
                    <div className="flex flex-wrap gap-1.5">
                      {Object.entries(colData.label_result.sample_mapping).map(([cat, val]: [string, any]) => (
                        <span key={cat} className="px-2 py-0.5 text-xs font-mono bg-blue-50 text-blue-800 rounded border border-blue-200">
                          {cat} → {val}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                <div className="pt-2 border-t border-gray-200">
                  <p className="text-xs text-green-700">+ {colData.label_result?.pros}</p>
                  <p className="text-xs text-red-600 mt-1">- {colData.label_result?.cons}</p>
                </div>
              </div>
            </div>
          </div>

          {colData.recommendation && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
              <div className="flex items-start gap-2">
                <Info className="w-4 h-4 text-blue-600 mt-0.5" />
                <p className="text-xs text-blue-700">{colData.recommendation}</p>
              </div>
            </div>
          )}
        </>
      )}

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
        <div className="flex items-start gap-2">
          <Info className="w-4 h-4 text-blue-600 mt-0.5" />
          <p className="text-xs text-blue-700">
            <strong>Nominal</strong> categories (colors, cities) have no order — use One-Hot.{" "}
            <strong>Ordinal</strong> categories (small/medium/large) have order — Label encoding can work.
            The method with the <span className="text-green-700 font-semibold">green border</span> is what was applied.
          </p>
        </div>
      </div>
    </div>
  );
}

// --- Quiz Tab ---
function QuizTab({ questions }: { questions: any[] }) {
  const [currentQ, setCurrentQ] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [answered, setAnswered] = useState(false);
  const [score, setScore] = useState(0);
  const [answers, setAnswers] = useState<(number | null)[]>(new Array(questions.length).fill(null));
  const [showResults, setShowResults] = useState(false);

  const question = questions[currentQ];

  const handleSelect = (optionIdx: number) => {
    if (answered) return;
    setSelectedAnswer(optionIdx);
    setAnswered(true);
    const newAnswers = [...answers];
    newAnswers[currentQ] = optionIdx;
    setAnswers(newAnswers);
    if (optionIdx === question.correct_answer) setScore((s) => s + 1);
  };

  const handleNext = () => {
    if (currentQ < questions.length - 1) {
      setCurrentQ((q) => q + 1);
      setSelectedAnswer(null);
      setAnswered(false);
    } else {
      setShowResults(true);
    }
  };

  const handleRetry = () => {
    setCurrentQ(0);
    setSelectedAnswer(null);
    setAnswered(false);
    setScore(0);
    setAnswers(new Array(questions.length).fill(null));
    setShowResults(false);
  };

  if (showResults) {
    const percentage = Math.round((score / questions.length) * 100);
    return (
      <div className="space-y-4">
        <div className={`text-center py-8 rounded-lg border-2 ${percentage >= 80 ? "bg-green-50 border-green-300" : percentage >= 50 ? "bg-yellow-50 border-yellow-300" : "bg-red-50 border-red-300"}`}>
          <Trophy className={`w-12 h-12 mx-auto mb-3 ${percentage >= 80 ? "text-green-600" : percentage >= 50 ? "text-yellow-600" : "text-red-600"}`} />
          <div className="text-3xl font-bold text-gray-900 mb-2">{score}/{questions.length}</div>
          <p className="text-sm text-gray-700">
            {percentage >= 80 ? "Excellent understanding!" : percentage >= 50 ? "Good effort! Review the explanations below." : "Keep learning! Try again after reviewing."}
          </p>
        </div>
        <button onClick={handleRetry} className="w-full py-2 bg-amber-600 text-white rounded-lg hover:bg-amber-700 text-sm font-medium">Try Again</button>
        <div className="space-y-3">
          {questions.map((q: any, qi: number) => (
            <div key={qi} className="bg-gray-50 border border-gray-200 rounded-lg p-3 text-sm">
              <p className="font-medium text-gray-800 mb-1">{q.question}</p>
              <p className={`text-xs ${answers[qi] === q.correct_answer ? "text-green-700" : "text-red-700"}`}>
                Your answer: {q.options[answers[qi] ?? 0]} {answers[qi] === q.correct_answer ? "(Correct)" : "(Incorrect)"}
              </p>
              <p className="text-xs text-blue-700 mt-1">{q.explanation}</p>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-amber-900">Knowledge Check</h3>
        <p className="text-xs text-amber-700 mt-1">Test your understanding of encoding using questions from YOUR dataset.</p>
      </div>

      <div className="flex justify-center gap-2">
        {questions.map((_: any, i: number) => (
          <div key={i} className={`w-3 h-3 rounded-full transition-all ${i === currentQ ? "bg-amber-600 scale-125" : i < currentQ ? (answers[i] === questions[i].correct_answer ? "bg-green-500" : "bg-red-400") : "bg-gray-300"}`} />
        ))}
      </div>

      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <div className="text-xs text-gray-500 mb-2">Question {currentQ + 1} of {questions.length}</div>
        <p className="text-sm font-medium text-gray-900 mb-4">{question.question}</p>
        <div className="space-y-2">
          {question.options.map((option: string, oi: number) => {
            let optionClass = "bg-white border-gray-300 hover:border-amber-400 cursor-pointer";
            if (answered) {
              if (oi === question.correct_answer) optionClass = "bg-green-50 border-green-500 text-green-900";
              else if (oi === selectedAnswer && oi !== question.correct_answer) optionClass = "bg-red-50 border-red-500 text-red-900";
              else optionClass = "bg-gray-50 border-gray-200 text-gray-400 cursor-default";
            }
            return (
              <button key={oi} onClick={() => handleSelect(oi)} disabled={answered} className={`w-full text-left p-3 rounded-lg border-2 text-sm transition-colors ${optionClass}`}>
                <span className="font-mono text-xs text-gray-400 mr-2">{String.fromCharCode(65 + oi)}.</span>
                {option}
                {answered && oi === question.correct_answer && <CheckCircle className="w-4 h-4 text-green-600 inline ml-2" />}
                {answered && oi === selectedAnswer && oi !== question.correct_answer && <XCircle className="w-4 h-4 text-red-600 inline ml-2" />}
              </button>
            );
          })}
        </div>
        {answered && (
          <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-xs text-blue-800">{question.explanation}</p>
          </div>
        )}
        {answered && (
          <button onClick={handleNext} className="mt-4 w-full py-2 bg-amber-600 text-white rounded-lg hover:bg-amber-700 text-sm font-medium flex items-center justify-center gap-2">
            {currentQ < questions.length - 1 ? (<>Next Question <ChevronRight className="w-4 h-4" /></>) : "See Results"}
          </button>
        )}
      </div>
    </div>
  );
}
