"use client";

import { useState, useEffect, useCallback } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Candidate interfaces ──────────────────────────────────

interface CandidateListItem {
  id: number;
  first_name: string;
  last_name: string;
  email: string;
  status: string;
  global_score: number | null;
  recommendation: string | null;
  match_percent: number;
  job_title: string | null; // Added
  created_at: string | null;
}

interface QuestionResult {
  question_id: string;
  question_text: string;
  answer_summary: string;
  score: number;
  evidence: string[];
}

interface KeywordCoverage {
  keyword: string;
  evidence: string;
  score: number;
}

interface CandidateDetail extends CandidateListItem {
  cv_text: string;
  interview: {
    transcript: string;
    summary: string;
    questions: QuestionResult[];
    keyword_coverage: KeywordCoverage[];
    global_score: number;
    recommendation: string;
    red_flags: string[];
  } | null;
}

// ── Job Config interfaces ─────────────────────────────────

interface QuestionItem {
  question: string;
  expected_themes: string[];
}

interface QuestionDraft {
  question: string;
  expectedThemesInput: string;
}

interface JobConfigItem {
  id: number;
  title: string;
  description: string;
  keywords: string[];
  mandatory_questions: Array<QuestionItem | string>;
  match_threshold: number;
  max_interview_minutes: number;
  is_active: boolean;
  candidate_count: number;
  created_at: string | null;
}

// ── Constants ─────────────────────────────────────────────

const BADGE_COLORS: Record<string, string> = {
  strong_yes: "bg-green-100 text-green-800",
  yes: "bg-emerald-100 text-emerald-800",
  maybe: "bg-yellow-100 text-yellow-800",
  no: "bg-red-100 text-red-800",
};

const STATUS_COLORS: Record<string, string> = {
  interview_completed: "bg-blue-100 text-blue-800",
  accepted: "bg-green-100 text-green-800",
  rejected_after_interview: "bg-red-100 text-red-800",
};

// ── Config Form Component ─────────────────────────────────

function ConfigForm({
  initial,
  onSave,
  onCancel,
}: {
  initial: JobConfigItem | null;
  onSave: (data: {
    title: string;
    description: string;
    keywords: string[];
    mandatory_questions: QuestionItem[];
    match_threshold: number;
    max_interview_minutes: number;
  }) => void;
  onCancel: () => void;
}) {
  const normalizeQuestions = (rawQuestions: Array<QuestionItem | string> | undefined): QuestionDraft[] => {
    if (!rawQuestions || rawQuestions.length === 0) {
      return [{ question: "", expectedThemesInput: "" }];
    }

    return rawQuestions.map((raw) => {
      if (typeof raw === "string") {
        return { question: raw, expectedThemesInput: "" };
      }
      return {
        question: raw.question ?? "",
        expectedThemesInput: Array.isArray(raw.expected_themes) ? raw.expected_themes.join(", ") : "",
      };
    });
  };

  const [title, setTitle] = useState(initial?.title || "");
  const [description, setDescription] = useState(initial?.description || "");
  const [keywordInput, setKeywordInput] = useState(initial?.keywords.join(", ") || "");
  const [threshold, setThreshold] = useState(initial ? initial.match_threshold * 100 : 30);
  const [maxMinutes, setMaxMinutes] = useState(initial?.max_interview_minutes || 8);
  const [questions, setQuestions] = useState<QuestionDraft[]>(normalizeQuestions(initial?.mandatory_questions));
  const [saving, setSaving] = useState(false);

  const addQuestion = () => setQuestions([...questions, { question: "", expectedThemesInput: "" }]);

  const removeQuestion = (idx: number) => {
    if (questions.length <= 1) return;
    setQuestions(questions.filter((_, i) => i !== idx));
  };

  const updateQuestion = (idx: number, field: "question" | "expectedThemesInput", value: string) => {
    const updated = [...questions];
    updated[idx] = { ...updated[idx], [field]: value };
    setQuestions(updated);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    const keywords = keywordInput.split(",").map(k => k.trim()).filter(Boolean);
    onSave({
      title,
      description,
      keywords,
      mandatory_questions: questions
        .filter(q => q.question.trim())
        .map(q => ({
          question: q.question.trim(),
          expected_themes: q.expectedThemesInput.split(",").map(s => s.trim()).filter(Boolean),
        })),
      match_threshold: threshold / 100,
      max_interview_minutes: maxMinutes,
    });
    setSaving(false);
  };

  return (
    <form onSubmit={handleSubmit} className="bg-white rounded-xl border border-slate-200 p-6 space-y-5">
      <h3 className="text-lg font-bold text-slate-900">
        {initial ? "Edit Configuration" : "New Configuration"}
      </h3>

      <div>
        <label className="block text-sm font-medium text-slate-700 mb-1">Job Title</label>
        <input
          type="text"
          value={title}
          onChange={e => setTitle(e.target.value)}
          required
          className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          placeholder="e.g. Full-Stack Developer"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-slate-700 mb-1">Job Description</label>
        <textarea
          value={description}
          onChange={e => setDescription(e.target.value)}
          required
          rows={3}
          className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          placeholder="Describe the role, responsibilities, and requirements..."
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-slate-700 mb-1">
          Keywords <span className="text-slate-400 font-normal">(comma-separated)</span>
        </label>
        <textarea
          value={keywordInput}
          onChange={e => setKeywordInput(e.target.value)}
          rows={2}
          className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          placeholder="react, python, docker, aws, sql..."
        />
        {keywordInput && (
          <div className="flex flex-wrap gap-1 mt-2">
            {keywordInput.split(",").map((k) => k.trim()).filter(Boolean).map((k, i) => (
              <span key={i} className="px-2 py-0.5 bg-blue-100 text-blue-800 text-xs rounded-full">{k}</span>
            ))}
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">
            Match Threshold <span className="text-slate-400 font-normal">({threshold}%)</span>
          </label>
          <input
            type="range"
            min={0}
            max={100}
            step={5}
            value={threshold}
            onChange={e => setThreshold(Number(e.target.value))}
            className="w-full"
          />
          <p className="text-xs text-slate-400 mt-1">
            Minimum % of keywords a CV must match to get an interview invite
          </p>
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">Max Interview Minutes</label>
          <input
            type="number"
            min={1}
            max={60}
            value={maxMinutes}
            onChange={e => setMaxMinutes(Number(e.target.value))}
            className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
      </div>

      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="block text-sm font-medium text-slate-700">Interview Questions</label>
          <button
            type="button"
            onClick={addQuestion}
            className="text-sm text-blue-600 hover:text-blue-800 font-medium"
          >
            + Add Question
          </button>
        </div>
        <div className="space-y-3">
          {questions.map((q, i) => (
            <div key={i} className="border border-slate-200 rounded-lg p-3 bg-slate-50">
              <div className="flex items-start gap-2">
                <span className="text-xs font-bold text-slate-400 mt-2 w-6 shrink-0">Q{i + 1}</span>
                <div className="flex-1 space-y-2">
                  <input
                    type="text"
                    value={q.question}
                    onChange={e => updateQuestion(i, "question", e.target.value)}
                    placeholder="Enter interview question..."
                    className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                  <input
                    type="text"
                    value={q.expectedThemesInput}
                    onChange={e => updateQuestion(i, "expectedThemesInput", e.target.value)}
                    placeholder="Expected themes (comma-separated, optional)..."
                    className="w-full border border-slate-200 rounded-lg px-3 py-1.5 text-xs text-slate-600 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
                {questions.length > 1 && (
                  <button
                    type="button"
                    onClick={() => removeQuestion(i)}
                    className="text-slate-400 hover:text-red-500 mt-2 text-lg leading-none"
                  >
                    &times;
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="flex gap-3 pt-2">
        <button
          type="submit"
          disabled={saving}
          className="bg-blue-600 text-white px-6 py-2 rounded-lg font-semibold hover:bg-blue-700 disabled:opacity-50"
        >
          {saving ? "Saving..." : initial ? "Update Configuration" : "Create Configuration"}
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="border border-slate-300 text-slate-700 px-6 py-2 rounded-lg font-semibold hover:bg-slate-50"
        >
          Cancel
        </button>
      </div>
    </form>
  );
}

// ── Main Dashboard Page ───────────────────────────────────

export default function DashboardPage() {
  const [activeTab, setActiveTab] = useState<"candidates" | "configs">("candidates");

  // Candidate state
  const [candidates, setCandidates] = useState<CandidateListItem[]>([]);
  const [selected, setSelected] = useState<CandidateDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [deciding, setDeciding] = useState(false);

  // Config state
  const [configs, setConfigs] = useState<JobConfigItem[]>([]);
  const [configsLoading, setConfigsLoading] = useState(true);
  const [showConfigForm, setShowConfigForm] = useState(false);
  const [editingConfig, setEditingConfig] = useState<JobConfigItem | null>(null);

  // ── Data fetching ─────────────────────────────────────

  const fetchCandidates = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/candidates`);
      const data = await res.json();
      setCandidates(data);
    } catch {
      console.error("Failed to fetch candidates");
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchConfigs = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/job-configs`);
      const data = await res.json();
      setConfigs(data);
    } catch {
      console.error("Failed to fetch configs");
    } finally {
      setConfigsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchCandidates();
    fetchConfigs();
  }, [fetchCandidates, fetchConfigs]);

  // ── Candidate actions ─────────────────────────────────

  const selectCandidate = async (id: number) => {
    try {
      const res = await fetch(`${API}/api/candidate/${id}`);
      const data = await res.json();
      setSelected(data);
    } catch {
      console.error("Failed to fetch candidate detail");
    }
  };

  const makeDecision = async (decision: "accept" | "reject") => {
    if (!selected) return;
    setDeciding(true);
    try {
      await fetch(`${API}/api/candidate/${selected.id}/decision`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ decision }),
      });
      await fetchCandidates();
      setSelected(prev => prev ? { ...prev, status: decision === "accept" ? "accepted" : "rejected_after_interview" } : null);
    } catch {
      console.error("Decision failed");
    } finally {
      setDeciding(false);
    }
  };

  // ── Config actions ────────────────────────────────────

  const saveConfig = async (data: {
    title: string;
    description: string;
    keywords: string[];
    mandatory_questions: QuestionItem[];
    match_threshold: number;
    max_interview_minutes: number;
  }) => {
    try {
      if (editingConfig) {
        await fetch(`${API}/api/job-configs/${editingConfig.id}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(data),
        });
      } else {
        await fetch(`${API}/api/job-configs`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(data),
        });
      }
      setShowConfigForm(false);
      setEditingConfig(null);
      fetchConfigs();
    } catch {
      console.error("Failed to save config");
    }
  };

  const activateConfig = async (id: number) => {
    try {
      await fetch(`${API}/api/job-configs/${id}/activate`, { method: "POST" });
      fetchConfigs();
    } catch {
      console.error("Failed to activate config");
    }
  };

  const deleteConfig = async (id: number) => {
    if (!confirm("Delete this configuration?")) return;
    try {
      const res = await fetch(`${API}/api/job-configs/${id}`, { method: "DELETE" });
      if (!res.ok) {
        const err = await res.json();
        alert(err.detail || "Cannot delete this configuration");
        return;
      }
      fetchConfigs();
    } catch {
      console.error("Failed to delete config");
    }
  };

  // ── Helpers ───────────────────────────────────────────

  const scoreColor = (score: number | null) => {
    if (score === null) return "text-slate-400";
    if (score >= 75) return "text-green-600";
    if (score >= 50) return "text-yellow-600";
    return "text-red-600";
  };

  // ── Render ────────────────────────────────────────────

  return (
    <div className="flex flex-col h-[calc(100vh-57px)]">
      {/* Tab Bar */}
      <div className="flex border-b border-slate-200 bg-white px-6">
        <button
          onClick={() => setActiveTab("candidates")}
          className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${activeTab === "candidates"
              ? "border-blue-600 text-blue-600"
              : "border-transparent text-slate-500 hover:text-slate-700"
            }`}
        >
          Candidates
        </button>
        <button
          onClick={() => setActiveTab("configs")}
          className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${activeTab === "configs"
              ? "border-blue-600 text-blue-600"
              : "border-transparent text-slate-500 hover:text-slate-700"
            }`}
        >
          Job Configurations
        </button>
      </div>

      {/* Content */}
      {activeTab === "candidates" ? (
        <div className="flex flex-1 overflow-hidden">
          {/* Candidate List */}
          <div className="w-96 border-r border-slate-200 bg-white overflow-y-auto">
            <div className="p-4 border-b border-slate-200">
              <h1 className="text-xl font-bold text-slate-900">Candidates</h1>
              <p className="text-sm text-slate-500">{candidates.length} interviewed</p>
            </div>
            {loading ? (
              <div className="p-8 text-center text-slate-400">Loading...</div>
            ) : candidates.length === 0 ? (
              <div className="p-8 text-center text-slate-400">No candidates yet</div>
            ) : (
              candidates.map(c => (
                <button
                  key={c.id}
                  onClick={() => selectCandidate(c.id)}
                  className={`w-full text-left p-4 border-b border-slate-100 hover:bg-slate-50 transition-colors ${selected?.id === c.id ? "bg-blue-50 border-l-4 border-l-blue-500" : ""
                    }`}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-semibold text-slate-900">{c.first_name} {c.last_name}</p>
                      <p className="text-xs font-medium text-slate-500 mb-1">{c.job_title || "Unknown Position"}</p>
                      <p className="text-sm text-slate-400">{c.email}</p>
                    </div>
                    <div className="text-right">
                      <p className={`text-2xl font-bold ${scoreColor(c.global_score)}`}>
                        {c.global_score !== null ? c.global_score : "-"}
                      </p>
                      {c.recommendation && (
                        <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${BADGE_COLORS[c.recommendation] || "bg-slate-100 text-slate-700"}`}>
                          {c.recommendation.replace("_", " ")}
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="mt-1">
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${STATUS_COLORS[c.status] || "bg-slate-100 text-slate-700"}`}>
                      {c.status.replace(/_/g, " ")}
                    </span>
                  </div>
                </button>
              ))
            )}
          </div>

          {/* Detail Panel */}
          <div className="flex-1 overflow-y-auto bg-slate-50">
            {selected ? (
              <div className="p-6 max-w-4xl">
                {/* Header */}
                <div className="flex items-start justify-between mb-6">
                  <div>
                    <h2 className="text-2xl font-bold text-slate-900">
                      {selected.first_name} {selected.last_name}
                    </h2>
                    <p className="font-medium text-slate-600 mb-1">{selected.job_title || "Unknown Position"}</p>
                    <p className="text-sm text-slate-500">{selected.email}</p>
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium mt-1 inline-block ${STATUS_COLORS[selected.status] || "bg-slate-100 text-slate-700"}`}>
                      {selected.status.replace(/_/g, " ")}
                    </span>
                  </div>
                  {selected.status === "interview_completed" && (
                    <div className="flex gap-2">
                      <button
                        onClick={() => makeDecision("accept")}
                        disabled={deciding}
                        className="bg-green-600 text-white px-6 py-2 rounded-lg font-semibold hover:bg-green-700 disabled:opacity-50"
                      >
                        Accept
                      </button>
                      <button
                        onClick={() => makeDecision("reject")}
                        disabled={deciding}
                        className="bg-red-600 text-white px-6 py-2 rounded-lg font-semibold hover:bg-red-700 disabled:opacity-50"
                      >
                        Reject
                      </button>
                    </div>
                  )}
                </div>

                {selected.interview ? (
                  <>
                    {/* Score Overview */}
                    <div className="grid grid-cols-3 gap-4 mb-6">
                      <div className="bg-white rounded-xl p-4 border border-slate-200">
                        <p className="text-sm text-slate-500">Global Score</p>
                        <p className={`text-3xl font-bold ${scoreColor(selected.interview.global_score)}`}>
                          {selected.interview.global_score}/100
                        </p>
                      </div>
                      <div className="bg-white rounded-xl p-4 border border-slate-200">
                        <p className="text-sm text-slate-500">Recommendation</p>
                        <p className="text-lg font-semibold text-slate-900 capitalize">
                          {selected.interview.recommendation?.replace("_", " ")}
                        </p>
                      </div>
                      <div className="bg-white rounded-xl p-4 border border-slate-200">
                        <p className="text-sm text-slate-500">CV Match</p>
                        <p className="text-3xl font-bold text-blue-600">{selected.match_percent}%</p>
                      </div>
                    </div>

                    {/* Summary */}
                    <div className="bg-white rounded-xl p-4 border border-slate-200 mb-6">
                      <h3 className="font-semibold text-slate-900 mb-2">Summary</h3>
                      <p className="text-slate-700">{selected.interview.summary}</p>
                    </div>

                    {/* Red Flags */}
                    {selected.interview.red_flags && selected.interview.red_flags.length > 0 && (
                      <div className="bg-red-50 rounded-xl p-4 border border-red-200 mb-6">
                        <h3 className="font-semibold text-red-800 mb-2">Red Flags</h3>
                        <ul className="list-disc list-inside text-red-700 text-sm space-y-1">
                          {selected.interview.red_flags.map((flag, i) => (
                            <li key={i}>{flag}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Questions */}
                    <div className="bg-white rounded-xl p-4 border border-slate-200 mb-6">
                      <h3 className="font-semibold text-slate-900 mb-3">Question Scores</h3>
                      <div className="space-y-4">
                        {selected.interview.questions?.map((q, i) => (
                          <div key={i} className="border-b border-slate-100 pb-3 last:border-0 last:pb-0">
                            <div className="flex items-start justify-between">
                              <p className="font-medium text-slate-800 text-sm flex-1">{q.question_text}</p>
                              <span className={`ml-2 text-lg font-bold ${q.score >= 4 ? "text-green-600" : q.score >= 3 ? "text-yellow-600" : "text-red-600"}`}>
                                {q.score}/5
                              </span>
                            </div>
                            <p className="text-sm text-slate-600 mt-1">{q.answer_summary}</p>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Keywords */}
                    <div className="bg-white rounded-xl p-4 border border-slate-200 mb-6">
                      <h3 className="font-semibold text-slate-900 mb-3">Keyword Coverage</h3>
                      <div className="flex flex-wrap gap-2">
                        {selected.interview.keyword_coverage?.map((kw, i) => (
                          <span
                            key={i}
                            className={`px-3 py-1 rounded-full text-sm font-medium ${kw.score > 0 ? "bg-green-100 text-green-800" : "bg-slate-100 text-slate-500"
                              }`}
                            title={kw.evidence}
                          >
                            {kw.keyword}
                          </span>
                        ))}
                      </div>
                    </div>

                    {/* Transcript */}
                    <div className="bg-white rounded-xl p-4 border border-slate-200">
                      <h3 className="font-semibold text-slate-900 mb-2">Full Transcript</h3>
                      <pre className="text-sm text-slate-700 whitespace-pre-wrap max-h-96 overflow-y-auto font-sans">
                        {selected.interview.transcript}
                      </pre>
                    </div>
                  </>
                ) : (
                  <div className="text-center text-slate-400 mt-16">
                    No interview data available
                  </div>
                )}
              </div>
            ) : (
              <div className="flex items-center justify-center h-full text-slate-400">
                Select a candidate to view details
              </div>
            )}
          </div>
        </div>
      ) : (
        /* ── Configurations Tab ──────────────────────────── */
        <div className="flex-1 overflow-y-auto bg-slate-50 p-6">
          <div className="max-w-4xl mx-auto">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
              <div>
                <h1 className="text-2xl font-bold text-slate-900">Job Configurations</h1>
                <p className="text-sm text-slate-500">
                  Manage screening keywords, interview questions, and thresholds
                </p>
              </div>
              {!showConfigForm && (
                <button
                  onClick={() => { setEditingConfig(null); setShowConfigForm(true); }}
                  className="bg-blue-600 text-white px-5 py-2 rounded-lg font-semibold hover:bg-blue-700"
                >
                  + Add Configuration
                </button>
              )}
            </div>

            {/* Config Form */}
            {showConfigForm && (
              <div className="mb-6">
                <ConfigForm
                  initial={editingConfig}
                  onSave={saveConfig}
                  onCancel={() => { setShowConfigForm(false); setEditingConfig(null); }}
                />
              </div>
            )}

            {/* Config List */}
            {configsLoading ? (
              <div className="text-center text-slate-400 py-12">Loading configurations...</div>
            ) : configs.length === 0 ? (
              <div className="text-center text-slate-400 py-12">No configurations yet. Create one to get started.</div>
            ) : (
              <div className="space-y-4">
                {configs.map(cfg => (
                  <div
                    key={cfg.id}
                    className={`bg-white rounded-xl border p-5 ${cfg.is_active ? "border-green-300 ring-1 ring-green-200" : "border-slate-200"
                      }`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <h3 className="text-lg font-bold text-slate-900">{cfg.title}</h3>
                          {cfg.is_active && (
                            <span className="px-2 py-0.5 bg-green-100 text-green-800 text-xs rounded-full font-medium">
                              Active
                            </span>
                          )}
                        </div>
                        <p className="text-sm text-slate-600 mb-3 line-clamp-2">{cfg.description}</p>

                        <div className="flex flex-wrap gap-4 text-xs text-slate-500">
                          <span>{cfg.keywords.length} keywords</span>
                          <span>{cfg.mandatory_questions.length} questions</span>
                          <span>Threshold: {Math.round(cfg.match_threshold * 100)}%</span>
                          <span>Max: {cfg.max_interview_minutes} min</span>
                          <span>{cfg.candidate_count} candidate{cfg.candidate_count !== 1 ? "s" : ""}</span>
                        </div>

                        {/* Keywords preview */}
                        <div className="flex flex-wrap gap-1 mt-3">
                          {cfg.keywords.slice(0, 10).map((kw, i) => (
                            <span key={i} className="px-2 py-0.5 bg-slate-100 text-slate-600 text-xs rounded-full">
                              {kw}
                            </span>
                          ))}
                          {cfg.keywords.length > 10 && (
                            <span className="px-2 py-0.5 text-slate-400 text-xs">
                              +{cfg.keywords.length - 10} more
                            </span>
                          )}
                        </div>
                      </div>

                      {/* Actions */}
                      <div className="flex gap-2 ml-4 shrink-0">
                        <button
                          onClick={() => { setEditingConfig(cfg); setShowConfigForm(true); }}
                          className="border border-slate-300 text-slate-700 px-3 py-1.5 rounded-lg text-sm hover:bg-slate-50"
                        >
                          Edit
                        </button>
                        {!cfg.is_active && (
                          <button
                            onClick={() => activateConfig(cfg.id)}
                            className="border border-green-300 text-green-700 px-3 py-1.5 rounded-lg text-sm hover:bg-green-50"
                          >
                            Activate
                          </button>
                        )}
                        {!cfg.is_active && cfg.candidate_count === 0 && (
                          <button
                            onClick={() => deleteConfig(cfg.id)}
                            className="border border-red-300 text-red-700 px-3 py-1.5 rounded-lg text-sm hover:bg-red-50"
                          >
                            Delete
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
