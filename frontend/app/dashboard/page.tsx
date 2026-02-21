"use client";

import { useState, useEffect, useCallback } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface CandidateListItem {
  id: number;
  first_name: string;
  last_name: string;
  email: string;
  status: string;
  global_score: number | null;
  recommendation: string | null;
  match_percent: number;
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

export default function DashboardPage() {
  const [candidates, setCandidates] = useState<CandidateListItem[]>([]);
  const [selected, setSelected] = useState<CandidateDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [deciding, setDeciding] = useState(false);

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

  useEffect(() => {
    fetchCandidates();
  }, [fetchCandidates]);

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

  const scoreColor = (score: number | null) => {
    if (score === null) return "text-slate-400";
    if (score >= 75) return "text-green-600";
    if (score >= 50) return "text-yellow-600";
    return "text-red-600";
  };

  return (
    <div className="flex h-[calc(100vh-57px)]">
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
              className={`w-full text-left p-4 border-b border-slate-100 hover:bg-slate-50 transition-colors ${
                selected?.id === c.id ? "bg-blue-50 border-l-4 border-l-blue-500" : ""
              }`}
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-semibold text-slate-900">{c.first_name} {c.last_name}</p>
                  <p className="text-sm text-slate-500">{c.email}</p>
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
                <p className="text-slate-500">{selected.email}</p>
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
                        className={`px-3 py-1 rounded-full text-sm font-medium ${
                          kw.score > 0 ? "bg-green-100 text-green-800" : "bg-slate-100 text-slate-500"
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
  );
}
