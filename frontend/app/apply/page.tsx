"use client";

import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface JobConfig {
  id: number;
  title: string;
  is_active: boolean;
}

export default function ApplyPage() {
  const [form, setForm] = useState({ first_name: "", last_name: "", email: "" });
  const [file, setFile] = useState<File | null>(null);
  const [jobs, setJobs] = useState<JobConfig[]>([]);
  const [jobId, setJobId] = useState<number | "">("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ status: string; match_percent: number } | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API}/api/job-configs`);
        if (!res.ok) return;
        const data = await res.json();
        const activeJobs = (data as JobConfig[]).filter(j => j.is_active);
        setJobs(activeJobs);
        if (activeJobs.length > 0) {
          setJobId(activeJobs[0].id);
        }
      } catch {
        // Keep silent and let submit errors surface to user.
      }
    })();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return setError("Please upload your CV (PDF)");
    if (!jobId) return setError("Please choose a job offer");
    setLoading(true);
    setError("");
    setResult(null);

    const fd = new FormData();
    fd.append("first_name", form.first_name);
    fd.append("last_name", form.last_name);
    fd.append("email", form.email);
    fd.append("job_id", String(jobId));
    fd.append("cv", file);

    try {
      const res = await fetch(`${API}/api/apply`, { method: "POST", body: fd });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Application failed");
      }
      const data = await res.json();
      setResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-lg mx-auto mt-16 px-4">
      <h1 className="text-3xl font-bold text-slate-900 mb-2">Apply for a Position</h1>
      <p className="text-slate-600 mb-8">Upload your CV and we'll screen it instantly.</p>

      {result ? (
        <div className={`rounded-xl p-6 text-center ${
          result.status === "invited"
            ? "bg-green-50 border border-green-200"
            : "bg-red-50 border border-red-200"
        }`}>
          {result.status === "invited" ? (
            <>
              <div className="text-4xl mb-3">&#10003;</div>
              <h2 className="text-xl font-bold text-green-800 mb-2">You're Invited!</h2>
              <p className="text-green-700">
                Your CV matched {result.match_percent}% of our requirements.
                Check your email for an interview link.
              </p>
            </>
          ) : (
            <>
              <div className="text-4xl mb-3">&#10007;</div>
              <h2 className="text-xl font-bold text-red-800 mb-2">Not a Match</h2>
              <p className="text-red-700">
                Your CV matched {result.match_percent}% of our requirements.
                We need a higher match to proceed.
              </p>
            </>
          )}
          <button
            onClick={() => { setResult(null); setForm({ first_name: "", last_name: "", email: "" }); setFile(null); }}
            className="mt-4 text-sm text-slate-600 hover:text-slate-900 underline"
          >
            Apply again
          </button>
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Job Offer</label>
            <select
              required
              value={jobId}
              onChange={e => setJobId(e.target.value ? Number(e.target.value) : "")}
              className="w-full border border-slate-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
            >
              {jobs.length === 0 && <option value="">No active offers available</option>}
              {jobs.map(j => (
                <option key={j.id} value={j.id}>{j.title}</option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">First Name</label>
              <input
                type="text"
                required
                value={form.first_name}
                onChange={e => setForm({ ...form, first_name: e.target.value })}
                className="w-full border border-slate-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Last Name</label>
              <input
                type="text"
                required
                value={form.last_name}
                onChange={e => setForm({ ...form, last_name: e.target.value })}
                className="w-full border border-slate-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Email</label>
            <input
              type="email"
              required
              value={form.email}
              onChange={e => setForm({ ...form, email: e.target.value })}
              className="w-full border border-slate-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">CV (PDF)</label>
            <div className="border-2 border-dashed border-slate-300 rounded-lg p-6 text-center hover:border-blue-400 transition-colors">
              <input
                type="file"
                accept=".pdf"
                onChange={e => setFile(e.target.files?.[0] || null)}
                className="hidden"
                id="cv-upload"
              />
              <label htmlFor="cv-upload" className="cursor-pointer">
                {file ? (
                  <p className="text-slate-700 font-medium">{file.name}</p>
                ) : (
                  <>
                    <p className="text-slate-500">Click to upload your PDF resume</p>
                    <p className="text-xs text-slate-400 mt-1">PDF format only</p>
                  </>
                )}
              </label>
            </div>
          </div>

          {error && <p className="text-red-600 text-sm">{error}</p>}

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 text-white py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? "Screening..." : "Submit Application"}
          </button>
        </form>
      )}
    </div>
  );
}
