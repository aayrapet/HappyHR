"use client";

import { useState, useEffect } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Mirroring JobConfigItem from dashboard
interface JobConfigItem {
    id: number;
    title: string;
    description: string;
    keywords: string[];
    mandatory_questions: any[];
    match_threshold: number;
    max_interview_minutes: number;
    is_active: boolean;
}

export default function JobsPage() {
    const [jobs, setJobs] = useState<JobConfigItem[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchJobs = async () => {
            try {
                const res = await fetch(`${API}/api/job-config`);
                const data = await res.json();
                setJobs(data);
            } catch (err) {
                console.error("Failed to fetch jobs");
            } finally {
                setLoading(false);
            }
        };
        fetchJobs();
    }, []);

    return (
        <div className="max-w-4xl mx-auto mt-12 px-4 pb-12">
            <div className="text-center mb-12">
                <h1 className="text-4xl font-bold text-slate-900 mb-4">Open Positions</h1>
                <p className="text-lg text-slate-600">Join our team and help us build the future of HR.</p>
            </div>

            {loading ? (
                <div className="flex items-center justify-center py-20">
                    <div className="text-slate-500">Loading open positions...</div>
                </div>
            ) : jobs.length === 0 ? (
                <div className="text-center py-20 bg-slate-50 rounded-2xl border border-slate-200">
                    <p className="text-slate-500">No open positions at the moment. Please check back later!</p>
                </div>
            ) : (
                <div className="space-y-6">
                    {jobs.map(job => (
                        <div key={job.id} className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm hover:shadow-md transition-shadow">
                            <div className="flex flex-col md:flex-row md:items-start justify-between gap-6">
                                <div className="flex-1">
                                    <h2 className="text-2xl font-bold text-slate-900 mb-2">{job.title}</h2>
                                    <p className="text-slate-600 mb-4 line-clamp-3">{job.description}</p>

                                    <div className="flex flex-wrap gap-2">
                                        {job.keywords.map((kw, i) => (
                                            <span key={i} className="bg-blue-50 text-blue-700 font-medium px-2.5 py-1 rounded-md text-xs">
                                                {kw}
                                            </span>
                                        ))}
                                    </div>
                                </div>

                                <div className="shrink-0 flex items-center md:items-end flex-col justify-between">
                                    <a
                                        href={`/apply/${job.id}`}
                                        className="w-full md:w-auto text-center bg-blue-600 text-white px-8 py-3 rounded-xl font-bold hover:bg-blue-700 transition-colors"
                                    >
                                        Apply Now
                                    </a>
                                    <span className="text-xs text-slate-500 mt-2 block md:mt-4">
                                        ~{job.max_interview_minutes} min interview
                                    </span>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
