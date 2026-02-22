export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[calc(100vh-57px)] px-4">
      <div className="max-w-2xl text-center">
        <h1 className="text-5xl font-bold text-slate-900 mb-4">
          AI-Powered <span className="text-blue-600">Recruiting</span>
        </h1>
        <p className="text-lg text-slate-600 mb-8">
          Apply with your CV, get screened instantly, and complete a natural voice interview with our AI recruiter.
        </p>
        <div className="flex gap-4 justify-center">
          <a
            href="/jobs"
            className="bg-blue-600 text-white px-8 py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors"
          >
            Job Posts
          </a>
          <a
            href="/dashboard"
            className="border border-slate-300 text-slate-700 px-8 py-3 rounded-lg font-semibold hover:bg-slate-50 transition-colors"
          >
            HR Dashboard
          </a>
        </div>
      </div>
    </div>
  );
}
