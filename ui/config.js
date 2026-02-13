(() => {
  const DEFAULT_API_BASE = "http://127.0.0.1:58452";
  let urlBase = "";
  try {
    const params = new URLSearchParams(location.search || "");
    urlBase = String(params.get("api") || params.get("apiBase") || "").trim();
  } catch {}

  let storedBase = "";
  try {
    storedBase = String(localStorage.getItem("AGENT_API_BASE") || "").trim();
  } catch {}

  window.AGENT_API_BASE = urlBase || storedBase || DEFAULT_API_BASE;
})();
