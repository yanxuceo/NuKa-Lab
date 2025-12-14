export async function post(path: string) {
  const res = await fetch(`http://localhost:8000${path}`, { method: "POST" });
  if (!res.ok) throw new Error(`POST ${path} failed`);
  return res.json().catch(() => ({}));
}
