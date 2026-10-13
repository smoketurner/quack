import { useEffect, useState } from "react";
import { messageOf, settings } from "../commands";
import type { SettingsView } from "../commands";

/** The configuration as it stands: the models, the paths, every setting. */
export default function Settings() {
  const [view, setView] = useState<SettingsView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [all, setAll] = useState(false);

  useEffect(() => {
    settings()
      .then(setView)
      .catch((e) => setError(messageOf(e)));
  }, []);

  if (error) return <p className="p-6 text-sm text-red-600">{error}</p>;
  if (!view) return <p className="p-6 text-sm text-gray-500">Loading…</p>;
  const shown = all ? view.settings : view.settings.filter((s) => !s.is_default);
  return (
    <div className="p-6 text-sm">
      <h1 className="mb-4 text-lg font-semibold">Settings</h1>
      <dl className="mb-6 grid grid-cols-[10rem_1fr] gap-y-1">
        <dt className="text-gray-500">Version</dt>
        <dd>{view.version}</dd>
        <dt className="text-gray-500">Chat model</dt>
        <dd>{view.chat_model}</dd>
        <dt className="text-gray-500">Config file</dt>
        <dd>
          <code>{view.config_path}</code> <span className="text-gray-500">({view.file_state})</span>
          {view.file_error && <p className="text-red-600">{view.file_error}</p>}
        </dd>
        <dt className="text-gray-500">Data directory</dt>
        <dd>
          <code>{view.data_dir}</code>
        </dd>
        <dt className="text-gray-500">Logs</dt>
        <dd>
          <code>{view.log_dir}</code>
        </dd>
      </dl>
      {view.unknown_keys.length > 0 && (
        <p className="mb-4 text-red-600">
          Unknown keys in the config file: {view.unknown_keys.join(", ")}
        </p>
      )}
      <div className="mb-2 flex items-center gap-3">
        <h2 className="font-semibold">Settings</h2>
        <label className="flex items-center gap-1 text-gray-500">
          <input type="checkbox" checked={all} onChange={(e) => setAll(e.target.checked)} />
          show defaults
        </label>
      </div>
      <table className="mb-6 w-full">
        <thead className="text-left text-gray-500">
          <tr>
            <th className="py-1">Setting</th>
            <th className="py-1">Value</th>
            <th className="py-1">From</th>
          </tr>
        </thead>
        <tbody>
          {shown.map((s) => (
            <tr key={s.path} className="border-t border-gray-100 dark:border-gray-800">
              <td className="py-1 font-mono text-xs">{s.path}</td>
              <td className="py-1 font-mono text-xs">{s.value ?? "(unset)"}</td>
              <td className="py-1 text-gray-500">{s.origin}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <h2 className="mb-2 font-semibold">Environment</h2>
      <table className="w-full">
        <tbody>
          {view.environment.map((v) => (
            <tr key={v.name} className="border-t border-gray-100 dark:border-gray-800">
              <td className="py-1 font-mono text-xs">{v.name}</td>
              <td className="py-1 text-gray-500">{v.set ? "set" : "not set"}</td>
              <td className="py-1 text-gray-500">{v.purpose}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
