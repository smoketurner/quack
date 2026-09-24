-- v4: index the audit log's sort key for paging through the whole log.
--
-- Frozen history once shipped: add a new version instead of editing this.

CREATE INDEX IF NOT EXISTS "audit_log_ts_id" ON "audit_log" ("timestamp", "id");
