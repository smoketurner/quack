-- v2: drop the content tables and give the audit log its access-record shape
-- (who, what resource, outcome, channel, when; no detail column).
--
-- "threads" and "messages" held workspace content in control.db in the first
-- draft; sessions and messages belong inside the workspace file (design doc
-- 5.4). The detail half of an audit row lives in "_quack_audit" there, under
-- the same UUID v7.
--
-- Frozen history: this file has shipped. Never edit it — add a new version
-- instead.

DROP TABLE IF EXISTS "messages";
DROP TABLE IF EXISTS "threads";
DROP TABLE IF EXISTS "audit_log";

CREATE TABLE IF NOT EXISTS "audit_log" (
    "id"            text NOT NULL PRIMARY KEY,
    "timestamp"     text NOT NULL DEFAULT (CURRENT_TIMESTAMP),
    "user_id"       text,
    "token_hash"    text,
    "workspace_id"  text,
    "action"        text NOT NULL,
    "resource_type" text,
    "resource_id"   text,
    "outcome"       text NOT NULL,
    "channel"       text NOT NULL,
    "client_addr"   text,
    "request_id"    text
);

CREATE INDEX IF NOT EXISTS "audit_log_user_ts" ON "audit_log" ("user_id", "timestamp");
CREATE INDEX IF NOT EXISTS "audit_log_workspace_ts" ON "audit_log" ("workspace_id", "timestamp");
