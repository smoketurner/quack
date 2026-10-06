-- v9: a user's lifecycle after creation: disabled, password changes, and
-- the failed-login count behind a timed lockout.
--
-- Frozen history once shipped: add a new version instead of editing this.

ALTER TABLE "users" ADD COLUMN "disabled_at" text;
ALTER TABLE "users" ADD COLUMN "password_changed_at" text;
ALTER TABLE "users" ADD COLUMN "failed_logins" integer NOT NULL DEFAULT 0;
ALTER TABLE "users" ADD COLUMN "locked_until" text;
