//! Identifiers, one type per kind of thing identified, so a workspace id
//! cannot be passed where a user id is expected. Each is the text a row
//! stores (a UUID v7 for the ones quack generates) and reads, writes,
//! binds, and serializes as that text.

/// An identifier type over `String`: `generate` (UUID v7), `as_str`,
/// `Display`, `FromStr`, serde as the bare string, `sqlx` decoding and a
/// `sea-query` value for `control.db`, and `DuckDB` `ToSql`/`FromSql`.
macro_rules! id_type {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(
            Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            /// A new id: a UUID v7, so ids sort by creation time.
            #[must_use]
            pub fn generate() -> Self {
                Self(uuid::Uuid::now_v7().to_string())
            }

            #[must_use]
            pub fn as_str(&self) -> &str {
                &self.0
            }

            #[must_use]
            pub fn into_string(self) -> String {
                self.0
            }

            /// The first eight characters, for listings where the whole
            /// id would crowd the line.
            #[must_use]
            pub fn short(&self) -> &str {
                self.0.get(..8).unwrap_or(&self.0)
            }
        }

        impl ::std::fmt::Display for $name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl ::std::str::FromStr for $name {
            type Err = ::std::convert::Infallible;

            fn from_str(text: &str) -> ::std::result::Result<Self, Self::Err> {
                Ok(Self(text.to_owned()))
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }

        impl From<String> for $name {
            fn from(id: String) -> Self {
                Self(id)
            }
        }

        impl From<&str> for $name {
            fn from(id: &str) -> Self {
                Self(id.to_owned())
            }
        }

        impl From<$name> for ::sea_query::Value {
            fn from(id: $name) -> Self {
                id.0.into()
            }
        }

        impl From<&$name> for ::sea_query::Value {
            fn from(id: &$name) -> Self {
                id.0.as_str().into()
            }
        }

        impl ::sqlx::Type<::sqlx::Sqlite> for $name {
            fn type_info() -> ::sqlx::sqlite::SqliteTypeInfo {
                <String as ::sqlx::Type<::sqlx::Sqlite>>::type_info()
            }

            fn compatible(ty: &::sqlx::sqlite::SqliteTypeInfo) -> bool {
                <String as ::sqlx::Type<::sqlx::Sqlite>>::compatible(ty)
            }
        }

        impl<'r> ::sqlx::Decode<'r, ::sqlx::Sqlite> for $name {
            fn decode(
                value: <::sqlx::Sqlite as ::sqlx::Database>::ValueRef<'r>,
            ) -> ::std::result::Result<Self, ::sqlx::error::BoxDynError> {
                <String as ::sqlx::Decode<'r, ::sqlx::Sqlite>>::decode(value).map(Self)
            }
        }

        impl ::duckdb::ToSql for $name {
            fn to_sql(&self) -> ::duckdb::Result<::duckdb::types::ToSqlOutput<'_>> {
                Ok(::duckdb::types::ToSqlOutput::from(self.0.as_str()))
            }
        }

        impl ::duckdb::types::FromSql for $name {
            fn column_result(
                value: ::duckdb::types::ValueRef<'_>,
            ) -> ::duckdb::types::FromSqlResult<Self> {
                Ok(Self(value.as_str()?.to_owned()))
            }
        }
    };
}

id_type!(
    /// A workspace, as `control.db` records it.
    WorkspaceId
);

id_type!(
    /// A server user.
    UserId
);

id_type!(
    /// A chat session in a workspace.
    SessionId
);

id_type!(
    /// An access-audit row, shared with the workspace's `_quack_audit`
    /// detail row for the same event.
    AuditId
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_their_text_everywhere() {
        let id = WorkspaceId::generate();
        let text = id.to_string();
        assert_eq!(text.parse::<WorkspaceId>().ok(), Some(id.clone()));
        assert_eq!(
            serde_json::to_value(&id).ok(),
            Some(serde_json::Value::String(text.clone()))
        );
        assert_eq!(id.as_str(), text);
    }
}
