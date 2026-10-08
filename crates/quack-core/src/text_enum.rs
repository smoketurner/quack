//! One text form per variant for the fieldless enums that are stored,
//! typed at a prompt, or sent over the API: `as_str`, `Display`, `FromStr`,
//! and `ALL`, generated from a single list so the four cannot disagree.

/// The conversions a two-valued enum needs where it meets a boolean: a
/// column or a JSON field that stores the flag, or a command-line switch.
/// `From` in both directions (so serde's `from`/`into` can carry it) and
/// `duckdb::ToSql` and `FromSql` as the boolean.
macro_rules! flag_enum {
    ($name:ident, false => $off:ident, true => $on:ident) => {
        impl From<bool> for $name {
            fn from(flag: bool) -> Self {
                if flag { Self::$on } else { Self::$off }
            }
        }

        impl From<$name> for bool {
            fn from(value: $name) -> Self {
                match value {
                    $name::$off => false,
                    $name::$on => true,
                }
            }
        }

        impl ::duckdb::ToSql for $name {
            fn to_sql(&self) -> ::duckdb::Result<::duckdb::types::ToSqlOutput<'_>> {
                Ok(::duckdb::types::ToSqlOutput::from(bool::from(*self)))
            }
        }

        impl ::duckdb::types::FromSql for $name {
            fn column_result(
                value: ::duckdb::types::ValueRef<'_>,
            ) -> ::duckdb::types::FromSqlResult<Self> {
                bool::column_result(value).map(Self::from)
            }
        }

        /// Serialized as the boolean, so documented as one.
        impl ::utoipa::PartialSchema for $name {
            fn schema() -> ::utoipa::openapi::RefOr<::utoipa::openapi::schema::Schema> {
                <bool as ::utoipa::PartialSchema>::schema()
            }
        }

        impl ::utoipa::ToSchema for $name {}
    };
}

/// Store a [`text_enum!`] enum in a `DuckDB` column as its text form: bound
/// as a parameter, and read back through `FromStr`, so an unknown stored
/// value is a conversion error rather than a guess.
macro_rules! text_enum_sql {
    ($name:ident) => {
        impl ::duckdb::ToSql for $name {
            fn to_sql(&self) -> ::duckdb::Result<::duckdb::types::ToSqlOutput<'_>> {
                Ok(::duckdb::types::ToSqlOutput::from(self.as_str()))
            }
        }

        impl ::duckdb::types::FromSql for $name {
            fn column_result(
                value: ::duckdb::types::ValueRef<'_>,
            ) -> ::duckdb::types::FromSqlResult<Self> {
                value.as_str()?.parse().map_err(|e: $crate::error::Error| {
                    ::duckdb::types::FromSqlError::Other(Box::new(e))
                })
            }
        }
    };
}

/// Like [`text_enum!`], for an enum that names stored history: `$unknown`
/// is its one tuple variant, carrying a stored name this build does not
/// define, so a row a newer build wrote reads back instead of failing.
/// Parsing never fails; `ALL` lists the defined values; serde reads and
/// writes the text form (`from`/`into` `String`).
macro_rules! history_enum {
    ($name:ident, $unknown:ident, { $($variant:ident => $text:literal),+ $(,)? }) => {
        impl $name {
            /// Every value this build defines, in declaration order.
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];

            /// The text form, as `Display` writes it and `FromStr` reads it.
            #[must_use]
            pub fn as_str(&self) -> &str {
                match self {
                    $(Self::$variant => $text,)+
                    Self::$unknown(name) => name,
                }
            }

            /// Whether this build defines the value: only those are written.
            #[must_use]
            pub const fn is_defined(&self) -> bool {
                !matches!(self, Self::$unknown(_))
            }
        }

        impl ::std::fmt::Display for $name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                f.pad(self.as_str())
            }
        }

        impl From<String> for $name {
            fn from(text: String) -> Self {
                text.parse().unwrap_or_else(|never: ::std::convert::Infallible| match never {})
            }
        }

        impl From<$name> for String {
            fn from(value: $name) -> Self {
                value.as_str().to_owned()
            }
        }

        impl ::std::str::FromStr for $name {
            type Err = ::std::convert::Infallible;

            fn from_str(text: &str) -> ::std::result::Result<Self, Self::Err> {
                let wanted = text.trim();
                Ok(Self::ALL
                    .iter()
                    .find(|value| value.as_str().eq_ignore_ascii_case(wanted))
                    .cloned()
                    .unwrap_or_else(|| Self::$unknown(wanted.to_owned())))
            }
        }

        /// Serialized as its text, which a newer build may extend, so
        /// documented as a string with the values this build defines.
        impl ::utoipa::PartialSchema for $name {
            fn schema() -> ::utoipa::openapi::RefOr<::utoipa::openapi::schema::Schema> {
                ::utoipa::openapi::schema::ObjectBuilder::new()
                    .schema_type(::utoipa::openapi::schema::Type::String)
                    .examples(Self::ALL.iter().map(|value| value.as_str().to_owned()))
                    .into()
            }
        }

        impl ::utoipa::ToSchema for $name {}
    };
}

/// Implement `as_str`, `ALL`, `Display`, and `FromStr` for a fieldless,
/// `Copy` enum from its variants' text forms, which must be the names its
/// serde attributes give. Parsing trims and ignores ASCII case; anything
/// else is [`Error::UnknownValue`](crate::error::Error::UnknownValue)
/// naming the accepted forms.
macro_rules! text_enum {
    ($name:ident, $what:literal, { $($variant:ident => $text:literal),+ $(,)? }) => {
        impl $name {
            /// Every value, in declaration order.
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];

            /// The text form, as `Display` writes it and `FromStr` reads it.
            #[must_use]
            pub const fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $text),+
                }
            }
        }

        impl ::std::fmt::Display for $name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                f.pad(self.as_str())
            }
        }

        impl ::std::str::FromStr for $name {
            type Err = $crate::error::Error;

            fn from_str(text: &str) -> ::std::result::Result<Self, Self::Err> {
                // A linear scan: every enum here has a handful of values.
                let wanted = text.trim();
                Self::ALL
                    .iter()
                    .copied()
                    .find(|value| value.as_str().eq_ignore_ascii_case(wanted))
                    .ok_or_else(|| $crate::error::Error::UnknownValue {
                        what: $what,
                        value: wanted.to_owned(),
                        allowed: Self::ALL
                            .iter()
                            .map(|value| value.as_str())
                            .collect::<Vec<_>>()
                            .join(", "),
                    })
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::fmt::{Debug, Display};
    use std::str::FromStr;

    use serde::Serialize;
    use serde_json::Value;

    use crate::analysis::chart::ChartKind;
    use crate::doctor::{Area, Status};
    use crate::error::Error;
    use crate::graph::ExtractSource;
    use crate::graph::resolve::{MergeDecision, MergeStatus};
    use crate::okf::ConceptType;
    use crate::ontology::PropertyType;
    use crate::ontology::candidates::{CandidateAction, CandidateStatus, Queue};
    use crate::ontology::induction::ItemKind;
    use crate::storage::control::{AuditAction, Channel, Outcome, ResourceKind, Role, Scope};
    use crate::storage::sessions::{ChatMode, MessageRole};
    use crate::storage::workspace::{DocumentSource, DocumentStatus, MetaKey};

    /// Every value's text form is its serde name, and reads back.
    fn round_trips<T>(all: &[T])
    where
        T: Copy + PartialEq + Debug + Display + FromStr<Err = Error> + Serialize,
    {
        for &value in all {
            assert_eq!(
                serde_json::to_value(value).ok(),
                Some(Value::String(value.to_string())),
                "{value:?}"
            );
        }
        text_round_trips(all);
    }

    /// Every value's text form reads back, trimmed and in any case, and
    /// `Display` honors width.
    fn text_round_trips<T>(all: &[T])
    where
        T: Copy + PartialEq + Debug + Display + FromStr<Err = Error>,
    {
        for &value in all {
            let text = value.to_string();
            assert_eq!(format!("{value:>30}"), format!("{text:>30}"));
            assert_eq!(text.parse::<T>().ok(), Some(value));
            assert_eq!(
                format!("  {}  ", text.to_ascii_uppercase())
                    .parse::<T>()
                    .ok(),
                Some(value)
            );
        }
    }

    #[test]
    fn every_text_enum_matches_its_serde_names() {
        round_trips(ChatMode::ALL);
        round_trips(MessageRole::ALL);
        round_trips(Role::ALL);
        round_trips(Scope::ALL);
        round_trips(Outcome::ALL);
        round_trips(Channel::ALL);
        round_trips(ChartKind::ALL);
        round_trips(PropertyType::ALL);
        round_trips(DocumentSource::ALL);
        round_trips(DocumentStatus::ALL);
        round_trips(MergeStatus::ALL);
        round_trips(MergeDecision::ALL);
        round_trips(CandidateStatus::ALL);
        round_trips(Queue::ALL);
        round_trips(CandidateAction::ALL);
        round_trips(ItemKind::ALL);
        round_trips(ExtractSource::ALL);
        round_trips(Status::ALL);
        round_trips(Area::ALL);
        text_round_trips(MetaKey::ALL);
        text_round_trips(ConceptType::ALL);
    }

    /// A history enum's defined values round-trip through text and serde
    /// like any other; an undefined stored name reads back carrying itself,
    /// prints as itself, and is not defined.
    #[test]
    fn history_enums_keep_undefined_names() {
        fn stored<T>(all: &[T])
        where
            T: Clone
                + PartialEq
                + Debug
                + Display
                + FromStr<Err = Infallible>
                + Serialize
                + From<String>,
        {
            for value in all {
                assert_eq!(
                    serde_json::to_value(value).ok(),
                    Some(Value::String(value.to_string())),
                    "{value:?}"
                );
                let text = value.to_string();
                let upper = format!("  {}  ", text.to_ascii_uppercase());
                assert_eq!(upper.parse::<T>().ok().as_ref(), Some(value));
                assert_eq!(T::from(text), *value);
            }
        }
        stored(AuditAction::ALL);
        stored(ResourceKind::ALL);
        let retired = AuditAction::from(String::from(" retired_action "));
        assert_eq!(
            retired,
            AuditAction::Unknown(String::from("retired_action"))
        );
        assert_eq!(retired.to_string(), "retired_action");
        assert!(!retired.is_defined());
        assert!(AuditAction::Open.is_defined());
        assert_eq!(
            serde_json::from_value::<ResourceKind>(Value::String(String::from("widget"))).ok(),
            Some(ResourceKind::Unknown(String::from("widget")))
        );
    }

    #[test]
    fn unknown_text_names_what_was_expected() {
        let err = "boss".parse::<Role>().err().map(|e| e.to_string());
        assert_eq!(
            err.as_deref(),
            Some("unknown role 'boss'; use one of: viewer, member, owner")
        );
        assert!("".parse::<Scope>().is_err());
    }
}
