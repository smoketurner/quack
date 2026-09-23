//! One text form per variant for the fieldless enums that are stored,
//! typed at a prompt, or sent over the API: `as_str`, `Display`, `FromStr`,
//! and `ALL`, generated from a single list so the four cannot disagree.

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
            pub fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $text),+
                }
            }
        }

        impl ::std::fmt::Display for $name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                f.write_str(self.as_str())
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
    use std::fmt::Display;
    use std::str::FromStr;

    use serde::Serialize;

    use crate::analysis::chart::ChartKind;
    use crate::error::Error;
    use crate::ontology::PropertyType;
    use crate::storage::control::{Channel, Outcome, Role, Scope};
    use crate::storage::sessions::{ChatMode, MessageRole};
    use crate::storage::workspace::{DocumentSource, DocumentStatus};

    /// Every value's text form is its serde name, and reads back, trimmed
    /// and in any case.
    fn round_trips<T>(all: &[T])
    where
        T: Copy + PartialEq + std::fmt::Debug + Display + FromStr<Err = Error> + Serialize,
    {
        for &value in all {
            let text = value.to_string();
            assert_eq!(
                serde_json::to_value(value).ok(),
                Some(serde_json::Value::String(text.clone())),
                "{value:?}"
            );
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
