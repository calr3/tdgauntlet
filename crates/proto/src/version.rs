use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// The only protocol version this build speaks.
pub const PROTOCOL_VERSION: u32 = 1;

/// A protocol version number that refuses to deserialize unless it is one we speak.
///
/// Rejecting at the edge means nothing downstream has to re-check it, and a client built against
/// a future version gets a clear error instead of a puzzling field-level one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProtocolVersion(u32);

impl ProtocolVersion {
    pub const CURRENT: ProtocolVersion = ProtocolVersion(PROTOCOL_VERSION);

    pub fn get(self) -> u32 {
        self.0
    }
}

impl Default for ProtocolVersion {
    fn default() -> Self {
        Self::CURRENT
    }
}

impl Serialize for ProtocolVersion {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u32(self.0)
    }
}

impl<'de> Deserialize<'de> for ProtocolVersion {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let v = u32::deserialize(d)?;
        if v == PROTOCOL_VERSION {
            Ok(ProtocolVersion(v))
        } else {
            Err(serde::de::Error::custom(format!(
                "unsupported protocol version {v}; this build speaks {PROTOCOL_VERSION}"
            )))
        }
    }
}
