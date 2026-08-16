//! Feature-independent bounds shared by the construction child and watchdog.

pub const CONSTRUCTION_MEMORY_EVENT_SCHEMA: &str = "gpt-oss-rs.construction-memory-event/v1";
pub const MAX_CONSTRUCTION_MEMORY_EVENTS: usize = 64;
pub const MAX_CONSTRUCTION_MEMORY_EVENT_BYTES: usize = 64 * 1024;
pub const MAX_CONSTRUCTION_MEMORY_JOURNAL_BYTES: usize = 4 * 1024 * 1024;
