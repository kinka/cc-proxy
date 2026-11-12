//! First-hand token ledger.
//!
//! Every request that reaches an upstream is written here exactly once, with the identity
//! headers the caller sent. The point is to be the number of record for "how many tokens did
//! we spend today": the callers' own accounting is reported by their agent kernels at turn end,
//! and structurally misses crashed/cancelled turns, background calls and anything that is not a
//! conversation turn at all. This proxy sees every request regardless.
//!
//! Storage is one JSONL file per Beijing date (`<dir>/YYYY-MM-DD.jsonl`), appended by a
//! dedicated thread so a slow disk never stalls a stream. A write failure is a warning, never a
//! request failure. No bodies are recorded — only sizes and identities.
//!
//! "Exactly once" is enforced by `UsageTracking::settled`: the first of done / truncated /
//! error / abort wins, and the drop guard only fires for requests nothing else claimed.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    fs::OpenOptions,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const BEIJING_OFFSET_SECS: i64 = 8 * 3600;
const IDENTITY_VALUE_MAX_CHARS: usize = 128;
const TODAY_CACHE_TTL: Duration = Duration::from_secs(30);
const CACHE_MAX_ENTRIES: usize = 64;

pub(crate) const TRACE_IDENTIFIER_HEADER: &str = "x-trace-identifier";
pub(crate) const TRACE_IDENTIFIER_TYPE_HEADER: &str = "x-trace-identifier-type";
pub(crate) const CURA_SESSION_HEADER: &str = "x-cura-session-id";
pub(crate) const CURA_ENV_HEADER: &str = "x-cura-env";

/// Identity headers that are part of the upstream contract (compliance requires every LLM call
/// to carry the caller's identity). Anything else the client sends stays here — in particular
/// `x-cura-*`, which only exists for this ledger.
pub(crate) const FORWARDED_TRACE_HEADERS: [&str; 2] = [TRACE_IDENTIFIER_HEADER, TRACE_IDENTIFIER_TYPE_HEADER];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Outcome {
    Done,
    StreamTruncated,
    ClientAborted,
    UpstreamError,
    TransportError,
    InternalError,
}

impl Outcome {
    fn as_str(self) -> &'static str {
        match self {
            Outcome::Done => "done",
            Outcome::StreamTruncated => "stream_truncated",
            Outcome::ClientAborted => "client_aborted",
            Outcome::UpstreamError => "upstream_error",
            Outcome::TransportError => "transport_error",
            Outcome::InternalError => "internal_error",
        }
    }
}

/// Normalised token counts. `prompt_tokens` always INCLUDES cached input, whatever the
/// upstream wire format reported, so `total_tokens = prompt + output` is comparable across
/// providers and matches how upstream quotas count.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct TokenCounts {
    pub(crate) prompt_tokens: u64,
    pub(crate) output_tokens: u64,
    pub(crate) cache_read_tokens: u64,
    pub(crate) cache_creation_tokens: u64,
}

impl TokenCounts {
    /// `input_tokens` in the Anthropic shape excludes cache reads/creations when the upstream
    /// was Chat Completions (`compute_anthropic_input_tokens` subtracts them); Responses-API
    /// usage reports input including cached tokens. `input_includes_cache` says which.
    pub(crate) fn from_parts(
        input_tokens: u64,
        output_tokens: u64,
        cache_read: Option<u64>,
        cache_creation: Option<u64>,
        input_includes_cache: bool,
    ) -> Self {
        let cache_read_tokens = cache_read.unwrap_or(0);
        let cache_creation_tokens = cache_creation.unwrap_or(0);
        let prompt_tokens = if input_includes_cache {
            input_tokens
        } else {
            input_tokens + cache_read_tokens + cache_creation_tokens
        };
        Self { prompt_tokens, output_tokens, cache_read_tokens, cache_creation_tokens }
    }

    /// Read an Anthropic-shaped usage object (`input_tokens` / `cache_read_input_tokens` …).
    pub(crate) fn from_anthropic_usage(usage: &Value, input_includes_cache: bool) -> Self {
        Self::from_parts(
            usage.get("input_tokens").and_then(Value::as_u64).unwrap_or(0),
            usage.get("output_tokens").and_then(Value::as_u64).unwrap_or(0),
            usage.get("cache_read_input_tokens").and_then(Value::as_u64),
            usage.get("cache_creation_input_tokens").and_then(Value::as_u64),
            input_includes_cache,
        )
    }

    /// Read a Responses-shaped usage object (`input_tokens` includes cached).
    pub(crate) fn from_responses_usage(usage: &Value) -> Self {
        Self::from_parts(
            usage.get("input_tokens").and_then(Value::as_u64).unwrap_or(0),
            usage.get("output_tokens").and_then(Value::as_u64).unwrap_or(0),
            usage.pointer("/input_tokens_details/cached_tokens").and_then(Value::as_u64),
            None,
            true,
        )
    }

    pub(crate) fn total(&self) -> u64 {
        self.prompt_tokens + self.output_tokens
    }
}

/// The caller-supplied identity of one request, read from headers before anything else.
#[derive(Debug, Clone, Default)]
pub(crate) struct RequestIdentity {
    pub(crate) trace_id: Option<String>,
    pub(crate) trace_type: Option<String>,
    pub(crate) cura_session_id: Option<String>,
    pub(crate) cura_env: Option<String>,
}

impl RequestIdentity {
    pub(crate) fn from_headers(headers: &http::HeaderMap) -> Self {
        let read = |name: &str| {
            headers
                .get(name)
                .and_then(|value| value.to_str().ok())
                .map(sanitize_identity_value)
                .filter(|value| !value.is_empty())
        };
        Self {
            trace_id: read(TRACE_IDENTIFIER_HEADER),
            trace_type: read(TRACE_IDENTIFIER_TYPE_HEADER),
            cura_session_id: read(CURA_SESSION_HEADER),
            cura_env: read(CURA_ENV_HEADER),
        }
    }
}

fn sanitize_identity_value(value: &str) -> String {
    value
        .trim()
        .chars()
        .filter(|ch| !ch.is_control())
        .take(IDENTITY_VALUE_MAX_CHARS)
        .collect()
}

/// Everything about a request that is known before the upstream answers.
#[derive(Debug, Clone, Default)]
pub(crate) struct UsageMeta {
    pub(crate) provider: Option<String>,
    pub(crate) stream: bool,
    pub(crate) requested_model: Option<String>,
    pub(crate) mapped_model: Option<String>,
    pub(crate) identity: RequestIdentity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct UsageRecord {
    pub(crate) ts: String,
    pub(crate) date: String,
    pub(crate) boot_id: String,
    pub(crate) req_id: u64,
    pub(crate) path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) provider: Option<String>,
    pub(crate) api_format: String,
    pub(crate) stream: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) requested_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) mapped_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) response_model: Option<String>,
    pub(crate) outcome: Outcome,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) status: Option<u16>,
    pub(crate) latency_ms: u64,
    #[serde(flatten)]
    pub(crate) tokens: TokenCounts,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) trace_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) trace_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) cura_session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) cura_env: Option<String>,
}

impl UsageRecord {
    fn model_key(&self) -> &str {
        self.response_model
            .as_deref()
            .or(self.mapped_model.as_deref())
            .or(self.requested_model.as_deref())
            .unwrap_or("(unknown)")
    }
}

/// Handle to the writer thread. Cheap to clone; dropping every clone stops the thread.
#[derive(Clone)]
pub(crate) struct UsageRecorder {
    tx: mpsc::Sender<UsageRecord>,
    boot_id: Arc<str>,
}

impl UsageRecorder {
    pub(crate) fn spawn(dir: PathBuf, retention_days: u32) -> anyhow::Result<Self> {
        std::fs::create_dir_all(&dir)?;
        let (tx, rx) = mpsc::channel::<UsageRecord>();
        std::thread::Builder::new()
            .name("usage-writer".to_string())
            .spawn(move || run_writer(dir, retention_days, rx))?;
        Ok(Self { tx, boot_id: Arc::from(new_boot_id()) })
    }

    fn send(&self, record: UsageRecord) {
        if self.tx.send(record).is_err() {
            log::warn!("usage.write_failed error=writer thread is gone");
        }
    }
}

fn new_boot_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or(0);
    format!("{:x}-{:x}", nanos, std::process::id())
}

fn run_writer(dir: PathBuf, retention_days: u32, rx: mpsc::Receiver<UsageRecord>) {
    let mut last_cleanup_date = String::new();
    while let Ok(record) = rx.recv() {
        if record.date != last_cleanup_date {
            cleanup_expired(&dir, &record.date, retention_days);
            last_cleanup_date = record.date.clone();
        }
        let path = dir.join(format!("{}.jsonl", record.date));
        let result = serde_json::to_string(&record)
            .map_err(std::io::Error::other)
            .and_then(|mut line| {
                line.push('\n');
                // One write per line on an O_APPEND handle: a concurrent reader sees either the
                // whole line or none of it (and skips a torn tail as unparsable anyway).
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)?
                    .write_all(line.as_bytes())
            });
        if let Err(err) = result {
            log::warn!("usage.write_failed path={} error={}", path.display(), err);
        }
    }
}

fn cleanup_expired(dir: &Path, today: &str, retention_days: u32) {
    if retention_days == 0 {
        return;
    }
    let Some(today_days) = parse_date_days(today) else {
        return;
    };
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(date) = name.to_str().and_then(|name| name.strip_suffix(".jsonl")) else {
            continue;
        };
        let Some(days) = parse_date_days(date) else {
            continue;
        };
        if today_days - days >= i64::from(retention_days) {
            match std::fs::remove_file(entry.path()) {
                Ok(()) => log::info!("usage.expired_removed file={}", entry.path().display()),
                Err(err) => log::warn!("usage.expired_remove_failed file={} error={}", entry.path().display(), err),
            }
        }
    }
}

/// Per-request ledger state shared by every place that may settle the request.
pub(crate) struct UsageTracking {
    recorder: UsageRecorder,
    meta: UsageMeta,
    settled: AtomicBool,
}

impl UsageTracking {
    pub(crate) fn new(recorder: UsageRecorder, meta: UsageMeta) -> Self {
        Self { recorder, meta, settled: AtomicBool::new(false) }
    }

    pub(crate) fn is_settled(&self) -> bool {
        self.settled.load(Ordering::Acquire)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn settle(
        &self,
        req_id: u64,
        path: &str,
        api_format: &str,
        started_at: Instant,
        outcome: Outcome,
        status: Option<u16>,
        tokens: TokenCounts,
        response_model: Option<&str>,
    ) {
        if self.settled.swap(true, Ordering::AcqRel) {
            return;
        }
        let now = SystemTime::now();
        let identity = &self.meta.identity;
        self.recorder.send(UsageRecord {
            ts: format_utc_ts(now),
            date: beijing_date(now),
            boot_id: self.recorder.boot_id.to_string(),
            req_id,
            path: path.to_string(),
            provider: self.meta.provider.clone(),
            api_format: api_format.to_string(),
            stream: self.meta.stream,
            requested_model: self.meta.requested_model.clone(),
            mapped_model: self.meta.mapped_model.clone(),
            response_model: response_model.filter(|value| !value.is_empty()).map(str::to_string),
            outcome,
            status,
            latency_ms: started_at.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
            tokens,
            trace_id: identity.trace_id.clone(),
            trace_type: identity.trace_type.clone(),
            cura_session_id: identity.cura_session_id.clone(),
            cura_env: identity.cura_env.clone(),
        });
    }
}

// ---------------------------------------------------------------------------
// Dates. Hand-rolled to avoid a chrono dependency for two conversions.
// ---------------------------------------------------------------------------

fn unix_secs(time: SystemTime) -> (i64, u32) {
    match time.duration_since(UNIX_EPOCH) {
        Ok(value) => (value.as_secs() as i64, value.subsec_millis()),
        Err(_) => (0, 0),
    }
}

/// Howard Hinnant's `civil_from_days`.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let m = i64::from(m);
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

pub(crate) fn parse_date_days(date: &str) -> Option<i64> {
    let bytes = date.as_bytes();
    if bytes.len() != 10 || bytes[4] != b'-' || bytes[7] != b'-' {
        return None;
    }
    let y: i64 = date[0..4].parse().ok()?;
    let m: u32 = date[5..7].parse().ok()?;
    let d: u32 = date[8..10].parse().ok()?;
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    Some(days_from_civil(y, m, d))
}

pub(crate) fn beijing_date(time: SystemTime) -> String {
    let (secs, _) = unix_secs(time);
    let (y, m, d) = civil_from_days((secs + BEIJING_OFFSET_SECS).div_euclid(86_400));
    format!("{y:04}-{m:02}-{d:02}")
}

fn format_utc_ts(time: SystemTime) -> String {
    let (secs, millis) = unix_secs(time);
    let (y, m, d) = civil_from_days(secs.div_euclid(86_400));
    let rem = secs.rem_euclid(86_400);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{millis:03}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60,
    )
}

// ---------------------------------------------------------------------------
// Query side.
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone, Copy)]
struct Bucket {
    requests: u64,
    tokens: TokenCounts,
}

impl Bucket {
    fn add(&mut self, record: &UsageRecord) {
        self.requests += 1;
        self.tokens.prompt_tokens += record.tokens.prompt_tokens;
        self.tokens.output_tokens += record.tokens.output_tokens;
        self.tokens.cache_read_tokens += record.tokens.cache_read_tokens;
        self.tokens.cache_creation_tokens += record.tokens.cache_creation_tokens;
    }

    fn to_json(self) -> Value {
        json!({
            "requests": self.requests,
            "prompt_tokens": self.tokens.prompt_tokens,
            "output_tokens": self.tokens.output_tokens,
            "cache_read_tokens": self.tokens.cache_read_tokens,
            "cache_creation_tokens": self.tokens.cache_creation_tokens,
            "total_tokens": self.tokens.total(),
        })
    }
}

fn read_records(dir: &Path, date: &str) -> std::io::Result<Vec<UsageRecord>> {
    let path = dir.join(format!("{date}.jsonl"));
    let file = match std::fs::File::open(&path) {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err),
    };
    Ok(BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter_map(|line| serde_json::from_str::<UsageRecord>(&line).ok())
        .collect())
}

fn sorted_buckets(map: HashMap<String, Bucket>, key_name: &str) -> Vec<Value> {
    let mut entries: Vec<_> = map.into_iter().collect();
    entries.sort_by(|a, b| b.1.tokens.total().cmp(&a.1.tokens.total()).then_with(|| a.0.cmp(&b.0)));
    entries
        .into_iter()
        .map(|(key, bucket)| {
            let mut value = bucket.to_json();
            value[key_name] = json!(key);
            value
        })
        .collect()
}

pub(crate) fn aggregate_daily(records: &[UsageRecord], date: &str, env: Option<&str>) -> Value {
    let mut totals = Bucket::default();
    let mut unattributed = Bucket::default();
    let mut outcomes: HashMap<&'static str, u64> = HashMap::new();
    let mut by_user: HashMap<String, Bucket> = HashMap::new();
    let mut by_model: HashMap<String, Bucket> = HashMap::new();
    let mut by_env: HashMap<String, Bucket> = HashMap::new();

    for record in records {
        if let Some(env) = env {
            if record.cura_env.as_deref() != Some(env) {
                continue;
            }
        }
        totals.add(record);
        *outcomes.entry(record.outcome.as_str()).or_default() += 1;
        match &record.trace_id {
            Some(trace_id) => by_user.entry(trace_id.clone()).or_default().add(record),
            None => unattributed.add(record),
        }
        by_model.entry(record.model_key().to_string()).or_default().add(record);
        by_env
            .entry(record.cura_env.clone().unwrap_or_else(|| "(none)".to_string()))
            .or_default()
            .add(record);
    }

    json!({
        "date": date,
        "env": env,
        "totals": totals.to_json(),
        "outcomes": outcomes,
        "unattributed": unattributed.to_json(),
        "by_user": sorted_buckets(by_user, "trace_id"),
        "by_model": sorted_buckets(by_model, "model"),
        "by_env": sorted_buckets(by_env, "env"),
    })
}

pub(crate) fn aggregate_sessions(records: &[UsageRecord], date: &str, trace_id: Option<&str>) -> Value {
    struct SessionBucket {
        bucket: Bucket,
        trace_id: Option<String>,
        cura_env: Option<String>,
        first_ts: String,
        last_ts: String,
        outcomes: HashMap<&'static str, u64>,
    }
    let mut sessions: HashMap<String, SessionBucket> = HashMap::new();
    for record in records {
        if let Some(trace_id) = trace_id {
            if record.trace_id.as_deref() != Some(trace_id) {
                continue;
            }
        }
        let key = record.cura_session_id.clone().unwrap_or_else(|| "(none)".to_string());
        let entry = sessions.entry(key).or_insert_with(|| SessionBucket {
            bucket: Bucket::default(),
            trace_id: record.trace_id.clone(),
            cura_env: record.cura_env.clone(),
            first_ts: record.ts.clone(),
            last_ts: record.ts.clone(),
            outcomes: HashMap::new(),
        });
        entry.bucket.add(record);
        *entry.outcomes.entry(record.outcome.as_str()).or_default() += 1;
        if record.ts < entry.first_ts {
            entry.first_ts = record.ts.clone();
        }
        if record.ts > entry.last_ts {
            entry.last_ts = record.ts.clone();
        }
    }
    let mut list: Vec<_> = sessions.into_iter().collect();
    list.sort_by(|a, b| b.1.bucket.tokens.total().cmp(&a.1.bucket.tokens.total()).then_with(|| a.0.cmp(&b.0)));
    let sessions: Vec<Value> = list
        .into_iter()
        .map(|(session_id, entry)| {
            let mut value = entry.bucket.to_json();
            value["session_id"] = json!(session_id);
            value["trace_id"] = json!(entry.trace_id);
            value["env"] = json!(entry.cura_env);
            value["first_ts"] = json!(entry.first_ts);
            value["last_ts"] = json!(entry.last_ts);
            value["outcomes"] = json!(entry.outcomes);
            value
        })
        .collect();
    json!({ "date": date, "trace_id": trace_id, "sessions": sessions })
}

/// Query-side state: where the ledger lives and a small result cache. The routes are
/// unauthenticated: the ledger holds counts and caller IDs, never request bodies.
pub(crate) struct UsageQuery {
    dir: PathBuf,
    cache: Mutex<HashMap<String, (Instant, Value)>>,
}

pub(crate) enum QueryError {
    BadRequest(String),
    Internal(String),
}

impl UsageQuery {
    pub(crate) fn new(dir: PathBuf) -> Self {
        Self { dir, cache: Mutex::new(HashMap::new()) }
    }

    fn resolve_date(date: Option<&str>) -> Result<String, QueryError> {
        match date {
            None | Some("") => Ok(beijing_date(SystemTime::now())),
            Some(value) if parse_date_days(value).is_some() => Ok(value.to_string()),
            Some(_) => Err(QueryError::BadRequest("date must be YYYY-MM-DD".to_string())),
        }
    }

    fn cached(&self, key: String, date: &str, compute: impl FnOnce() -> Result<Value, QueryError>) -> Result<Value, QueryError> {
        let is_today = date == beijing_date(SystemTime::now());
        if let Ok(cache) = self.cache.lock() {
            if let Some((at, value)) = cache.get(&key) {
                // A past day's file only changes if a request straddled midnight, which lands
                // within seconds of it; a short TTL right after midnight is not worth the code.
                if !is_today || at.elapsed() < TODAY_CACHE_TTL {
                    return Ok(value.clone());
                }
            }
        }
        let value = compute()?;
        if let Ok(mut cache) = self.cache.lock() {
            if cache.len() >= CACHE_MAX_ENTRIES {
                cache.clear();
            }
            cache.insert(key, (Instant::now(), value.clone()));
        }
        Ok(value)
    }

    pub(crate) fn daily(&self, date: Option<&str>, env: Option<&str>) -> Result<Value, QueryError> {
        let date = Self::resolve_date(date)?;
        let env = env.filter(|value| !value.is_empty());
        let key = format!("daily|{date}|{}", env.unwrap_or(""));
        self.cached(key, &date, || {
            let records = read_records(&self.dir, &date).map_err(|err| QueryError::Internal(err.to_string()))?;
            Ok(aggregate_daily(&records, &date, env))
        })
    }

    pub(crate) fn sessions(&self, date: Option<&str>, trace_id: Option<&str>) -> Result<Value, QueryError> {
        let date = Self::resolve_date(date)?;
        let trace_id = trace_id.filter(|value| !value.is_empty());
        let key = format!("sessions|{date}|{}", trace_id.unwrap_or(""));
        self.cached(key, &date, || {
            let records = read_records(&self.dir, &date).map_err(|err| QueryError::Internal(err.to_string()))?;
            Ok(aggregate_sessions(&records, &date, trace_id))
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A request context backed by a real recorder writing into a fresh temp dir.
    pub(crate) fn tracked_context(path: &str, api_format: &'static str) -> (crate::RequestLogContext, PathBuf) {
        let dir = std::env::temp_dir().join(format!("cc-proxy-usage-test-{}", new_boot_id()));
        let recorder = UsageRecorder::spawn(dir.clone(), 0).unwrap();
        let mut context = crate::RequestLogContext::for_test(path, api_format);
        context.usage = Some(std::sync::Arc::new(UsageTracking::new(recorder, UsageMeta { stream: true, ..Default::default() })));
        (context, dir)
    }

    /// Wait for the writer thread to flush `expected` lines, then remove the dir.
    pub(crate) fn take_records(dir: PathBuf, expected: usize) -> Vec<UsageRecord> {
        let date = beijing_date(SystemTime::now());
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let records = read_records(&dir, &date).unwrap_or_default();
            if records.len() >= expected || Instant::now() > deadline {
                std::thread::sleep(Duration::from_millis(50));
                let records = read_records(&dir, &date).unwrap_or_default();
                let _ = std::fs::remove_dir_all(&dir);
                return records;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    pub(crate) fn record(trace_id: Option<&str>, env: Option<&str>, outcome: Outcome, prompt: u64, output: u64) -> UsageRecord {
        UsageRecord {
            ts: "2026-11-11T01:00:00.000Z".to_string(),
            date: "2026-11-11".to_string(),
            boot_id: "b".to_string(),
            req_id: 1,
            path: "/v1/messages".to_string(),
            provider: None,
            api_format: "openai_chat".to_string(),
            stream: true,
            requested_model: Some("claude-haiku".to_string()),
            mapped_model: Some("qwen".to_string()),
            response_model: None,
            outcome,
            status: Some(200),
            latency_ms: 10,
            tokens: TokenCounts { prompt_tokens: prompt, output_tokens: output, cache_read_tokens: 0, cache_creation_tokens: 0 },
            trace_id: trace_id.map(str::to_string),
            trace_type: Some("oa".to_string()),
            cura_session_id: Some("s1".to_string()),
            cura_env: env.map(str::to_string),
        }
    }

    #[test]
    fn beijing_date_rolls_over_at_16_utc() {
        // 2026-11-10T15:59:59Z is 23:59:59 in Beijing; one second later is the 11th.
        let before = UNIX_EPOCH + Duration::from_secs(1_794_326_399);
        let after = UNIX_EPOCH + Duration::from_secs(1_794_326_400);
        assert_eq!(format_utc_ts(before), "2026-11-10T15:59:59.000Z");
        assert_eq!(beijing_date(before), "2026-11-10");
        assert_eq!(beijing_date(after), "2026-11-11");
    }

    #[test]
    fn date_round_trips_through_days() {
        let days = parse_date_days("2024-02-29").unwrap();
        assert_eq!(civil_from_days(days), (2024, 2, 29));
        assert!(parse_date_days("2024-13-01").is_none());
        assert!(parse_date_days("20241301").is_none());
    }

    #[test]
    fn prompt_tokens_always_include_cache() {
        // Chat Completions → Anthropic shape: input excludes cache, so add it back.
        let chat = TokenCounts::from_parts(10, 5, Some(90), Some(3), false);
        assert_eq!(chat.prompt_tokens, 103);
        assert_eq!(chat.total(), 108);
        // Responses API: input already includes cached tokens.
        let responses = TokenCounts::from_responses_usage(&json!({
            "input_tokens": 100, "output_tokens": 5, "input_tokens_details": { "cached_tokens": 90 }
        }));
        assert_eq!(responses.prompt_tokens, 100);
        assert_eq!(responses.cache_read_tokens, 90);
    }

    #[test]
    fn identity_headers_are_trimmed_and_bounded() {
        let mut headers = http::HeaderMap::new();
        headers.insert(TRACE_IDENTIFIER_HEADER, "  alice ".parse().unwrap());
        headers.insert(CURA_SESSION_HEADER, "x".repeat(500).parse().unwrap());
        headers.insert(CURA_ENV_HEADER, "".parse().unwrap());
        let identity = RequestIdentity::from_headers(&headers);
        assert_eq!(identity.trace_id.as_deref(), Some("alice"));
        assert_eq!(identity.cura_session_id.unwrap().len(), IDENTITY_VALUE_MAX_CHARS);
        assert!(identity.cura_env.is_none());
        assert!(identity.trace_type.is_none());
    }

    #[test]
    fn aggregate_splits_users_unattributed_and_outcomes() {
        let records = vec![
            record(Some("alice"), Some("prod"), Outcome::Done, 100, 10),
            record(Some("alice"), Some("prod"), Outcome::ClientAborted, 50, 0),
            record(Some("bob"), Some("test"), Outcome::Done, 20, 2),
            record(None, None, Outcome::UpstreamError, 0, 0),
            record(None, None, Outcome::Done, 7, 1),
        ];
        let all = aggregate_daily(&records, "2026-11-11", None);
        assert_eq!(all["totals"]["requests"], 5);
        assert_eq!(all["totals"]["total_tokens"], 190);
        assert_eq!(all["unattributed"]["requests"], 2);
        assert_eq!(all["unattributed"]["total_tokens"], 8);
        assert_eq!(all["outcomes"]["done"], 3);
        assert_eq!(all["outcomes"]["client_aborted"], 1);
        assert_eq!(all["by_user"][0]["trace_id"], "alice");
        assert_eq!(all["by_user"][0]["total_tokens"], 160);
        assert_eq!(all["by_env"][0]["env"], "prod");

        let prod = aggregate_daily(&records, "2026-11-11", Some("prod"));
        assert_eq!(prod["totals"]["requests"], 2);
        assert_eq!(prod["by_user"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn sessions_group_by_cura_session() {
        let mut other = record(Some("alice"), Some("prod"), Outcome::Done, 1, 1);
        other.cura_session_id = Some("s2".to_string());
        let records = vec![
            record(Some("alice"), Some("prod"), Outcome::Done, 100, 10),
            record(Some("alice"), Some("prod"), Outcome::Done, 100, 10),
            other,
            record(Some("bob"), Some("prod"), Outcome::Done, 5, 5),
        ];
        let value = aggregate_sessions(&records, "2026-11-11", Some("alice"));
        let sessions = value["sessions"].as_array().unwrap();
        assert_eq!(sessions.len(), 2);
        assert_eq!(sessions[0]["session_id"], "s1");
        assert_eq!(sessions[0]["requests"], 2);
    }

    #[test]
    fn writer_appends_and_reader_skips_torn_lines() {
        let dir = std::env::temp_dir().join(format!("cc-proxy-usage-test-{}", new_boot_id()));
        let recorder = UsageRecorder::spawn(dir.clone(), 90).unwrap();
        let tracking = UsageTracking::new(
            recorder,
            UsageMeta { stream: true, identity: RequestIdentity { trace_id: Some("alice".into()), ..Default::default() }, ..Default::default() },
        );
        let tokens = TokenCounts { prompt_tokens: 3, output_tokens: 4, ..Default::default() };
        tracking.settle(9, "/v1/messages", "openai_chat", Instant::now(), Outcome::Done, Some(200), tokens, Some("m"));
        // Second settle must be a no-op: one request, one line.
        tracking.settle(9, "/v1/messages", "openai_chat", Instant::now(), Outcome::ClientAborted, None, tokens, None);
        let date = beijing_date(SystemTime::now());
        let path = dir.join(format!("{date}.jsonl"));
        let deadline = Instant::now() + Duration::from_secs(5);
        while !path.exists() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        std::thread::sleep(Duration::from_millis(50));
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(b"{\"torn\":").unwrap();
        let records = read_records(&dir, &date).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].outcome, Outcome::Done);
        assert_eq!(records[0].trace_id.as_deref(), Some("alice"));
        assert_eq!(records[0].tokens.total(), 7);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn cleanup_removes_only_expired_files() {
        let dir = std::env::temp_dir().join(format!("cc-proxy-usage-cleanup-{}", new_boot_id()));
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["2026-08-01.jsonl", "2026-11-01.jsonl", "notes.txt"] {
            std::fs::write(dir.join(name), b"").unwrap();
        }
        cleanup_expired(&dir, "2026-11-11", 90);
        assert!(!dir.join("2026-08-01.jsonl").exists());
        assert!(dir.join("2026-11-01.jsonl").exists());
        assert!(dir.join("notes.txt").exists());
        let _ = std::fs::remove_dir_all(dir);
    }
}
