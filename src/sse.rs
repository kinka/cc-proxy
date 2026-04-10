pub fn strip_sse_field<'a>(line: &'a str, field: &str) -> Option<&'a str> {
    let prefix = format!("{field}:");
    if !line.starts_with(&prefix) {
        return None;
    }

    Some(line[prefix.len()..].trim_start())
}
